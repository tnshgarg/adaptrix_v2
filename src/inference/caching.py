"""
Caching System for Adaptrix Inference.

This module provides comprehensive caching capabilities including:
- Response caching for repeated queries
- KV-cache management for transformer models
- Prefix caching for common prompt patterns
- Memory-efficient cache eviction policies
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import threading
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of caches supported."""
    RESPONSE = "response"
    KV_CACHE = "kv_cache"
    PREFIX = "prefix"
    EMBEDDING = "embedding"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class BaseCache:
    """Base cache implementation with common functionality."""
    
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: Optional[float] = None,
        max_memory_mb: Optional[float] = None
    ):
        """
        Initialize base cache.
        
        Args:
            max_size: Maximum number of entries
            eviction_policy: Policy for evicting entries
            default_ttl: Default time-to-live in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                return 1024  # Default estimate
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        size_exceeded = len(self._cache) >= self.max_size
        memory_exceeded = (
            self.max_memory_bytes is not None and 
            self._current_memory >= self.max_memory_bytes
        )
        return size_exceeded or memory_exceeded
    
    def _evict_entries(self):
        """Evict entries based on policy."""
        if not self._cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            key_to_remove = next(iter(self._cache))
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key_to_remove = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                key_to_remove = expired_keys[0]
            else:
                key_to_remove = next(iter(self._cache))
        else:  # FIFO
            key_to_remove = next(iter(self._cache))
        
        self._remove_entry(key_to_remove)
        self.stats["evictions"] += 1
    
    def _remove_entry(self, key: str):
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self.stats["total_requests"] += 1
            
            if key not in self._cache:
                self.stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None
            
            # Update access and move to end (for LRU)
            entry.update_access()
            self._cache.move_to_end(key)
            
            self.stats["hits"] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Check if single entry exceeds memory limit
            if (self.max_memory_bytes is not None and 
                size_bytes > self.max_memory_bytes):
                logger.warning(f"Cache entry too large: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries if necessary
            while self._should_evict():
                self._evict_entries()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._cache[key] = entry
            self._current_memory += size_bytes
            
            return True
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self.stats["hits"] / self.stats["total_requests"] 
                if self.stats["total_requests"] > 0 else 0
            )
            
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "current_memory_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024) if self.max_memory_bytes else None,
                "eviction_policy": self.eviction_policy.value
            }


class ResponseCache(BaseCache):
    """Cache for storing generated responses."""
    
    def __init__(self, **kwargs):
        """Initialize response cache."""
        super().__init__(**kwargs)
        self.cache_type = CacheType.RESPONSE
    
    def get_response(
        self, 
        prompt: str, 
        generation_config: Dict[str, Any],
        adapter_name: Optional[str] = None
    ) -> Optional[str]:
        """Get cached response for prompt and config."""
        key = self._generate_key(prompt, generation_config, adapter_name)
        return self.get(key)
    
    def cache_response(
        self, 
        prompt: str, 
        generation_config: Dict[str, Any],
        response: str,
        adapter_name: Optional[str] = None,
        ttl: Optional[float] = None
    ) -> bool:
        """Cache response for prompt and config."""
        key = self._generate_key(prompt, generation_config, adapter_name)
        return self.put(key, response, ttl)


class EmbeddingCache(BaseCache):
    """Cache for storing text embeddings."""
    
    def __init__(self, **kwargs):
        """Initialize embedding cache."""
        super().__init__(**kwargs)
        self.cache_type = CacheType.EMBEDDING
    
    def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._generate_key(text, model_name)
        return self.get(key)
    
    def cache_embedding(
        self, 
        text: str, 
        model_name: str, 
        embedding: List[float],
        ttl: Optional[float] = None
    ) -> bool:
        """Cache embedding for text."""
        key = self._generate_key(text, model_name)
        return self.put(key, embedding, ttl)


class PrefixCache(BaseCache):
    """Cache for storing common prompt prefixes."""
    
    def __init__(self, **kwargs):
        """Initialize prefix cache."""
        super().__init__(**kwargs)
        self.cache_type = CacheType.PREFIX
    
    def get_prefix_data(self, prefix: str) -> Optional[Any]:
        """Get cached data for prefix."""
        key = self._generate_key(prefix)
        return self.get(key)
    
    def cache_prefix_data(
        self, 
        prefix: str, 
        data: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Cache data for prefix."""
        key = self._generate_key(prefix)
        return self.put(key, data, ttl)


class CacheManager:
    """
    Manages multiple cache instances for different purposes.
    
    Provides a unified interface for all caching needs in Adaptrix.
    """
    
    def __init__(
        self,
        response_cache_config: Optional[Dict[str, Any]] = None,
        embedding_cache_config: Optional[Dict[str, Any]] = None,
        prefix_cache_config: Optional[Dict[str, Any]] = None,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            response_cache_config: Configuration for response cache
            embedding_cache_config: Configuration for embedding cache
            prefix_cache_config: Configuration for prefix cache
            enable_persistence: Whether to persist caches to disk
            persistence_path: Path for cache persistence
        """
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path) if persistence_path else Path("cache")
        
        # Initialize caches
        self.response_cache = ResponseCache(**(response_cache_config or {}))
        self.embedding_cache = EmbeddingCache(**(embedding_cache_config or {}))
        self.prefix_cache = PrefixCache(**(prefix_cache_config or {}))
        
        # Cache registry
        self.caches = {
            CacheType.RESPONSE: self.response_cache,
            CacheType.EMBEDDING: self.embedding_cache,
            CacheType.PREFIX: self.prefix_cache
        }
        
        # Load persisted caches
        if self.enable_persistence:
            self._load_caches()
        
        logger.info("Cache manager initialized with all cache types")
    
    def get_cache(self, cache_type: CacheType) -> BaseCache:
        """Get cache instance by type."""
        return self.caches[cache_type]
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("All caches cleared")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        total_hits = 0
        total_requests = 0
        
        for cache_type, cache in self.caches.items():
            cache_stats = cache.get_stats()
            stats[cache_type.value] = cache_stats
            total_hits += cache_stats["hits"]
            total_requests += cache_stats["total_requests"]
        
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        stats["global"] = {
            "total_hits": total_hits,
            "total_requests": total_requests,
            "global_hit_rate": global_hit_rate
        }
        
        return stats
    
    def _save_caches(self):
        """Save caches to disk."""
        if not self.enable_persistence:
            return
        
        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            
            for cache_type, cache in self.caches.items():
                cache_file = self.persistence_path / f"{cache_type.value}_cache.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache._cache, f)
            
            logger.info(f"Caches saved to {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Failed to save caches: {e}")
    
    def _load_caches(self):
        """Load caches from disk."""
        if not self.enable_persistence or not self.persistence_path.exists():
            return
        
        try:
            for cache_type, cache in self.caches.items():
                cache_file = self.persistence_path / f"{cache_type.value}_cache.pkl"
                
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        cache._cache.update(cached_data)
                    
                    logger.info(f"Loaded {cache_type.value} cache with {len(cached_data)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load caches: {e}")
    
    def cleanup(self):
        """Cleanup cache manager."""
        if self.enable_persistence:
            self._save_caches()
        
        self.clear_all_caches()
        logger.info("Cache manager cleaned up")


# Convenience functions for creating cache configurations
def create_cache_config(
    max_size: int = 1000,
    eviction_policy: str = "lru",
    default_ttl: Optional[float] = None,
    max_memory_mb: Optional[float] = None
) -> Dict[str, Any]:
    """Create cache configuration dictionary."""
    return {
        "max_size": max_size,
        "eviction_policy": EvictionPolicy(eviction_policy.lower()),
        "default_ttl": default_ttl,
        "max_memory_mb": max_memory_mb
    }


def create_default_cache_manager(
    response_cache_size: int = 1000,
    embedding_cache_size: int = 5000,
    prefix_cache_size: int = 100,
    enable_persistence: bool = False
) -> CacheManager:
    """Create cache manager with default configurations."""
    return CacheManager(
        response_cache_config=create_cache_config(max_size=response_cache_size),
        embedding_cache_config=create_cache_config(max_size=embedding_cache_size),
        prefix_cache_config=create_cache_config(max_size=prefix_cache_size),
        enable_persistence=enable_persistence
    )
