"""
Dynamic adapter loading system for Adaptrix.
"""

import logging
import threading
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import time
from .layer_injector import LayerInjector
from ..adapters.adapter_manager import AdapterManager
from ..utils.config import config
from ..utils.helpers import get_memory_info, timer

logger = logging.getLogger(__name__)


class DynamicLoader:
    """
    Manages dynamic loading and unloading of LoRA adapters.
    
    Features:
    - LRU cache for adapter weights
    - Hot-swapping without model reload
    - Memory monitoring and cleanup
    - Background preloading
    """
    
    def __init__(self, injector: LayerInjector, adapter_manager: AdapterManager):
        """
        Initialize dynamic loader.
        
        Args:
            injector: Layer injector instance
            adapter_manager: Adapter manager instance
        """
        self.injector = injector
        self.adapter_manager = adapter_manager
        
        # Configuration
        self.max_cache_size = config.get('adapters.cache_size', 3)
        self.auto_cleanup = config.get('adapters.auto_cleanup', True)
        self.preload_popular = config.get('adapters.preload_popular', True)
        
        # LRU cache for adapter weights
        self._adapter_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Currently loaded adapters (injected into model)
        self._loaded_adapters: Dict[str, List[int]] = {}
        
        # Usage statistics for preloading
        self._usage_stats: Dict[str, int] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Memory monitoring
        self._memory_threshold_gb = config.get('performance.max_memory_gb', 8)
        
        logger.info("DynamicLoader initialized")
    
    def load_adapter(self, adapter_name: str, layer_indices: Optional[List[int]] = None) -> bool:
        """
        Load and inject an adapter into the model.
        
        Args:
            adapter_name: Name of the adapter to load
            layer_indices: Specific layers to inject into (None for all target layers)
            
        Returns:
            True if loading successful
        """
        with self._lock:
            try:
                with timer(f"Loading adapter {adapter_name}"):
                    # Check if adapter is already loaded
                    if adapter_name in self._loaded_adapters:
                        logger.info(f"Adapter {adapter_name} already loaded")
                        return True
                    
                    # Get adapter data (from cache or storage)
                    adapter_data = self._get_adapter_data(adapter_name)
                    if adapter_data is None:
                        logger.error(f"Failed to get adapter data for {adapter_name}")
                        return False
                    
                    # Determine target layers
                    if layer_indices is None:
                        layer_indices = adapter_data['metadata'].get('target_layers', [])
                    
                    # Check memory before loading
                    if not self._check_memory_availability():
                        logger.warning("Memory threshold exceeded, cleaning up")
                        self._cleanup_unused_adapters()
                    
                    # Inject adapter into each target layer
                    successful_layers = []

                    for layer_idx in layer_indices:
                        if layer_idx in adapter_data['weights']:
                            layer_weights = adapter_data['weights'][layer_idx]

                            # Inject into each module in the layer
                            layer_success = True
                            for module_name, module_data in layer_weights.items():
                                if not self.injector.inject_adapter(adapter_name, layer_idx, module_name, module_data):
                                    logger.error(f"Failed to inject adapter {adapter_name} into layer {layer_idx}, module {module_name}")
                                    layer_success = False

                            if layer_success:
                                successful_layers.append(layer_idx)
                            else:
                                logger.error(f"Failed to inject adapter {adapter_name} into layer {layer_idx}")
                    
                    if successful_layers:
                        self._loaded_adapters[adapter_name] = successful_layers
                        self._update_usage_stats(adapter_name)
                        
                        logger.info(f"Successfully loaded adapter {adapter_name} into layers {successful_layers}")
                        return True
                    else:
                        logger.error(f"Failed to load adapter {adapter_name} into any layer")
                        return False
                        
            except Exception as e:
                logger.error(f"Error loading adapter {adapter_name}: {e}")
                return False
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """
        Unload an adapter from the model.
        
        Args:
            adapter_name: Name of the adapter to unload
            
        Returns:
            True if unloading successful
        """
        with self._lock:
            try:
                if adapter_name not in self._loaded_adapters:
                    logger.warning(f"Adapter {adapter_name} not currently loaded")
                    return False
                
                with timer(f"Unloading adapter {adapter_name}"):
                    # Remove from each layer
                    layer_indices = self._loaded_adapters[adapter_name]

                    for layer_idx in layer_indices:
                        self.injector.remove_adapter(adapter_name)
                    
                    # Remove from loaded adapters
                    del self._loaded_adapters[adapter_name]
                    
                    logger.info(f"Successfully unloaded adapter {adapter_name}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error unloading adapter {adapter_name}: {e}")
                return False
    
    def switch_adapter(self, old_name: str, new_name: str) -> bool:
        """
        Hot-swap one adapter for another.
        
        Args:
            old_name: Name of adapter to unload
            new_name: Name of adapter to load
            
        Returns:
            True if switch successful
        """
        with self._lock:
            try:
                with timer(f"Switching adapter {old_name} -> {new_name}"):
                    # Unload old adapter
                    if old_name in self._loaded_adapters:
                        if not self.unload_adapter(old_name):
                            logger.error(f"Failed to unload adapter {old_name}")
                            return False
                    
                    # Load new adapter
                    if not self.load_adapter(new_name):
                        logger.error(f"Failed to load adapter {new_name}")
                        return False
                    
                    logger.info(f"Successfully switched adapter {old_name} -> {new_name}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error switching adapters: {e}")
                return False
    
    def _get_adapter_data(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get adapter data from cache or load from storage.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Adapter data or None if not found
        """
        # Check cache first
        if adapter_name in self._adapter_cache:
            # Move to end (most recently used)
            self._adapter_cache.move_to_end(adapter_name)
            logger.debug(f"Retrieved adapter {adapter_name} from cache")
            return self._adapter_cache[adapter_name]
        
        # Load from storage
        adapter_data = self.adapter_manager.load_adapter(adapter_name)
        if adapter_data is None:
            return None
        
        # Add to cache
        self._add_to_cache(adapter_name, adapter_data)
        
        return adapter_data
    
    def _add_to_cache(self, adapter_name: str, adapter_data: Dict[str, Any]) -> None:
        """
        Add adapter data to cache with LRU eviction.
        
        Args:
            adapter_name: Name of the adapter
            adapter_data: Adapter data to cache
        """
        # Remove oldest items if cache is full
        while len(self._adapter_cache) >= self.max_cache_size:
            oldest_name, _ = self._adapter_cache.popitem(last=False)
            logger.debug(f"Evicted adapter {oldest_name} from cache")
        
        # Add new item
        self._adapter_cache[adapter_name] = adapter_data
        logger.debug(f"Added adapter {adapter_name} to cache")
    
    def _update_usage_stats(self, adapter_name: str) -> None:
        """Update usage statistics for an adapter."""
        self._usage_stats[adapter_name] = self._usage_stats.get(adapter_name, 0) + 1
    
    def _check_memory_availability(self) -> bool:
        """
        Check if there's enough memory available.
        
        Returns:
            True if memory is available
        """
        try:
            memory_info = get_memory_info()
            
            # Check system memory
            if memory_info.get('system_available', 0) < 1.0:  # Less than 1GB available
                return False
            
            # Check GPU memory if available
            if 'gpu_allocated' in memory_info:
                gpu_usage_gb = memory_info['gpu_allocated']
                if gpu_usage_gb > self._memory_threshold_gb * 0.8:  # 80% of threshold
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking memory availability: {e}")
            return True  # Assume available if check fails
    
    def cleanup_unused_adapters(self) -> int:
        """
        Clean up unused adapters from cache.
        
        Returns:
            Number of adapters cleaned up
        """
        with self._lock:
            return self._cleanup_unused_adapters()
    
    def _cleanup_unused_adapters(self) -> int:
        """Internal cleanup method."""
        cleaned_count = 0
        
        # Find adapters in cache but not loaded
        cached_names = list(self._adapter_cache.keys())
        
        for adapter_name in cached_names:
            if adapter_name not in self._loaded_adapters:
                del self._adapter_cache[adapter_name]
                cleaned_count += 1
                logger.debug(f"Cleaned up cached adapter {adapter_name}")
        
        return cleaned_count
    
    def preload_adapters(self, adapter_names: List[str]) -> int:
        """
        Preload adapters into cache.
        
        Args:
            adapter_names: List of adapter names to preload
            
        Returns:
            Number of adapters successfully preloaded
        """
        with self._lock:
            preloaded_count = 0
            
            for adapter_name in adapter_names:
                if adapter_name not in self._adapter_cache:
                    try:
                        adapter_data = self.adapter_manager.load_adapter(adapter_name)
                        if adapter_data is not None:
                            self._add_to_cache(adapter_name, adapter_data)
                            preloaded_count += 1
                            logger.debug(f"Preloaded adapter {adapter_name}")
                    except Exception as e:
                        logger.warning(f"Failed to preload adapter {adapter_name}: {e}")
            
            logger.info(f"Preloaded {preloaded_count} adapters")
            return preloaded_count
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        with self._lock:
            # Get system memory info
            memory_info = get_memory_info()
            
            # Get injector memory usage
            injector_memory = self.injector.get_memory_usage()
            
            # Calculate cache memory usage (rough estimate)
            cache_memory_mb = 0
            for adapter_data in self._adapter_cache.values():
                # Estimate based on number of parameters
                for layer_weights in adapter_data.get('weights', {}).values():
                    for module_weights in layer_weights.values():
                        if 'lora_A' in module_weights and 'lora_B' in module_weights:
                            lora_A = module_weights['lora_A']
                            lora_B = module_weights['lora_B']
                            
                            if hasattr(lora_A, 'numel') and hasattr(lora_B, 'numel'):
                                params = lora_A.numel() + lora_B.numel()
                                cache_memory_mb += params * 4 / (1024**2)  # Assume fp32
            
            return {
                'system_memory': memory_info,
                'injector_memory': injector_memory,
                'cache_memory_mb': cache_memory_mb,
                'cache_memory_gb': cache_memory_mb / 1024,
                'cached_adapters': len(self._adapter_cache),
                'loaded_adapters': len(self._loaded_adapters),
                'memory_threshold_gb': self._memory_threshold_gb
            }
    
    def get_loaded_adapters(self) -> Dict[str, List[int]]:
        """
        Get currently loaded adapters.
        
        Returns:
            Dictionary mapping adapter names to layer indices
        """
        with self._lock:
            return self._loaded_adapters.copy()
    
    def get_cached_adapters(self) -> List[str]:
        """
        Get list of cached adapter names.
        
        Returns:
            List of cached adapter names
        """
        with self._lock:
            return list(self._adapter_cache.keys())
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get adapter usage statistics.
        
        Returns:
            Dictionary mapping adapter names to usage counts
        """
        with self._lock:
            return self._usage_stats.copy()
    
    def clear_all(self) -> None:
        """Clear all loaded adapters and cache."""
        with self._lock:
            # Unload all adapters
            for adapter_name in list(self._loaded_adapters.keys()):
                self.unload_adapter(adapter_name)
            
            # Clear cache
            self._adapter_cache.clear()
            
            # Clear usage stats
            self._usage_stats.clear()
            
            logger.info("Cleared all adapters and cache")
