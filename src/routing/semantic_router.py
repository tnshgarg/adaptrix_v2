"""
Advanced Semantic Router for intelligent adapter selection.
Uses sentence-transformers for semantic similarity matching and query classification.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class AdapterProfile:
    """Profile for an adapter including its capabilities and embeddings."""
    name: str
    description: str
    capabilities: List[str]
    example_queries: List[str]
    embedding: Optional[np.ndarray] = None
    performance_metrics: Dict[str, float] = None
    usage_count: int = 0
    last_used: float = 0.0


@dataclass
class RoutingResult:
    """Result of adapter routing decision."""
    primary_adapter: str
    confidence: float
    secondary_adapters: List[Tuple[str, float]]
    reasoning: str
    query_embedding: np.ndarray
    processing_time: float


class SemanticRouter:
    """
    Advanced semantic router using sentence transformers for intelligent adapter selection.
    
    Features:
    - Semantic similarity matching with cosine similarity
    - Multi-adapter ensemble voting
    - Dynamic confidence thresholds
    - Query classification with learned patterns
    - Usage-based adapter ranking
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 confidence_threshold: float = 0.7,
                 max_secondary_adapters: int = 2,
                 cache_embeddings: bool = True):
        """
        Initialize the semantic router.
        
        Args:
            model_name: Sentence transformer model name
            confidence_threshold: Minimum confidence for adapter selection
            max_secondary_adapters: Maximum number of secondary adapters to suggest
            cache_embeddings: Whether to cache query embeddings
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_secondary_adapters = max_secondary_adapters
        self.cache_embeddings = cache_embeddings
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Adapter profiles and embeddings
        self.adapter_profiles: Dict[str, AdapterProfile] = {}
        self.adapter_embeddings: Dict[str, np.ndarray] = {}
        
        # Query classification
        self.query_clusters = None
        self.cluster_adapter_mapping = {}
        self.scaler = StandardScaler()
        
        # Caching
        self.embedding_cache = {} if cache_embeddings else None
        self.routing_history = []
        
        # Performance tracking
        self.routing_stats = {
            'total_queries': 0,
            'successful_routes': 0,
            'fallback_routes': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info("Semantic router initialized successfully")
    
    def register_adapter(self, 
                        name: str, 
                        description: str, 
                        capabilities: List[str],
                        example_queries: List[str],
                        performance_metrics: Dict[str, float] = None) -> bool:
        """
        Register an adapter with its semantic profile.
        
        Args:
            name: Adapter name
            description: Detailed description of adapter capabilities
            capabilities: List of capability keywords
            example_queries: Example queries this adapter handles well
            performance_metrics: Performance metrics (accuracy, latency, etc.)
        
        Returns:
            True if registration successful
        """
        try:
            logger.info(f"Registering adapter: {name}")
            
            # Create adapter profile
            profile = AdapterProfile(
                name=name,
                description=description,
                capabilities=capabilities,
                example_queries=example_queries,
                performance_metrics=performance_metrics or {},
                usage_count=0,
                last_used=0.0
            )
            
            # Generate semantic embedding
            # Combine description, capabilities, and example queries
            semantic_text = f"{description}. Capabilities: {', '.join(capabilities)}. Examples: {'. '.join(example_queries[:3])}"
            embedding = self.encoder.encode(semantic_text, convert_to_tensor=False)
            
            profile.embedding = embedding
            self.adapter_profiles[name] = profile
            self.adapter_embeddings[name] = embedding
            
            logger.info(f"Adapter {name} registered with embedding shape: {embedding.shape}")
            
            # Update query clustering if we have enough adapters
            if len(self.adapter_profiles) >= 2:
                self._update_query_clustering()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register adapter {name}: {e}")
            return False
    
    def route_query(self, query: str, available_adapters: List[str] = None) -> RoutingResult:
        """
        Route a query to the most appropriate adapter(s).
        
        Args:
            query: Input query to route
            available_adapters: List of available adapters (None for all)
        
        Returns:
            RoutingResult with primary adapter and alternatives
        """
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Filter available adapters
            if available_adapters is None:
                available_adapters = list(self.adapter_profiles.keys())
            
            available_profiles = {
                name: profile for name, profile in self.adapter_profiles.items()
                if name in available_adapters
            }
            
            if not available_profiles:
                raise ValueError("No available adapters for routing")
            
            # Calculate similarities
            similarities = self._calculate_similarities(query_embedding, available_profiles)
            
            # Apply ensemble voting
            ensemble_scores = self._ensemble_voting(similarities, available_profiles, query)
            
            # Select primary adapter
            primary_adapter, primary_confidence = max(ensemble_scores.items(), key=lambda x: x[1])
            
            # Select secondary adapters
            secondary_adapters = [
                (name, score) for name, score in ensemble_scores.items()
                if name != primary_adapter and score > self.confidence_threshold * 0.7
            ]
            secondary_adapters.sort(key=lambda x: x[1], reverse=True)
            secondary_adapters = secondary_adapters[:self.max_secondary_adapters]
            
            # Generate reasoning
            reasoning = self._generate_reasoning(query, primary_adapter, primary_confidence, similarities)
            
            # Update usage statistics
            self._update_usage_stats(primary_adapter)
            
            # Create result
            processing_time = time.time() - start_time
            result = RoutingResult(
                primary_adapter=primary_adapter,
                confidence=primary_confidence,
                secondary_adapters=secondary_adapters,
                reasoning=reasoning,
                query_embedding=query_embedding,
                processing_time=processing_time
            )
            
            # Update statistics
            self._update_routing_stats(result)
            
            # Store in history
            self.routing_history.append({
                'query': query,
                'result': result,
                'timestamp': time.time()
            })
            
            # Keep history manageable
            if len(self.routing_history) > 1000:
                self.routing_history = self.routing_history[-500:]
            
            logger.info(f"Routed query to {primary_adapter} with confidence {primary_confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to route query: {e}")
            # Return fallback result
            fallback_adapter = available_adapters[0] if available_adapters else "base_model"
            return RoutingResult(
                primary_adapter=fallback_adapter,
                confidence=0.0,
                secondary_adapters=[],
                reasoning=f"Fallback routing due to error: {str(e)}",
                query_embedding=np.zeros(384),  # Default embedding size
                processing_time=time.time() - start_time
            )
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query with caching."""
        if self.embedding_cache is not None and query in self.embedding_cache:
            return self.embedding_cache[query]
        
        embedding = self.encoder.encode(query, convert_to_tensor=False)
        
        if self.embedding_cache is not None:
            self.embedding_cache[query] = embedding
            
            # Limit cache size
            if len(self.embedding_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.embedding_cache.keys())[:100]
                for key in oldest_keys:
                    del self.embedding_cache[key]
        
        return embedding
    
    def _calculate_similarities(self, 
                              query_embedding: np.ndarray, 
                              available_profiles: Dict[str, AdapterProfile]) -> Dict[str, float]:
        """Calculate cosine similarities between query and adapters."""
        similarities = {}
        
        for name, profile in available_profiles.items():
            if profile.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    profile.embedding.reshape(1, -1)
                )[0][0]
                similarities[name] = float(similarity)
            else:
                similarities[name] = 0.0
        
        return similarities
    
    def _ensemble_voting(self, 
                        similarities: Dict[str, float],
                        available_profiles: Dict[str, AdapterProfile],
                        query: str) -> Dict[str, float]:
        """Apply ensemble voting with multiple factors."""
        ensemble_scores = {}
        
        for name, similarity in similarities.items():
            profile = available_profiles[name]
            
            # Base similarity score
            score = similarity
            
            # Performance boost
            if profile.performance_metrics:
                accuracy = profile.performance_metrics.get('accuracy', 0.5)
                latency_penalty = 1.0 - min(profile.performance_metrics.get('latency_ms', 100) / 1000, 0.5)
                performance_boost = (accuracy * 0.7 + latency_penalty * 0.3) * 0.2
                score += performance_boost
            
            # Usage frequency boost (popular adapters get slight boost)
            if profile.usage_count > 0:
                usage_boost = min(np.log(profile.usage_count + 1) / 10, 0.1)
                score += usage_boost
            
            # Recency boost (recently used adapters get slight boost)
            if profile.last_used > 0:
                recency = time.time() - profile.last_used
                if recency < 3600:  # Within last hour
                    recency_boost = (3600 - recency) / 3600 * 0.05
                    score += recency_boost
            
            # Capability keyword matching
            query_lower = query.lower()
            capability_matches = sum(1 for cap in profile.capabilities if cap.lower() in query_lower)
            if capability_matches > 0:
                capability_boost = min(capability_matches * 0.1, 0.3)
                score += capability_boost
            
            ensemble_scores[name] = min(score, 1.0)  # Cap at 1.0
        
        return ensemble_scores

    def _generate_reasoning(self,
                           query: str,
                           primary_adapter: str,
                           confidence: float,
                           similarities: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the routing decision."""
        profile = self.adapter_profiles[primary_adapter]

        reasoning_parts = [
            f"Selected '{primary_adapter}' with {confidence:.1%} confidence."
        ]

        # Add similarity reasoning
        similarity = similarities.get(primary_adapter, 0.0)
        if similarity > 0.8:
            reasoning_parts.append(f"High semantic similarity ({similarity:.1%}) to adapter capabilities.")
        elif similarity > 0.6:
            reasoning_parts.append(f"Good semantic match ({similarity:.1%}) with adapter domain.")
        else:
            reasoning_parts.append(f"Moderate similarity ({similarity:.1%}), selected based on ensemble factors.")

        # Add capability matching
        query_lower = query.lower()
        matched_capabilities = [cap for cap in profile.capabilities if cap.lower() in query_lower]
        if matched_capabilities:
            reasoning_parts.append(f"Matched capabilities: {', '.join(matched_capabilities[:3])}.")

        # Add performance reasoning
        if profile.performance_metrics:
            accuracy = profile.performance_metrics.get('accuracy', 0)
            if accuracy > 0.8:
                reasoning_parts.append(f"High accuracy adapter ({accuracy:.1%}).")

        return " ".join(reasoning_parts)

    def _update_usage_stats(self, adapter_name: str):
        """Update usage statistics for an adapter."""
        if adapter_name in self.adapter_profiles:
            profile = self.adapter_profiles[adapter_name]
            profile.usage_count += 1
            profile.last_used = time.time()

    def _update_routing_stats(self, result: RoutingResult):
        """Update overall routing statistics."""
        self.routing_stats['total_queries'] += 1

        if result.confidence > self.confidence_threshold:
            self.routing_stats['successful_routes'] += 1
        else:
            self.routing_stats['fallback_routes'] += 1

        # Update running averages
        total = self.routing_stats['total_queries']
        self.routing_stats['avg_confidence'] = (
            (self.routing_stats['avg_confidence'] * (total - 1) + result.confidence) / total
        )
        self.routing_stats['avg_processing_time'] = (
            (self.routing_stats['avg_processing_time'] * (total - 1) + result.processing_time) / total
        )

    def _update_query_clustering(self):
        """Update query clustering for improved routing."""
        if len(self.adapter_profiles) < 2:
            return

        try:
            # Collect all adapter embeddings
            embeddings = []
            adapter_names = []

            for name, profile in self.adapter_profiles.items():
                if profile.embedding is not None:
                    embeddings.append(profile.embedding)
                    adapter_names.append(name)

            if len(embeddings) < 2:
                return

            embeddings = np.array(embeddings)

            # Perform clustering
            n_clusters = min(len(embeddings), 5)  # Max 5 clusters
            self.query_clusters = KMeans(n_clusters=n_clusters, random_state=42)
            self.query_clusters.fit(embeddings)

            # Map clusters to adapters
            self.cluster_adapter_mapping = {}
            for i, adapter_name in enumerate(adapter_names):
                cluster_id = self.query_clusters.labels_[i]
                if cluster_id not in self.cluster_adapter_mapping:
                    self.cluster_adapter_mapping[cluster_id] = []
                self.cluster_adapter_mapping[cluster_id].append(adapter_name)

            logger.info(f"Updated query clustering with {n_clusters} clusters")

        except Exception as e:
            logger.warning(f"Failed to update query clustering: {e}")

    def get_cluster_prediction(self, query: str) -> List[str]:
        """Get adapter suggestions based on query clustering."""
        if self.query_clusters is None:
            return []

        try:
            query_embedding = self._get_query_embedding(query)
            cluster_id = self.query_clusters.predict(query_embedding.reshape(1, -1))[0]
            return self.cluster_adapter_mapping.get(cluster_id, [])
        except Exception as e:
            logger.warning(f"Failed to get cluster prediction: {e}")
            return []

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        stats = self.routing_stats.copy()

        # Add adapter-specific stats
        adapter_stats = {}
        for name, profile in self.adapter_profiles.items():
            adapter_stats[name] = {
                'usage_count': profile.usage_count,
                'last_used': profile.last_used,
                'performance_metrics': profile.performance_metrics
            }

        stats['adapter_stats'] = adapter_stats
        stats['total_adapters'] = len(self.adapter_profiles)
        stats['cache_size'] = len(self.embedding_cache) if self.embedding_cache else 0

        return stats

    def update_confidence_threshold(self, new_threshold: float):
        """Dynamically update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")

    def save_profiles(self, filepath: str) -> bool:
        """Save adapter profiles to file."""
        try:
            profiles_data = {}
            for name, profile in self.adapter_profiles.items():
                profiles_data[name] = {
                    'name': profile.name,
                    'description': profile.description,
                    'capabilities': profile.capabilities,
                    'example_queries': profile.example_queries,
                    'embedding': profile.embedding.tolist() if profile.embedding is not None else None,
                    'performance_metrics': profile.performance_metrics,
                    'usage_count': profile.usage_count,
                    'last_used': profile.last_used
                }

            with open(filepath, 'w') as f:
                json.dump(profiles_data, f, indent=2)

            logger.info(f"Saved {len(profiles_data)} adapter profiles to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
            return False

    def load_profiles(self, filepath: str) -> bool:
        """Load adapter profiles from file."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Profile file not found: {filepath}")
                return False

            with open(filepath, 'r') as f:
                profiles_data = json.load(f)

            for name, data in profiles_data.items():
                profile = AdapterProfile(
                    name=data['name'],
                    description=data['description'],
                    capabilities=data['capabilities'],
                    example_queries=data['example_queries'],
                    embedding=np.array(data['embedding']) if data['embedding'] else None,
                    performance_metrics=data.get('performance_metrics', {}),
                    usage_count=data.get('usage_count', 0),
                    last_used=data.get('last_used', 0.0)
                )

                self.adapter_profiles[name] = profile
                if profile.embedding is not None:
                    self.adapter_embeddings[name] = profile.embedding

            # Update clustering
            if len(self.adapter_profiles) >= 2:
                self._update_query_clustering()

            logger.info(f"Loaded {len(profiles_data)} adapter profiles from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return False

    def clear_cache(self):
        """Clear embedding cache."""
        if self.embedding_cache:
            self.embedding_cache.clear()
            logger.info("Cleared embedding cache")

    def get_similar_queries(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar queries from routing history."""
        if not self.routing_history:
            return []

        try:
            query_embedding = self._get_query_embedding(query)
            similarities = []

            for entry in self.routing_history[-100:]:  # Check last 100 queries
                hist_embedding = entry['result'].query_embedding
                if hist_embedding is not None and len(hist_embedding) > 0:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        hist_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((entry['query'], float(similarity)))

            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.warning(f"Failed to find similar queries: {e}")
            return []
