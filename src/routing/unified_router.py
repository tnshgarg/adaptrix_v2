"""
Unified router that combines semantic and keyword-based routing.
Provides intelligent adapter selection with multiple routing strategies.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from .semantic_router import SemanticRouter, RoutingResult
from .keyword_router import KeywordRouter, KeywordRoutingResult

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies."""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


@dataclass
class UnifiedRoutingResult:
    """Result of unified routing with multiple strategies."""
    primary_adapter: str
    confidence: float
    secondary_adapters: List[Tuple[str, float]]
    strategy_used: RoutingStrategy
    semantic_result: Optional[RoutingResult] = None
    keyword_result: Optional[KeywordRoutingResult] = None
    reasoning: str = ""
    processing_time: float = 0.0


class UnifiedRouter:
    """
    Unified router combining semantic and keyword-based routing strategies.
    
    Features:
    - Multiple routing strategies (semantic, keyword, hybrid, ensemble)
    - Adaptive strategy selection based on query characteristics
    - Fallback mechanisms for robustness
    - Performance monitoring and optimization
    """
    
    def __init__(self,
                 semantic_model: str = "all-MiniLM-L6-v2",
                 default_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3,
                 confidence_threshold: float = 0.6):
        """
        Initialize unified router.
        
        Args:
            semantic_model: Sentence transformer model for semantic routing
            default_strategy: Default routing strategy
            semantic_weight: Weight for semantic routing in hybrid mode
            keyword_weight: Weight for keyword routing in hybrid mode
            confidence_threshold: Minimum confidence threshold
        """
        self.default_strategy = default_strategy
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.confidence_threshold = confidence_threshold
        
        # Initialize sub-routers
        self.semantic_router = SemanticRouter(
            model_name=semantic_model,
            confidence_threshold=confidence_threshold
        )
        
        self.keyword_router = KeywordRouter(
            confidence_threshold=confidence_threshold * 0.8  # Lower threshold for keywords
        )
        
        # Performance tracking
        self.strategy_performance = {
            strategy: {'success_count': 0, 'total_count': 0, 'avg_confidence': 0.0}
            for strategy in RoutingStrategy
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'semantic_min_confidence': 0.5,
            'keyword_min_matches': 2,
            'query_length_threshold': 50
        }
        
        logger.info(f"Unified router initialized with strategy: {default_strategy}")
    
    def register_adapter(self,
                        name: str,
                        description: str,
                        capabilities: List[str],
                        example_queries: List[str],
                        keywords: List[str] = None,
                        performance_metrics: Dict[str, float] = None) -> bool:
        """
        Register an adapter with both semantic and keyword profiles.
        
        Args:
            name: Adapter name
            description: Detailed description
            capabilities: List of capabilities
            example_queries: Example queries
            keywords: Keywords for keyword routing (auto-generated if None)
            performance_metrics: Performance metrics
            
        Returns:
            True if registration successful
        """
        try:
            # Register with semantic router
            semantic_success = self.semantic_router.register_adapter(
                name=name,
                description=description,
                capabilities=capabilities,
                example_queries=example_queries,
                performance_metrics=performance_metrics
            )
            
            # Auto-generate keywords if not provided
            if keywords is None:
                keywords = capabilities + self._extract_keywords_from_examples(example_queries)
            
            # Register with keyword router
            from .keyword_router import KeywordRule
            keyword_rule = KeywordRule(
                adapter_name=name,
                keywords=keywords,
                weight=1.0
            )
            keyword_success = self.keyword_router.add_routing_rule(keyword_rule)
            
            success = semantic_success and keyword_success
            
            if success:
                logger.info(f"Successfully registered adapter: {name}")
            else:
                logger.warning(f"Partial registration failure for adapter: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to register adapter {name}: {e}")
            return False
    
    def route_query(self,
                   query: str,
                   available_adapters: List[str] = None,
                   strategy: RoutingStrategy = None) -> UnifiedRoutingResult:
        """
        Route query using specified or default strategy.
        
        Args:
            query: Input query to route
            available_adapters: List of available adapters
            strategy: Routing strategy to use (None for default)
            
        Returns:
            UnifiedRoutingResult with routing decision
        """
        start_time = time.time()
        
        if strategy is None:
            strategy = self._select_adaptive_strategy(query) if self.default_strategy == RoutingStrategy.ADAPTIVE else self.default_strategy
        
        try:
            result = self._route_with_strategy(query, available_adapters, strategy)
            result.processing_time = time.time() - start_time
            
            # Update performance tracking
            self._update_strategy_performance(strategy, result.confidence)
            
            logger.info(f"Routed query using {strategy.value} to {result.primary_adapter} (confidence: {result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Routing failed with strategy {strategy}: {e}")
            
            # Fallback to keyword routing
            try:
                keyword_result = self.keyword_router.route_query(query, available_adapters)
                return UnifiedRoutingResult(
                    primary_adapter=keyword_result.primary_adapter,
                    confidence=keyword_result.confidence,
                    secondary_adapters=[],
                    strategy_used=RoutingStrategy.KEYWORD_ONLY,
                    keyword_result=keyword_result,
                    reasoning=f"Fallback to keyword routing: {keyword_result.reasoning}",
                    processing_time=time.time() - start_time
                )
            except Exception as fallback_error:
                logger.error(f"Fallback routing also failed: {fallback_error}")
                
                # Final fallback
                fallback_adapter = available_adapters[0] if available_adapters else "base_model"
                return UnifiedRoutingResult(
                    primary_adapter=fallback_adapter,
                    confidence=0.0,
                    secondary_adapters=[],
                    strategy_used=RoutingStrategy.KEYWORD_ONLY,
                    reasoning=f"Emergency fallback due to routing failure",
                    processing_time=time.time() - start_time
                )
    
    def _route_with_strategy(self,
                           query: str,
                           available_adapters: List[str],
                           strategy: RoutingStrategy) -> UnifiedRoutingResult:
        """Route query with specific strategy."""
        
        if strategy == RoutingStrategy.SEMANTIC_ONLY:
            return self._route_semantic_only(query, available_adapters)
        
        elif strategy == RoutingStrategy.KEYWORD_ONLY:
            return self._route_keyword_only(query, available_adapters)
        
        elif strategy == RoutingStrategy.HYBRID:
            return self._route_hybrid(query, available_adapters)
        
        elif strategy == RoutingStrategy.ENSEMBLE:
            return self._route_ensemble(query, available_adapters)

        elif strategy == RoutingStrategy.ADAPTIVE:
            # Adaptive strategy selects the best strategy for the query
            selected_strategy = self._select_adaptive_strategy(query)
            return self._route_with_strategy(query, available_adapters, selected_strategy)

        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")
    
    def _route_semantic_only(self, query: str, available_adapters: List[str]) -> UnifiedRoutingResult:
        """Route using semantic similarity only."""
        semantic_result = self.semantic_router.route_query(query, available_adapters)
        
        return UnifiedRoutingResult(
            primary_adapter=semantic_result.primary_adapter,
            confidence=semantic_result.confidence,
            secondary_adapters=semantic_result.secondary_adapters,
            strategy_used=RoutingStrategy.SEMANTIC_ONLY,
            semantic_result=semantic_result,
            reasoning=semantic_result.reasoning
        )
    
    def _route_keyword_only(self, query: str, available_adapters: List[str]) -> UnifiedRoutingResult:
        """Route using keyword matching only."""
        keyword_result = self.keyword_router.route_query(query, available_adapters)
        
        return UnifiedRoutingResult(
            primary_adapter=keyword_result.primary_adapter,
            confidence=keyword_result.confidence,
            secondary_adapters=[],
            strategy_used=RoutingStrategy.KEYWORD_ONLY,
            keyword_result=keyword_result,
            reasoning=keyword_result.reasoning
        )
    
    def _route_hybrid(self, query: str, available_adapters: List[str]) -> UnifiedRoutingResult:
        """Route using hybrid semantic + keyword approach."""
        # Get results from both routers
        semantic_result = self.semantic_router.route_query(query, available_adapters)
        keyword_result = self.keyword_router.route_query(query, available_adapters)
        
        # Combine scores
        adapter_scores = {}
        
        # Add semantic scores
        adapter_scores[semantic_result.primary_adapter] = semantic_result.confidence * self.semantic_weight
        for adapter, score in semantic_result.secondary_adapters:
            if adapter in adapter_scores:
                adapter_scores[adapter] += score * self.semantic_weight
            else:
                adapter_scores[adapter] = score * self.semantic_weight
        
        # Add keyword scores
        if keyword_result.confidence > 0:
            if keyword_result.primary_adapter in adapter_scores:
                adapter_scores[keyword_result.primary_adapter] += keyword_result.confidence * self.keyword_weight
            else:
                adapter_scores[keyword_result.primary_adapter] = keyword_result.confidence * self.keyword_weight
        
        # Select best adapter
        if adapter_scores:
            primary_adapter = max(adapter_scores.keys(), key=lambda x: adapter_scores[x])
            primary_confidence = adapter_scores[primary_adapter]
            
            # Get secondary adapters
            secondary_adapters = [
                (adapter, score) for adapter, score in adapter_scores.items()
                if adapter != primary_adapter and score > self.confidence_threshold * 0.5
            ]
            secondary_adapters.sort(key=lambda x: x[1], reverse=True)
            secondary_adapters = secondary_adapters[:2]
        else:
            primary_adapter = semantic_result.primary_adapter
            primary_confidence = semantic_result.confidence
            secondary_adapters = semantic_result.secondary_adapters
        
        # Generate combined reasoning
        reasoning = f"Hybrid routing (semantic: {self.semantic_weight}, keyword: {self.keyword_weight}). "
        reasoning += f"Semantic: {semantic_result.reasoning} "
        if keyword_result.confidence > 0:
            reasoning += f"Keyword: {keyword_result.reasoning}"
        
        return UnifiedRoutingResult(
            primary_adapter=primary_adapter,
            confidence=primary_confidence,
            secondary_adapters=secondary_adapters,
            strategy_used=RoutingStrategy.HYBRID,
            semantic_result=semantic_result,
            keyword_result=keyword_result,
            reasoning=reasoning
        )
    
    def _route_ensemble(self, query: str, available_adapters: List[str]) -> UnifiedRoutingResult:
        """Route using ensemble voting from multiple strategies."""
        # Get results from multiple strategies
        semantic_result = self.semantic_router.route_query(query, available_adapters)
        keyword_result = self.keyword_router.route_query(query, available_adapters)
        
        # Ensemble voting
        adapter_votes = {}
        adapter_confidences = {}
        
        # Semantic vote
        adapter_votes[semantic_result.primary_adapter] = adapter_votes.get(semantic_result.primary_adapter, 0) + 1
        adapter_confidences[semantic_result.primary_adapter] = semantic_result.confidence
        
        # Keyword vote
        if keyword_result.confidence > self.confidence_threshold * 0.5:
            adapter_votes[keyword_result.primary_adapter] = adapter_votes.get(keyword_result.primary_adapter, 0) + 1
            if keyword_result.primary_adapter not in adapter_confidences:
                adapter_confidences[keyword_result.primary_adapter] = keyword_result.confidence
            else:
                adapter_confidences[keyword_result.primary_adapter] = max(
                    adapter_confidences[keyword_result.primary_adapter],
                    keyword_result.confidence
                )
        
        # Select adapter with most votes (tie-break by confidence)
        if adapter_votes:
            max_votes = max(adapter_votes.values())
            top_adapters = [adapter for adapter, votes in adapter_votes.items() if votes == max_votes]
            
            if len(top_adapters) == 1:
                primary_adapter = top_adapters[0]
            else:
                # Tie-break by confidence
                primary_adapter = max(top_adapters, key=lambda x: adapter_confidences.get(x, 0))
            
            primary_confidence = adapter_confidences.get(primary_adapter, 0.0)
        else:
            primary_adapter = semantic_result.primary_adapter
            primary_confidence = semantic_result.confidence
        
        reasoning = f"Ensemble voting selected {primary_adapter} with {adapter_votes.get(primary_adapter, 0)} votes."
        
        return UnifiedRoutingResult(
            primary_adapter=primary_adapter,
            confidence=primary_confidence,
            secondary_adapters=semantic_result.secondary_adapters,
            strategy_used=RoutingStrategy.ENSEMBLE,
            semantic_result=semantic_result,
            keyword_result=keyword_result,
            reasoning=reasoning
        )

    def _select_adaptive_strategy(self, query: str) -> RoutingStrategy:
        """Select routing strategy based on query characteristics."""
        query_length = len(query.split())

        # Short queries favor keyword routing
        if query_length < 5:
            return RoutingStrategy.KEYWORD_ONLY

        # Long, complex queries favor semantic routing
        elif query_length > 20:
            return RoutingStrategy.SEMANTIC_ONLY

        # Medium queries use hybrid approach
        else:
            return RoutingStrategy.HYBRID

    def _extract_keywords_from_examples(self, example_queries: List[str]) -> List[str]:
        """Extract keywords from example queries."""
        keywords = set()

        for query in example_queries[:5]:  # Limit to first 5 examples
            # Simple keyword extraction
            words = query.lower().split()
            for word in words:
                # Filter out common words and keep meaningful terms
                if (len(word) > 3 and
                    word not in {'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were'}):
                    keywords.add(word)

        return list(keywords)

    def _update_strategy_performance(self, strategy: RoutingStrategy, confidence: float):
        """Update performance tracking for routing strategies."""
        stats = self.strategy_performance[strategy]
        stats['total_count'] += 1

        if confidence > self.confidence_threshold:
            stats['success_count'] += 1

        # Update running average
        total = stats['total_count']
        stats['avg_confidence'] = (stats['avg_confidence'] * (total - 1) + confidence) / total

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all routing strategies."""
        stats = {}

        for strategy, perf in self.strategy_performance.items():
            if perf['total_count'] > 0:
                success_rate = perf['success_count'] / perf['total_count']
                stats[strategy.value] = {
                    'success_rate': success_rate,
                    'total_queries': perf['total_count'],
                    'avg_confidence': perf['avg_confidence']
                }
            else:
                stats[strategy.value] = {
                    'success_rate': 0.0,
                    'total_queries': 0,
                    'avg_confidence': 0.0
                }

        # Add sub-router stats
        stats['semantic_router'] = self.semantic_router.get_routing_stats()

        return stats

    def update_weights(self, semantic_weight: float, keyword_weight: float):
        """Update routing weights for hybrid strategy."""
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total

        logger.info(f"Updated routing weights: semantic={self.semantic_weight:.2f}, keyword={self.keyword_weight:.2f}")

    def optimize_thresholds(self):
        """Optimize routing thresholds based on performance history."""
        # Simple optimization based on success rates
        for strategy, perf in self.strategy_performance.items():
            if perf['total_count'] > 10:  # Need sufficient data
                success_rate = perf['success_count'] / perf['total_count']

                if success_rate < 0.6:  # Low success rate
                    # Lower threshold to be more permissive
                    if strategy == RoutingStrategy.SEMANTIC_ONLY:
                        self.semantic_router.update_confidence_threshold(
                            max(0.3, self.semantic_router.confidence_threshold - 0.1)
                        )
                elif success_rate > 0.9:  # Very high success rate
                    # Raise threshold to be more selective
                    if strategy == RoutingStrategy.SEMANTIC_ONLY:
                        self.semantic_router.update_confidence_threshold(
                            min(0.9, self.semantic_router.confidence_threshold + 0.05)
                        )

        logger.info("Optimized routing thresholds based on performance")

    def save_configuration(self, filepath: str) -> bool:
        """Save router configuration to file."""
        try:
            config = {
                'default_strategy': self.default_strategy.value,
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight,
                'confidence_threshold': self.confidence_threshold,
                'adaptive_thresholds': self.adaptive_thresholds,
                'strategy_performance': {
                    strategy.value: stats for strategy, stats in self.strategy_performance.items()
                }
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved router configuration to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def load_configuration(self, filepath: str) -> bool:
        """Load router configuration from file."""
        try:
            import json
            import os

            if not os.path.exists(filepath):
                logger.warning(f"Configuration file not found: {filepath}")
                return False

            with open(filepath, 'r') as f:
                config = json.load(f)

            self.default_strategy = RoutingStrategy(config.get('default_strategy', 'hybrid'))
            self.semantic_weight = config.get('semantic_weight', 0.7)
            self.keyword_weight = config.get('keyword_weight', 0.3)
            self.confidence_threshold = config.get('confidence_threshold', 0.6)
            self.adaptive_thresholds = config.get('adaptive_thresholds', self.adaptive_thresholds)

            # Load performance stats if available
            if 'strategy_performance' in config:
                for strategy_name, stats in config['strategy_performance'].items():
                    try:
                        strategy = RoutingStrategy(strategy_name)
                        self.strategy_performance[strategy] = stats
                    except ValueError:
                        logger.warning(f"Unknown strategy in config: {strategy_name}")

            logger.info(f"Loaded router configuration from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
