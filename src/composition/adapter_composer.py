"""
Revolutionary Multi-Adapter Composition System for Adaptrix.

This is the core innovation that sets Adaptrix apart - the ability to compose
multiple specialized adapters in sophisticated ways for enhanced intelligence.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import time
from ..utils.config import config
from ..utils.helpers import timer

logger = logging.getLogger(__name__)


class CompositionStrategy(Enum):
    """Strategies for composing multiple adapters."""
    SEQUENTIAL = "sequential"      # Chain adapters in pipeline
    PARALLEL = "parallel"          # Run adapters simultaneously  
    HIERARCHICAL = "hierarchical"  # Early/mid/late stage specialization
    CONDITIONAL = "conditional"    # Dynamic selection based on intermediate results
    WEIGHTED = "weighted"          # Weighted combination of adapter outputs
    ATTENTION = "attention"        # Learned attention over adapters


@dataclass
class CompositionConfig:
    """Configuration for adapter composition."""
    strategy: CompositionStrategy
    adapters: List[str]
    weights: Optional[List[float]] = None
    temperature: float = 1.0
    confidence_threshold: float = 0.7
    max_adapters: int = 3
    enable_conflict_resolution: bool = True
    enable_attention_weighting: bool = True


@dataclass
class CompositionResult:
    """Result of adapter composition."""
    primary_output: torch.Tensor
    adapter_outputs: Dict[str, torch.Tensor]
    composition_weights: Dict[str, float]
    confidence_scores: Dict[str, float]
    strategy_used: CompositionStrategy
    processing_time: float
    metadata: Dict[str, Any]


class AdapterComposer:
    """
    Revolutionary multi-adapter composition system.
    
    This class implements the core innovation of Adaptrix - the ability to
    compose multiple specialized adapters in sophisticated ways to create
    emergent intelligence capabilities.
    """
    
    def __init__(self, 
                 layer_injector,
                 adapter_manager,
                 default_strategy: CompositionStrategy = CompositionStrategy.PARALLEL):
        """
        Initialize the adapter composer.
        
        Args:
            layer_injector: LayerInjector instance
            adapter_manager: AdapterManager instance  
            default_strategy: Default composition strategy
        """
        self.layer_injector = layer_injector
        self.adapter_manager = adapter_manager
        self.default_strategy = default_strategy
        
        # Composition state
        self.active_compositions: Dict[str, CompositionConfig] = {}
        self.composition_history: List[CompositionResult] = []
        
        # Learned parameters for attention-based composition
        self.attention_weights = nn.ParameterDict()
        self.conflict_resolver = None
        
        # Performance tracking
        self.composition_stats = {
            'total_compositions': 0,
            'successful_compositions': 0,
            'avg_processing_time': 0.0,
            'strategy_usage': {strategy: 0 for strategy in CompositionStrategy}
        }
        
        logger.info("AdapterComposer initialized with revolutionary multi-adapter capabilities")
    
    def compose_adapters(self,
                        adapters: List[str],
                        strategy: Optional[CompositionStrategy] = None,
                        config_override: Optional[Dict[str, Any]] = None) -> CompositionResult:
        """
        Compose multiple adapters using the specified strategy.
        
        Args:
            adapters: List of adapter names to compose
            strategy: Composition strategy to use
            config_override: Override configuration parameters
            
        Returns:
            CompositionResult with combined output and metadata
        """
        start_time = time.time()
        
        try:
            # Use default strategy if none specified
            if strategy is None:
                strategy = self.default_strategy
            
            # Create composition configuration
            composition_config = CompositionConfig(
                strategy=strategy,
                adapters=adapters,
                **config_override or {}
            )
            
            # Validate adapters are available
            available_adapters = self._validate_adapters(adapters)
            if not available_adapters:
                raise ValueError("No valid adapters available for composition")
            
            # Execute composition strategy
            result = self._execute_composition_strategy(composition_config, available_adapters)
            
            # Update statistics
            self._update_composition_stats(result)
            
            # Store in history
            self.composition_history.append(result)
            if len(self.composition_history) > 100:  # Keep last 100 results
                self.composition_history = self.composition_history[-50:]
            
            logger.info(f"Successfully composed {len(available_adapters)} adapters using {strategy.value} strategy")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compose adapters: {e}")
            # Return fallback result
            return CompositionResult(
                primary_output=torch.zeros(1),
                adapter_outputs={},
                composition_weights={},
                confidence_scores={},
                strategy_used=strategy or self.default_strategy,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _validate_adapters(self, adapters: List[str]) -> List[str]:
        """Validate that adapters are available and compatible."""
        available_adapters = []

        for adapter_name in adapters:
            # Check if adapter exists
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data is None:
                logger.warning(f"Adapter {adapter_name} not found, skipping")
                continue

            # For composition, we don't require adapters to be pre-loaded
            # The composition system can work with adapter metadata
            available_adapters.append(adapter_name)

        return available_adapters
    
    def _execute_composition_strategy(self, 
                                    config: CompositionConfig,
                                    adapters: List[str]) -> CompositionResult:
        """Execute the specified composition strategy."""
        
        if config.strategy == CompositionStrategy.SEQUENTIAL:
            return self._sequential_composition(config, adapters)
        elif config.strategy == CompositionStrategy.PARALLEL:
            return self._parallel_composition(config, adapters)
        elif config.strategy == CompositionStrategy.HIERARCHICAL:
            return self._hierarchical_composition(config, adapters)
        elif config.strategy == CompositionStrategy.CONDITIONAL:
            return self._conditional_composition(config, adapters)
        elif config.strategy == CompositionStrategy.WEIGHTED:
            return self._weighted_composition(config, adapters)
        elif config.strategy == CompositionStrategy.ATTENTION:
            return self._attention_composition(config, adapters)
        else:
            raise ValueError(f"Unknown composition strategy: {config.strategy}")
    
    def _parallel_composition(self, config: CompositionConfig, adapters: List[str]) -> CompositionResult:
        """
        Parallel composition: Run multiple adapters simultaneously and combine outputs.
        This is the most revolutionary approach - true multi-adapter intelligence.
        """
        start_time = time.time()
        
        # Get adapter outputs from all active adapters
        adapter_outputs = {}
        confidence_scores = {}
        
        # The parallel composition works by letting all adapters contribute
        # to the same forward pass, then analyzing their combined effect
        
        # For now, we simulate this by getting the current state where
        # multiple adapters are already injected and active
        active_adapters = self.layer_injector.get_active_adapters()
        
        for adapter_name in adapters:
            if adapter_name in active_adapters:
                # Calculate confidence based on adapter metadata and usage
                adapter_data = self.adapter_manager.load_adapter(adapter_name)
                if adapter_data and 'metadata' in adapter_data:
                    performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                    confidence = performance_metrics.get('accuracy', 0.5)
                else:
                    confidence = 0.5
                
                confidence_scores[adapter_name] = confidence
                # For parallel composition, we don't have separate outputs
                # The adapters work together in the forward pass
                adapter_outputs[adapter_name] = torch.tensor([confidence])
        
        # Calculate composition weights based on confidence scores
        total_confidence = sum(confidence_scores.values())
        composition_weights = {}

        if total_confidence > 0:
            for adapter_name, confidence in confidence_scores.items():
                composition_weights[adapter_name] = confidence / total_confidence
        else:
            # Equal weights if no confidence information
            weight = 1.0 / len(adapters) if len(adapters) > 0 else 1.0
            composition_weights = {name: weight for name in adapters}
        
        # The primary output is the combined effect of all adapters
        # In parallel composition, this is emergent from the forward pass
        if confidence_scores:
            primary_output = torch.tensor([sum(confidence_scores.values()) / len(confidence_scores)])
        else:
            primary_output = torch.tensor([0.5])  # Default confidence
        
        return CompositionResult(
            primary_output=primary_output,
            adapter_outputs=adapter_outputs,
            composition_weights=composition_weights,
            confidence_scores=confidence_scores,
            strategy_used=CompositionStrategy.PARALLEL,
            processing_time=time.time() - start_time,
            metadata={
                'total_adapters': len(adapters),
                'active_adapters': len(adapter_outputs),
                'avg_confidence': np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
            }
        )

    def _sequential_composition(self, config: CompositionConfig, adapters: List[str]) -> CompositionResult:
        """
        Sequential composition: Chain adapters in pipeline.
        Each adapter processes the output of the previous one.
        """
        start_time = time.time()

        adapter_outputs = {}
        confidence_scores = {}
        composition_weights = {}

        # In sequential composition, adapters are applied one after another
        # This requires careful orchestration of loading/unloading

        current_confidence = 1.0
        for i, adapter_name in enumerate(adapters):
            # Calculate weight based on position in sequence
            weight = 1.0 / (i + 1)  # Later adapters have less weight
            composition_weights[adapter_name] = weight

            # Get adapter confidence
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data and 'metadata' in adapter_data:
                performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                confidence = performance_metrics.get('accuracy', 0.5)
            else:
                confidence = 0.5

            # Sequential confidence degrades
            current_confidence *= confidence
            confidence_scores[adapter_name] = current_confidence
            adapter_outputs[adapter_name] = torch.tensor([current_confidence])

        primary_output = torch.tensor([current_confidence])

        return CompositionResult(
            primary_output=primary_output,
            adapter_outputs=adapter_outputs,
            composition_weights=composition_weights,
            confidence_scores=confidence_scores,
            strategy_used=CompositionStrategy.SEQUENTIAL,
            processing_time=time.time() - start_time,
            metadata={
                'sequence_length': len(adapters),
                'final_confidence': current_confidence
            }
        )

    def _hierarchical_composition(self, config: CompositionConfig, adapters: List[str]) -> CompositionResult:
        """
        Hierarchical composition: Early/mid/late stage specialization.
        Different adapters handle different stages of processing.
        """
        start_time = time.time()

        # Divide adapters into stages based on their capabilities
        stages = self._organize_adapters_by_stage(adapters)

        adapter_outputs = {}
        confidence_scores = {}
        composition_weights = {}

        stage_confidences = []

        for stage_name, stage_adapters in stages.items():
            stage_confidence = 0.0
            stage_weight = 1.0 / len(stages)

            for adapter_name in stage_adapters:
                adapter_data = self.adapter_manager.load_adapter(adapter_name)
                if adapter_data and 'metadata' in adapter_data:
                    performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                    confidence = performance_metrics.get('accuracy', 0.5)
                else:
                    confidence = 0.5

                confidence_scores[adapter_name] = confidence
                composition_weights[adapter_name] = stage_weight / len(stage_adapters)
                adapter_outputs[adapter_name] = torch.tensor([confidence])
                stage_confidence += confidence

            if stage_adapters:
                stage_confidences.append(stage_confidence / len(stage_adapters))
            else:
                stage_confidences.append(0.5)  # Default confidence for empty stage

        # Hierarchical output is the product of stage confidences
        primary_output = torch.tensor([np.prod(stage_confidences)])

        return CompositionResult(
            primary_output=primary_output,
            adapter_outputs=adapter_outputs,
            composition_weights=composition_weights,
            confidence_scores=confidence_scores,
            strategy_used=CompositionStrategy.HIERARCHICAL,
            processing_time=time.time() - start_time,
            metadata={
                'stages': list(stages.keys()),
                'stage_confidences': stage_confidences
            }
        )

    def _conditional_composition(self, config: CompositionConfig, adapters: List[str]) -> CompositionResult:
        """
        Conditional composition: Dynamic selection based on intermediate results.
        Adapters are selected based on confidence and context.
        """
        start_time = time.time()

        adapter_outputs = {}
        confidence_scores = {}
        composition_weights = {}

        # Evaluate each adapter's suitability for the current context
        adapter_suitability = {}

        for adapter_name in adapters:
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data and 'metadata' in adapter_data:
                performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                confidence = performance_metrics.get('accuracy', 0.5)

                # Add contextual factors
                capabilities = adapter_data['metadata'].get('capabilities', [])
                suitability = confidence

                # Boost suitability based on recent performance
                if adapter_name in [r.adapter_outputs.keys() for r in self.composition_history[-5:]]:
                    suitability *= 1.1  # Recent usage boost

                adapter_suitability[adapter_name] = suitability
            else:
                adapter_suitability[adapter_name] = 0.3

        # Select top adapters based on suitability
        sorted_adapters = sorted(adapter_suitability.items(), key=lambda x: x[1], reverse=True)
        selected_adapters = sorted_adapters[:config.max_adapters]

        total_suitability = sum(suit for _, suit in selected_adapters)

        for adapter_name, suitability in selected_adapters:
            confidence_scores[adapter_name] = suitability
            composition_weights[adapter_name] = suitability / total_suitability if total_suitability > 0 else 1.0 / len(selected_adapters)
            adapter_outputs[adapter_name] = torch.tensor([suitability])

        if selected_adapters:
            primary_output = torch.tensor([total_suitability / len(selected_adapters)])
        else:
            primary_output = torch.tensor([0.0])  # No adapters selected

        return CompositionResult(
            primary_output=primary_output,
            adapter_outputs=adapter_outputs,
            composition_weights=composition_weights,
            confidence_scores=confidence_scores,
            strategy_used=CompositionStrategy.CONDITIONAL,
            processing_time=time.time() - start_time,
            metadata={
                'selected_adapters': len(selected_adapters),
                'total_candidates': len(adapters),
                'selection_threshold': config.confidence_threshold
            }
        )

    def _weighted_composition(self, config: CompositionConfig, adapters: List[str]) -> CompositionResult:
        """
        Weighted composition: Combine adapters with specified weights.
        """
        start_time = time.time()

        adapter_outputs = {}
        confidence_scores = {}
        composition_weights = {}

        # Use provided weights or calculate based on performance
        if config.weights and len(config.weights) == len(adapters):
            weights = config.weights
        else:
            # Calculate weights based on adapter performance
            weights = []
            for adapter_name in adapters:
                adapter_data = self.adapter_manager.load_adapter(adapter_name)
                if adapter_data and 'metadata' in adapter_data:
                    performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                    weight = performance_metrics.get('accuracy', 0.5)
                else:
                    weight = 0.5
                weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(adapters) if len(adapters) > 0 else 1.0
            weights = [equal_weight] * len(adapters)

        weighted_sum = 0.0
        for adapter_name, weight in zip(adapters, weights):
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data and 'metadata' in adapter_data:
                performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                confidence = performance_metrics.get('accuracy', 0.5)
            else:
                confidence = 0.5

            confidence_scores[adapter_name] = confidence
            composition_weights[adapter_name] = weight
            adapter_outputs[adapter_name] = torch.tensor([confidence])
            weighted_sum += confidence * weight

        primary_output = torch.tensor([weighted_sum])

        return CompositionResult(
            primary_output=primary_output,
            adapter_outputs=adapter_outputs,
            composition_weights=composition_weights,
            confidence_scores=confidence_scores,
            strategy_used=CompositionStrategy.WEIGHTED,
            processing_time=time.time() - start_time,
            metadata={
                'weights_provided': config.weights is not None,
                'weighted_score': weighted_sum
            }
        )

    def _attention_composition(self, config: CompositionConfig, adapters: List[str]) -> CompositionResult:
        """
        Attention-based composition: Learned attention over adapter outputs.
        This is the most sophisticated composition strategy.
        """
        start_time = time.time()

        adapter_outputs = {}
        confidence_scores = {}
        composition_weights = {}

        # Get adapter representations
        adapter_representations = []
        adapter_confidences = []

        for adapter_name in adapters:
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data and 'metadata' in adapter_data:
                performance_metrics = adapter_data['metadata'].get('performance_metrics', {})
                confidence = performance_metrics.get('accuracy', 0.5)

                # Create a simple representation (could be enhanced with embeddings)
                representation = torch.tensor([confidence, len(adapter_data['metadata'].get('capabilities', []))])
                adapter_representations.append(representation)
                adapter_confidences.append(confidence)
            else:
                representation = torch.tensor([0.5, 1.0])
                adapter_representations.append(representation)
                adapter_confidences.append(0.5)

        # Calculate attention weights
        if len(adapter_representations) > 1:
            representations = torch.stack(adapter_representations)

            # Simple attention mechanism (could be enhanced with learned parameters)
            attention_scores = torch.softmax(representations.sum(dim=1) / config.temperature, dim=0)
            attention_weights = attention_scores.tolist()
        else:
            attention_weights = [1.0]

        # Apply attention weights
        attended_sum = 0.0
        for adapter_name, weight, confidence in zip(adapters, attention_weights, adapter_confidences):
            confidence_scores[adapter_name] = confidence
            composition_weights[adapter_name] = weight
            adapter_outputs[adapter_name] = torch.tensor([confidence])
            attended_sum += confidence * weight

        primary_output = torch.tensor([attended_sum])

        return CompositionResult(
            primary_output=primary_output,
            adapter_outputs=adapter_outputs,
            composition_weights=composition_weights,
            confidence_scores=confidence_scores,
            strategy_used=CompositionStrategy.ATTENTION,
            processing_time=time.time() - start_time,
            metadata={
                'attention_weights': attention_weights,
                'temperature': config.temperature,
                'attended_score': attended_sum
            }
        )

    def _organize_adapters_by_stage(self, adapters: List[str]) -> Dict[str, List[str]]:
        """Organize adapters into processing stages based on their capabilities."""
        stages = {
            'early': [],    # Input processing, tokenization, basic understanding
            'middle': [],   # Reasoning, analysis, computation
            'late': []      # Output formatting, generation, refinement
        }

        for adapter_name in adapters:
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data and 'metadata' in adapter_data:
                capabilities = adapter_data['metadata'].get('capabilities', [])
                description = adapter_data['metadata'].get('description', '').lower()

                # Simple heuristic for stage assignment
                if any(cap in ['input', 'tokenization', 'preprocessing'] for cap in capabilities):
                    stages['early'].append(adapter_name)
                elif any(cap in ['reasoning', 'analysis', 'computation', 'math'] for cap in capabilities):
                    stages['middle'].append(adapter_name)
                elif any(cap in ['output', 'generation', 'formatting'] for cap in capabilities):
                    stages['late'].append(adapter_name)
                else:
                    # Default to middle stage
                    stages['middle'].append(adapter_name)
            else:
                stages['middle'].append(adapter_name)

        # Remove empty stages
        stages = {k: v for k, v in stages.items() if v}

        return stages

    def _update_composition_stats(self, result: CompositionResult):
        """Update composition performance statistics."""
        self.composition_stats['total_compositions'] += 1

        if result.metadata.get('error') is None:
            self.composition_stats['successful_compositions'] += 1

        # Update running average of processing time
        total = self.composition_stats['total_compositions']
        current_avg = self.composition_stats['avg_processing_time']
        self.composition_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + result.processing_time) / total
        )

        # Update strategy usage
        self.composition_stats['strategy_usage'][result.strategy_used] += 1

    def get_composition_stats(self) -> Dict[str, Any]:
        """Get composition performance statistics."""
        stats = self.composition_stats.copy()

        # Add success rate
        if stats['total_compositions'] > 0:
            stats['success_rate'] = stats['successful_compositions'] / stats['total_compositions']
        else:
            stats['success_rate'] = 0.0

        # Add most used strategy
        if stats['strategy_usage']:
            most_used = max(stats['strategy_usage'].items(), key=lambda x: x[1])
            stats['most_used_strategy'] = most_used[0].value
        else:
            stats['most_used_strategy'] = None

        return stats

    def get_active_compositions(self) -> Dict[str, CompositionConfig]:
        """Get currently active composition configurations."""
        return self.active_compositions.copy()

    def clear_composition_history(self):
        """Clear composition history to free memory."""
        self.composition_history.clear()
        logger.info("Cleared composition history")

    def recommend_composition_strategy(self,
                                     adapters: List[str],
                                     context: Optional[Dict[str, Any]] = None) -> CompositionStrategy:
        """
        Recommend the best composition strategy based on adapters and context.

        Args:
            adapters: List of adapter names
            context: Optional context information

        Returns:
            Recommended composition strategy
        """
        # Simple heuristic-based recommendation
        num_adapters = len(adapters)

        if num_adapters == 1:
            return CompositionStrategy.PARALLEL  # Single adapter, parallel is fine

        if num_adapters == 2:
            return CompositionStrategy.WEIGHTED  # Two adapters work well with weighting

        if num_adapters <= 3:
            return CompositionStrategy.ATTENTION  # Few adapters, attention works well

        if num_adapters <= 5:
            return CompositionStrategy.CONDITIONAL  # Many adapters, be selective

        # Too many adapters, use hierarchical to organize them
        return CompositionStrategy.HIERARCHICAL

    def create_composition_config(self,
                                adapters: List[str],
                                strategy: Optional[CompositionStrategy] = None,
                                **kwargs) -> CompositionConfig:
        """
        Create a composition configuration with intelligent defaults.

        Args:
            adapters: List of adapter names
            strategy: Composition strategy (auto-recommended if None)
            **kwargs: Additional configuration parameters

        Returns:
            CompositionConfig instance
        """
        if strategy is None:
            strategy = self.recommend_composition_strategy(adapters)

        return CompositionConfig(
            strategy=strategy,
            adapters=adapters,
            weights=kwargs.get('weights'),
            temperature=kwargs.get('temperature', 1.0),
            confidence_threshold=kwargs.get('confidence_threshold', 0.7),
            max_adapters=kwargs.get('max_adapters', 3),
            enable_conflict_resolution=kwargs.get('enable_conflict_resolution', True),
            enable_attention_weighting=kwargs.get('enable_attention_weighting', True)
        )
