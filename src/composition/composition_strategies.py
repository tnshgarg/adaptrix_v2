"""
Specialized composition strategy implementations for Adaptrix.

This module provides concrete implementations of different composition strategies,
each optimized for specific use cases and adapter combinations.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


class BaseComposer(ABC):
    """Base class for all composition strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.usage_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
    
    @abstractmethod
    def compose(self, 
                adapter_outputs: Dict[str, torch.Tensor],
                adapter_metadata: Dict[str, Dict[str, Any]],
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compose adapter outputs using this strategy.
        
        Args:
            adapter_outputs: Dictionary of adapter outputs
            adapter_metadata: Metadata for each adapter
            context: Optional context information
            
        Returns:
            Tuple of (composed_output, composition_metadata)
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this composer."""
        return {
            'name': self.name,
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.usage_count, 1),
            'avg_processing_time': self.total_processing_time / max(self.usage_count, 1)
        }


class SequentialComposer(BaseComposer):
    """
    Sequential composition strategy.
    
    Processes adapters in sequence, where each adapter's output
    becomes the input for the next adapter in the chain.
    """
    
    def __init__(self):
        super().__init__("sequential")
        self.chain_order_strategy = "performance"  # or "manual", "random"
    
    def compose(self, 
                adapter_outputs: Dict[str, torch.Tensor],
                adapter_metadata: Dict[str, Dict[str, Any]],
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compose adapters sequentially.
        """
        start_time = time.time()
        self.usage_count += 1
        
        try:
            if not adapter_outputs:
                raise ValueError("No adapter outputs provided")
            
            # Determine processing order
            adapter_order = self._determine_processing_order(adapter_outputs, adapter_metadata)
            
            # Process adapters in sequence
            current_output = None
            processing_chain = []
            
            for adapter_name in adapter_order:
                if adapter_name not in adapter_outputs:
                    continue
                
                adapter_output = adapter_outputs[adapter_name]
                metadata = adapter_metadata.get(adapter_name, {})
                
                if current_output is None:
                    # First adapter in chain
                    current_output = adapter_output
                else:
                    # Combine with previous output
                    # For sequential composition, we add the outputs
                    current_output = current_output + adapter_output
                
                processing_chain.append({
                    'adapter': adapter_name,
                    'confidence': metadata.get('performance_metrics', {}).get('accuracy', 0.5),
                    'output_norm': torch.norm(adapter_output).item()
                })
            
            if current_output is None:
                raise ValueError("No valid adapters processed")
            
            # Normalize the final output
            final_output = current_output / len(adapter_order)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.success_count += 1
            
            composition_metadata = {
                'strategy': 'sequential',
                'processing_order': adapter_order,
                'processing_chain': processing_chain,
                'chain_length': len(adapter_order),
                'processing_time': processing_time
            }
            
            return final_output, composition_metadata
            
        except Exception as e:
            logger.error(f"Sequential composition failed: {e}")
            # Return zero tensor as fallback
            fallback_output = torch.zeros_like(list(adapter_outputs.values())[0])
            return fallback_output, {'error': str(e), 'strategy': 'sequential'}
    
    def _determine_processing_order(self, 
                                   adapter_outputs: Dict[str, torch.Tensor],
                                   adapter_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
        """Determine the order in which to process adapters."""
        adapter_names = list(adapter_outputs.keys())
        
        if self.chain_order_strategy == "performance":
            # Order by performance (highest first)
            def get_performance(name):
                metadata = adapter_metadata.get(name, {})
                return metadata.get('performance_metrics', {}).get('accuracy', 0.5)
            
            adapter_names.sort(key=get_performance, reverse=True)
        
        elif self.chain_order_strategy == "random":
            np.random.shuffle(adapter_names)
        
        # "manual" or default: use provided order
        return adapter_names


class ParallelComposer(BaseComposer):
    """
    Parallel composition strategy.
    
    Runs all adapters simultaneously and combines their outputs
    using various aggregation methods.
    """
    
    def __init__(self, aggregation_method: str = "weighted_sum"):
        super().__init__("parallel")
        self.aggregation_method = aggregation_method  # "weighted_sum", "max", "mean", "attention"
    
    def compose(self, 
                adapter_outputs: Dict[str, torch.Tensor],
                adapter_metadata: Dict[str, Dict[str, Any]],
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compose adapters in parallel.
        """
        start_time = time.time()
        self.usage_count += 1
        
        try:
            if not adapter_outputs:
                raise ValueError("No adapter outputs provided")
            
            # Calculate weights for each adapter
            adapter_weights = self._calculate_adapter_weights(adapter_outputs, adapter_metadata)
            
            # Aggregate outputs based on method
            if self.aggregation_method == "weighted_sum":
                final_output = self._weighted_sum_aggregation(adapter_outputs, adapter_weights)
            elif self.aggregation_method == "max":
                final_output = self._max_aggregation(adapter_outputs)
            elif self.aggregation_method == "mean":
                final_output = self._mean_aggregation(adapter_outputs)
            elif self.aggregation_method == "attention":
                final_output = self._attention_aggregation(adapter_outputs, adapter_weights)
            else:
                # Default to weighted sum
                final_output = self._weighted_sum_aggregation(adapter_outputs, adapter_weights)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.success_count += 1
            
            composition_metadata = {
                'strategy': 'parallel',
                'aggregation_method': self.aggregation_method,
                'adapter_weights': adapter_weights,
                'num_adapters': len(adapter_outputs),
                'processing_time': processing_time
            }
            
            return final_output, composition_metadata
            
        except Exception as e:
            logger.error(f"Parallel composition failed: {e}")
            fallback_output = torch.zeros_like(list(adapter_outputs.values())[0])
            return fallback_output, {'error': str(e), 'strategy': 'parallel'}
    
    def _calculate_adapter_weights(self, 
                                  adapter_outputs: Dict[str, torch.Tensor],
                                  adapter_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weights for each adapter based on performance and other factors."""
        weights = {}
        total_weight = 0.0
        
        for adapter_name in adapter_outputs.keys():
            metadata = adapter_metadata.get(adapter_name, {})
            performance_metrics = metadata.get('performance_metrics', {})
            
            # Base weight from accuracy
            accuracy = performance_metrics.get('accuracy', 0.5)
            
            # Adjust for latency (lower latency = higher weight)
            latency = performance_metrics.get('latency_ms', 100)
            latency_factor = max(0.1, 1.0 - (latency / 1000))  # Normalize latency
            
            # Combine factors
            weight = accuracy * latency_factor
            weights[adapter_name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(adapter_outputs)
            weights = {name: equal_weight for name in adapter_outputs.keys()}
        
        return weights
    
    def _weighted_sum_aggregation(self, 
                                 adapter_outputs: Dict[str, torch.Tensor],
                                 weights: Dict[str, float]) -> torch.Tensor:
        """Aggregate outputs using weighted sum."""
        weighted_sum = None
        
        for adapter_name, output in adapter_outputs.items():
            weight = weights.get(adapter_name, 1.0 / len(adapter_outputs))
            weighted_output = output * weight
            
            if weighted_sum is None:
                weighted_sum = weighted_output
            else:
                weighted_sum += weighted_output
        
        return weighted_sum
    
    def _max_aggregation(self, adapter_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate outputs using element-wise maximum."""
        outputs = list(adapter_outputs.values())
        return torch.stack(outputs).max(dim=0)[0]
    
    def _mean_aggregation(self, adapter_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate outputs using mean."""
        outputs = list(adapter_outputs.values())
        return torch.stack(outputs).mean(dim=0)
    
    def _attention_aggregation(self, 
                              adapter_outputs: Dict[str, torch.Tensor],
                              weights: Dict[str, float]) -> torch.Tensor:
        """Aggregate outputs using attention mechanism."""
        # Convert weights to attention scores
        weight_values = list(weights.values())
        attention_scores = torch.softmax(torch.tensor(weight_values), dim=0)
        
        # Apply attention to outputs
        outputs = list(adapter_outputs.values())
        stacked_outputs = torch.stack(outputs)
        
        # Weighted combination
        attended_output = torch.sum(stacked_outputs * attention_scores.unsqueeze(-1), dim=0)
        
        return attended_output


class HierarchicalComposer(BaseComposer):
    """
    Hierarchical composition strategy.
    
    Organizes adapters into stages (early, middle, late) and
    processes them hierarchically for complex reasoning tasks.
    """
    
    def __init__(self):
        super().__init__("hierarchical")
        self.stage_weights = {'early': 0.3, 'middle': 0.5, 'late': 0.2}
    
    def compose(self, 
                adapter_outputs: Dict[str, torch.Tensor],
                adapter_metadata: Dict[str, Dict[str, Any]],
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compose adapters hierarchically.
        """
        start_time = time.time()
        self.usage_count += 1
        
        try:
            if not adapter_outputs:
                raise ValueError("No adapter outputs provided")
            
            # Organize adapters into stages
            stages = self._organize_into_stages(adapter_outputs, adapter_metadata)
            
            # Process each stage
            stage_outputs = {}
            stage_metadata = {}
            
            for stage_name, stage_adapters in stages.items():
                if not stage_adapters:
                    continue
                
                # Combine adapters within this stage
                stage_output = self._combine_stage_adapters(stage_adapters, adapter_outputs)
                stage_outputs[stage_name] = stage_output
                stage_metadata[stage_name] = {
                    'num_adapters': len(stage_adapters),
                    'adapter_names': list(stage_adapters.keys())
                }
            
            # Combine stage outputs
            final_output = self._combine_stage_outputs(stage_outputs)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.success_count += 1
            
            composition_metadata = {
                'strategy': 'hierarchical',
                'stages': stage_metadata,
                'stage_weights': self.stage_weights,
                'processing_time': processing_time
            }
            
            return final_output, composition_metadata
            
        except Exception as e:
            logger.error(f"Hierarchical composition failed: {e}")
            fallback_output = torch.zeros_like(list(adapter_outputs.values())[0])
            return fallback_output, {'error': str(e), 'strategy': 'hierarchical'}
    
    def _organize_into_stages(self, 
                             adapter_outputs: Dict[str, torch.Tensor],
                             adapter_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Organize adapters into processing stages."""
        stages = {'early': {}, 'middle': {}, 'late': {}}
        
        for adapter_name, output in adapter_outputs.items():
            metadata = adapter_metadata.get(adapter_name, {})
            capabilities = metadata.get('capabilities', [])
            description = metadata.get('description', '').lower()
            
            # Simple heuristic for stage assignment
            if any(cap in ['input', 'preprocessing', 'tokenization'] for cap in capabilities):
                stages['early'][adapter_name] = output
            elif any(cap in ['output', 'generation', 'formatting'] for cap in capabilities):
                stages['late'][adapter_name] = output
            else:
                # Default to middle stage for reasoning, math, etc.
                stages['middle'][adapter_name] = output
        
        return stages
    
    def _combine_stage_adapters(self, 
                               stage_adapters: Dict[str, torch.Tensor],
                               all_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine adapters within a single stage."""
        if len(stage_adapters) == 1:
            return list(stage_adapters.values())[0]
        
        # Simple mean combination within stage
        outputs = list(stage_adapters.values())
        return torch.stack(outputs).mean(dim=0)
    
    def _combine_stage_outputs(self, stage_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine outputs from different stages."""
        if not stage_outputs:
            raise ValueError("No stage outputs to combine")
        
        if len(stage_outputs) == 1:
            return list(stage_outputs.values())[0]
        
        # Weighted combination of stages
        weighted_sum = None
        total_weight = 0.0
        
        for stage_name, output in stage_outputs.items():
            weight = self.stage_weights.get(stage_name, 1.0 / len(stage_outputs))
            weighted_output = output * weight
            
            if weighted_sum is None:
                weighted_sum = weighted_output
            else:
                weighted_sum += weighted_output
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight
        
        return weighted_sum


class ConditionalComposer(BaseComposer):
    """
    Conditional composition strategy.
    
    Dynamically selects and combines adapters based on
    intermediate results and confidence scores.
    """
    
    def __init__(self, selection_threshold: float = 0.7):
        super().__init__("conditional")
        self.selection_threshold = selection_threshold
        self.max_selected_adapters = 3
    
    def compose(self, 
                adapter_outputs: Dict[str, torch.Tensor],
                adapter_metadata: Dict[str, Dict[str, Any]],
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compose adapters conditionally based on performance and context.
        """
        start_time = time.time()
        self.usage_count += 1
        
        try:
            if not adapter_outputs:
                raise ValueError("No adapter outputs provided")
            
            # Evaluate adapter suitability
            adapter_scores = self._evaluate_adapter_suitability(adapter_outputs, adapter_metadata, context)
            
            # Select top adapters
            selected_adapters = self._select_adapters(adapter_scores)
            
            if not selected_adapters:
                # Fallback: select best single adapter
                best_adapter = max(adapter_scores.items(), key=lambda x: x[1])[0]
                selected_adapters = {best_adapter: adapter_scores[best_adapter]}
            
            # Combine selected adapters
            selected_outputs = {name: adapter_outputs[name] for name in selected_adapters.keys()}
            final_output = self._combine_selected_adapters(selected_outputs, selected_adapters)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.success_count += 1
            
            composition_metadata = {
                'strategy': 'conditional',
                'selected_adapters': list(selected_adapters.keys()),
                'adapter_scores': adapter_scores,
                'selection_threshold': self.selection_threshold,
                'processing_time': processing_time
            }
            
            return final_output, composition_metadata
            
        except Exception as e:
            logger.error(f"Conditional composition failed: {e}")
            fallback_output = torch.zeros_like(list(adapter_outputs.values())[0])
            return fallback_output, {'error': str(e), 'strategy': 'conditional'}
    
    def _evaluate_adapter_suitability(self, 
                                     adapter_outputs: Dict[str, torch.Tensor],
                                     adapter_metadata: Dict[str, Dict[str, Any]],
                                     context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate how suitable each adapter is for the current context."""
        scores = {}
        
        for adapter_name, output in adapter_outputs.items():
            metadata = adapter_metadata.get(adapter_name, {})
            performance_metrics = metadata.get('performance_metrics', {})
            
            # Base score from accuracy
            base_score = performance_metrics.get('accuracy', 0.5)
            
            # Adjust based on output confidence (using output norm as proxy)
            output_confidence = min(torch.norm(output).item() / 10.0, 1.0)
            
            # Combine factors
            final_score = (base_score * 0.7) + (output_confidence * 0.3)
            scores[adapter_name] = final_score
        
        return scores
    
    def _select_adapters(self, adapter_scores: Dict[str, float]) -> Dict[str, float]:
        """Select adapters based on scores and threshold."""
        # Sort by score
        sorted_adapters = sorted(adapter_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select adapters above threshold
        selected = {}
        for adapter_name, score in sorted_adapters:
            if score >= self.selection_threshold and len(selected) < self.max_selected_adapters:
                selected[adapter_name] = score
        
        return selected
    
    def _combine_selected_adapters(self, 
                                  selected_outputs: Dict[str, torch.Tensor],
                                  adapter_scores: Dict[str, float]) -> torch.Tensor:
        """Combine outputs from selected adapters."""
        if len(selected_outputs) == 1:
            return list(selected_outputs.values())[0]
        
        # Weighted combination based on scores
        total_score = sum(adapter_scores.values())
        weights = {name: score / total_score for name, score in adapter_scores.items()}
        
        weighted_sum = None
        for adapter_name, output in selected_outputs.items():
            weight = weights[adapter_name]
            weighted_output = output * weight
            
            if weighted_sum is None:
                weighted_sum = weighted_output
            else:
                weighted_sum += weighted_output
        
        return weighted_sum
