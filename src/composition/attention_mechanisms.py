"""
Advanced attention mechanisms and conflict resolution for multi-adapter composition.

This module implements sophisticated attention mechanisms that allow adapters
to dynamically weight their contributions based on context and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """Result of attention computation."""
    attention_weights: torch.Tensor
    attended_output: torch.Tensor
    attention_entropy: float
    confidence_score: float
    metadata: Dict[str, Any]


class AdapterAttention(nn.Module):
    """
    Learned attention mechanism for adapter composition.
    
    This module learns to weight adapter contributions based on:
    - Adapter performance history
    - Current context
    - Inter-adapter compatibility
    - Query characteristics
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize adapter attention mechanism.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout
        
        # Attention components
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Adapter-specific components
        self.adapter_embeddings = nn.ParameterDict()
        self.performance_weighting = nn.Linear(1, hidden_size)
        self.context_encoder = nn.Linear(hidden_size, hidden_size)
        
        # Dropout and normalization
        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"AdapterAttention initialized with {num_attention_heads} heads")
    
    def register_adapter(self, adapter_name: str, adapter_metadata: Dict[str, Any]):
        """
        Register an adapter with the attention mechanism.
        
        Args:
            adapter_name: Name of the adapter
            adapter_metadata: Adapter metadata including performance metrics
        """
        # Create learnable embedding for this adapter
        adapter_embedding = nn.Parameter(torch.randn(self.hidden_size))
        self.adapter_embeddings[adapter_name] = adapter_embedding
        
        logger.info(f"Registered adapter {adapter_name} with attention mechanism")
    
    def forward(self,
                query_context: torch.Tensor,
                adapter_outputs: Dict[str, torch.Tensor],
                adapter_metadata: Dict[str, Dict[str, Any]]) -> AttentionResult:
        """
        Compute attention-weighted combination of adapter outputs.
        
        Args:
            query_context: Context representation of the current query
            adapter_outputs: Dictionary of adapter outputs
            adapter_metadata: Metadata for each adapter
            
        Returns:
            AttentionResult with weighted outputs and attention information
        """
        batch_size = query_context.size(0)
        adapter_names = list(adapter_outputs.keys())
        num_adapters = len(adapter_names)
        
        if num_adapters == 0:
            return AttentionResult(
                attention_weights=torch.empty(0),
                attended_output=torch.zeros_like(query_context),
                attention_entropy=0.0,
                confidence_score=0.0,
                metadata={'error': 'No adapters provided'}
            )
        
        # Prepare adapter representations
        adapter_representations = []
        performance_scores = []
        
        for adapter_name in adapter_names:
            # Get adapter embedding
            if adapter_name in self.adapter_embeddings:
                adapter_emb = self.adapter_embeddings[adapter_name]
            else:
                # Create temporary embedding for unregistered adapters
                adapter_emb = torch.randn(self.hidden_size, device=query_context.device)
            
            # Get performance score
            metadata = adapter_metadata.get(adapter_name, {})
            performance_metrics = metadata.get('performance_metrics', {})
            performance_score = performance_metrics.get('accuracy', 0.5)
            performance_scores.append(performance_score)
            
            # Combine adapter embedding with performance
            perf_tensor = torch.tensor([performance_score], device=query_context.device)
            perf_weighted = self.performance_weighting(perf_tensor.unsqueeze(0))
            
            adapter_repr = adapter_emb.unsqueeze(0) + perf_weighted
            adapter_representations.append(adapter_repr)
        
        # Stack adapter representations
        adapter_reprs = torch.stack(adapter_representations, dim=1)  # [batch, num_adapters, hidden]
        
        # Encode query context
        encoded_context = self.context_encoder(query_context.unsqueeze(1))  # [batch, 1, hidden]
        
        # Compute attention scores
        queries = self.query_projection(encoded_context)
        keys = self.key_projection(adapter_reprs)
        values = self.value_projection(adapter_reprs)
        
        # Multi-head attention
        queries = self._reshape_for_attention(queries)  # [batch, heads, 1, head_dim]
        keys = self._reshape_for_attention(keys)        # [batch, heads, num_adapters, head_dim]
        values = self._reshape_for_attention(values)    # [batch, heads, num_adapters, head_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Reshape and project output
        attended_output = self._reshape_from_attention(attended_values)
        attended_output = self.output_projection(attended_output)
        attended_output = self.layer_norm(attended_output + encoded_context)
        
        # Calculate attention entropy (measure of attention distribution)
        avg_attention = attention_weights.mean(dim=1).squeeze(1)  # [batch, num_adapters]
        attention_entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8), dim=-1)
        
        # Calculate confidence score
        max_attention = torch.max(avg_attention, dim=-1)[0]
        confidence_score = max_attention.mean().item()
        
        return AttentionResult(
            attention_weights=avg_attention.squeeze(0),
            attended_output=attended_output.squeeze(1),
            attention_entropy=attention_entropy.mean().item(),
            confidence_score=confidence_score,
            metadata={
                'num_adapters': num_adapters,
                'adapter_names': adapter_names,
                'performance_scores': performance_scores
            }
        )
    
    def _reshape_for_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, hidden_size = tensor.size()
        return tensor.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
    
    def _reshape_from_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from multi-head attention."""
        batch_size, num_heads, seq_len, head_dim = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)


class ConflictResolver:
    """
    Resolves conflicts between adapter outputs when they disagree.
    
    This component detects when adapters produce conflicting outputs
    and applies resolution strategies to produce coherent results.
    """
    
    def __init__(self, 
                 conflict_threshold: float = 0.3,
                 resolution_strategy: str = "weighted_voting"):
        """
        Initialize conflict resolver.
        
        Args:
            conflict_threshold: Threshold for detecting conflicts
            resolution_strategy: Strategy for resolving conflicts
        """
        self.conflict_threshold = conflict_threshold
        self.resolution_strategy = resolution_strategy
        
        # Conflict detection metrics
        self.conflict_history = []
        self.resolution_stats = {
            'total_conflicts': 0,
            'resolved_conflicts': 0,
            'resolution_methods': {}
        }
        
        logger.info(f"ConflictResolver initialized with {resolution_strategy} strategy")
    
    def detect_conflicts(self, 
                        adapter_outputs: Dict[str, torch.Tensor],
                        adapter_confidences: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect conflicts between adapter outputs.
        
        Args:
            adapter_outputs: Dictionary of adapter outputs
            adapter_confidences: Confidence scores for each adapter
            
        Returns:
            Conflict detection results
        """
        if len(adapter_outputs) < 2:
            return {'has_conflict': False, 'conflict_score': 0.0}
        
        # Calculate pairwise similarities between outputs
        adapter_names = list(adapter_outputs.keys())
        similarities = []
        
        for i in range(len(adapter_names)):
            for j in range(i + 1, len(adapter_names)):
                output1 = adapter_outputs[adapter_names[i]]
                output2 = adapter_outputs[adapter_names[j]]
                
                # Simple cosine similarity
                similarity = F.cosine_similarity(
                    output1.flatten().unsqueeze(0),
                    output2.flatten().unsqueeze(0)
                ).item()
                similarities.append(similarity)
        
        # Calculate conflict score (lower similarity = higher conflict)
        avg_similarity = np.mean(similarities)
        conflict_score = 1.0 - avg_similarity
        
        has_conflict = conflict_score > self.conflict_threshold
        
        return {
            'has_conflict': has_conflict,
            'conflict_score': conflict_score,
            'avg_similarity': avg_similarity,
            'pairwise_similarities': similarities,
            'conflicting_adapters': adapter_names if has_conflict else []
        }
    
    def resolve_conflict(self,
                        adapter_outputs: Dict[str, torch.Tensor],
                        adapter_confidences: Dict[str, float],
                        conflict_info: Dict[str, Any]) -> torch.Tensor:
        """
        Resolve conflicts between adapter outputs.
        
        Args:
            adapter_outputs: Dictionary of adapter outputs
            adapter_confidences: Confidence scores for each adapter
            conflict_info: Information about detected conflicts
            
        Returns:
            Resolved output tensor
        """
        if not conflict_info['has_conflict']:
            # No conflict, return weighted average
            return self._weighted_average(adapter_outputs, adapter_confidences)
        
        self.conflict_history.append(conflict_info)
        self.resolution_stats['total_conflicts'] += 1
        
        if self.resolution_strategy == "weighted_voting":
            resolved_output = self._weighted_voting_resolution(adapter_outputs, adapter_confidences)
        elif self.resolution_strategy == "highest_confidence":
            resolved_output = self._highest_confidence_resolution(adapter_outputs, adapter_confidences)
        elif self.resolution_strategy == "median_output":
            resolved_output = self._median_output_resolution(adapter_outputs)
        else:
            # Fallback to weighted average
            resolved_output = self._weighted_average(adapter_outputs, adapter_confidences)
        
        self.resolution_stats['resolved_conflicts'] += 1
        method = self.resolution_strategy
        self.resolution_stats['resolution_methods'][method] = self.resolution_stats['resolution_methods'].get(method, 0) + 1
        
        return resolved_output
    
    def _weighted_average(self, 
                         adapter_outputs: Dict[str, torch.Tensor],
                         adapter_confidences: Dict[str, float]) -> torch.Tensor:
        """Compute weighted average of adapter outputs."""
        total_weight = sum(adapter_confidences.values())
        if total_weight == 0:
            total_weight = len(adapter_outputs)
            weights = {name: 1.0 / len(adapter_outputs) for name in adapter_outputs.keys()}
        else:
            weights = {name: conf / total_weight for name, conf in adapter_confidences.items()}
        
        weighted_sum = None
        for name, output in adapter_outputs.items():
            weighted_output = output * weights[name]
            if weighted_sum is None:
                weighted_sum = weighted_output
            else:
                weighted_sum += weighted_output
        
        return weighted_sum
    
    def _weighted_voting_resolution(self,
                                   adapter_outputs: Dict[str, torch.Tensor],
                                   adapter_confidences: Dict[str, float]) -> torch.Tensor:
        """Resolve conflicts using weighted voting."""
        # For now, same as weighted average but could be enhanced
        return self._weighted_average(adapter_outputs, adapter_confidences)
    
    def _highest_confidence_resolution(self,
                                      adapter_outputs: Dict[str, torch.Tensor],
                                      adapter_confidences: Dict[str, float]) -> torch.Tensor:
        """Resolve conflicts by selecting highest confidence adapter."""
        if not adapter_confidences:
            # Return first output if no confidence information
            return list(adapter_outputs.values())[0]
        
        highest_conf_adapter = max(adapter_confidences.items(), key=lambda x: x[1])[0]
        return adapter_outputs[highest_conf_adapter]
    
    def _median_output_resolution(self, adapter_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Resolve conflicts using median of outputs."""
        if len(adapter_outputs) == 1:
            return list(adapter_outputs.values())[0]
        
        # Stack outputs and compute median
        outputs = torch.stack(list(adapter_outputs.values()))
        median_output = torch.median(outputs, dim=0)[0]
        return median_output
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """Get conflict resolution statistics."""
        stats = self.resolution_stats.copy()
        
        if stats['total_conflicts'] > 0:
            stats['resolution_rate'] = stats['resolved_conflicts'] / stats['total_conflicts']
        else:
            stats['resolution_rate'] = 0.0
        
        # Add recent conflict information
        if self.conflict_history:
            recent_conflicts = self.conflict_history[-10:]  # Last 10 conflicts
            stats['recent_avg_conflict_score'] = np.mean([c['conflict_score'] for c in recent_conflicts])
            stats['recent_conflicts'] = len(recent_conflicts)
        
        return stats
