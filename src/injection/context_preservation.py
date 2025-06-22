"""
Context preservation engine for maintaining coherence across multiple layer injections.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time

from ..utils.config import config

logger = logging.getLogger(__name__)


class ContextPreservingInjector:
    """
    Enhanced injector that maintains context integrity across multiple layer injections.
    
    Features:
    - Residual connections with context preservation
    - Attention mask propagation
    - Context drift detection and correction
    - Semantic coherence validation
    """
    
    def __init__(self, base_model: torch.nn.Module):
        """
        Initialize context-preserving injector.
        
        Args:
            base_model: Base transformer model
        """
        self.base_model = base_model
        self.device = next(base_model.parameters()).device
        
        # Context tracking
        self.context_cache = {}
        self.attention_states = {}
        self.original_query_embedding = None
        
        # Configuration
        self.drift_threshold = config.get('injection.drift_threshold', 0.3)
        self.context_preservation_weight = config.get('injection.context_weight', 0.8)
        self.enable_context_validation = config.get('injection.validate_context', True)
        
        # Statistics
        self.injection_stats = defaultdict(list)
        
        logger.info("ContextPreservingInjector initialized")
    
    def inject_with_context(self, 
                          layer_idx: int, 
                          input_hidden_states: torch.Tensor,
                          adapter_output: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adapter injection with context preservation.
        
        Args:
            layer_idx: Layer index
            input_hidden_states: Original hidden states
            adapter_output: LoRA adapter output
            attention_mask: Attention mask for context preservation
            
        Returns:
            Context-preserved output
        """
        start_time = time.time()
        
        # 1. Preserve original context
        original_context = input_hidden_states.clone()
        
        # 2. Apply context-aware blending
        blended_output = self._blend_with_context(
            original_context,
            adapter_output,
            attention_mask,
            layer_idx
        )
        
        # 3. Validate context integrity if enabled
        if self.enable_context_validation:
            context_metrics = self._validate_context_integrity(
                original_context,
                blended_output,
                layer_idx
            )
            
            # Apply correction if needed
            if context_metrics['context_preserved'] < 0.7:
                logger.warning(f"Context drift detected at layer {layer_idx}: {context_metrics}")
                blended_output = self._apply_context_correction(
                    original_context,
                    blended_output,
                    context_metrics
                )
        
        # 4. Update context cache
        self._update_context_cache(layer_idx, blended_output, attention_mask)
        
        # 5. Record statistics
        processing_time = time.time() - start_time
        self.injection_stats[layer_idx].append({
            'processing_time': processing_time,
            'input_norm': torch.norm(input_hidden_states).item(),
            'output_norm': torch.norm(blended_output).item(),
            'adapter_norm': torch.norm(adapter_output).item()
        })
        
        return blended_output
    
    def _blend_with_context(self,
                          original_context: torch.Tensor,
                          adapter_output: torch.Tensor,
                          attention_mask: Optional[torch.Tensor],
                          layer_idx: int) -> torch.Tensor:
        """
        Blend adapter output with original context using sophisticated weighting.

        Args:
            original_context: Original hidden states (module output)
            adapter_output: Adapter transformation output (should match original_context shape)
            attention_mask: Attention mask
            layer_idx: Current layer index

        Returns:
            Blended output with preserved context
        """
        # Validate shapes before addition - they should match for proper residual connection
        if original_context.shape != adapter_output.shape:
            logger.warning(f"Shape mismatch in context blending at layer {layer_idx}: "
                         f"original {original_context.shape} vs adapter {adapter_output.shape}")
            # For mismatched shapes, fall back to standard residual connection
            # This should not happen if LoRA dimensions are correct
            return original_context

        # Standard LoRA residual connection
        base_output = original_context + adapter_output
        
        # Apply attention-aware weighting if mask is available
        if attention_mask is not None:
            # Expand attention mask to match hidden dimensions
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(original_context)
            
            # Apply stronger context preservation to attended tokens
            context_weight = self.context_preservation_weight
            adapter_weight = 1.0 - context_weight
            
            # Weighted combination
            blended = (context_weight * original_context + 
                      adapter_weight * adapter_output) * expanded_mask.float()
            
            # Add unmasked positions with standard residual
            blended = blended + base_output * (1 - expanded_mask.float())
        else:
            # Fallback to standard residual connection
            blended = base_output
        
        # Apply layer-specific context preservation
        if layer_idx in self.context_cache:
            previous_context = self.context_cache[layer_idx]
            
            # Compute context similarity
            similarity = F.cosine_similarity(
                blended.view(-1, blended.size(-1)),
                previous_context.view(-1, previous_context.size(-1)),
                dim=-1
            ).mean()
            
            # Apply context anchoring if similarity is too low
            if similarity < 0.5:
                anchor_weight = 0.3
                blended = (1 - anchor_weight) * blended + anchor_weight * previous_context
        
        return blended
    
    def _validate_context_integrity(self, 
                                  pre_injection: torch.Tensor,
                                  post_injection: torch.Tensor,
                                  layer_idx: int) -> Dict[str, float]:
        """
        Validate context integrity after injection.
        
        Args:
            pre_injection: Hidden states before injection
            post_injection: Hidden states after injection
            layer_idx: Current layer index
            
        Returns:
            Context integrity metrics
        """
        # 1. Semantic similarity check
        similarity = F.cosine_similarity(
            pre_injection.view(-1, pre_injection.size(-1)),
            post_injection.view(-1, post_injection.size(-1)),
            dim=-1
        ).mean().item()
        
        # 2. Magnitude preservation check
        pre_norm = torch.norm(pre_injection, dim=-1).mean().item()
        post_norm = torch.norm(post_injection, dim=-1).mean().item()
        magnitude_ratio = min(pre_norm, post_norm) / max(pre_norm, post_norm)
        
        # 3. Attention pattern consistency (if we have previous states)
        attention_drift = 0.0
        if layer_idx in self.attention_states:
            prev_attention = self.attention_states[layer_idx]
            current_attention = self._compute_attention_pattern(post_injection)
            attention_drift = F.mse_loss(prev_attention, current_attention).item()
        
        # 4. Context coherence score
        coherence = self._calculate_coherence_score(post_injection)
        
        # 5. Overall context preservation score
        context_preserved = (similarity * 0.4 + 
                           magnitude_ratio * 0.3 + 
                           (1 - min(attention_drift, 1.0)) * 0.2 + 
                           coherence * 0.1)
        
        return {
            'similarity': similarity,
            'magnitude_ratio': magnitude_ratio,
            'attention_drift': attention_drift,
            'coherence': coherence,
            'context_preserved': context_preserved
        }
    
    def _apply_context_correction(self, 
                                original_context: torch.Tensor,
                                drifted_output: torch.Tensor,
                                context_metrics: Dict[str, float]) -> torch.Tensor:
        """
        Apply context correction when drift is detected.
        
        Args:
            original_context: Original context
            drifted_output: Output with detected drift
            context_metrics: Context validation metrics
            
        Returns:
            Corrected output
        """
        # Calculate correction strength based on drift severity
        drift_severity = 1.0 - context_metrics['context_preserved']
        correction_strength = min(drift_severity * 2.0, 0.5)  # Max 50% correction
        
        # Apply correction
        corrected_output = (1 - correction_strength) * drifted_output + \
                          correction_strength * original_context
        
        logger.debug(f"Applied context correction with strength {correction_strength:.3f}")
        
        return corrected_output
    
    def _compute_attention_pattern(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified attention pattern for drift detection.
        
        Args:
            hidden_states: Hidden states tensor
            
        Returns:
            Attention pattern tensor
        """
        # Simplified attention computation for monitoring
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute self-attention scores (simplified)
        query = hidden_states
        key = hidden_states
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (hidden_dim ** 0.5)
        attention_pattern = F.softmax(attention_scores, dim=-1)
        
        # Return averaged pattern for comparison
        return attention_pattern.mean(dim=0)  # Average across batch
    
    def _calculate_coherence_score(self, hidden_states: torch.Tensor) -> float:
        """
        Calculate context coherence score.
        
        Args:
            hidden_states: Hidden states tensor
            
        Returns:
            Coherence score (0-1)
        """
        # Calculate variance across sequence dimension
        variance = torch.var(hidden_states, dim=1).mean().item()
        
        # Calculate mean activation
        mean_activation = torch.mean(torch.abs(hidden_states)).item()
        
        # Coherence is inverse of normalized variance
        if mean_activation > 0:
            coherence = 1.0 / (1.0 + variance / mean_activation)
        else:
            coherence = 0.5
        
        return min(coherence, 1.0)
    
    def _update_context_cache(self, 
                            layer_idx: int, 
                            hidden_states: torch.Tensor,
                            attention_mask: Optional[torch.Tensor]) -> None:
        """
        Update context cache for future reference.
        
        Args:
            layer_idx: Layer index
            hidden_states: Current hidden states
            attention_mask: Attention mask
        """
        # Store context (detached to avoid memory leaks)
        self.context_cache[layer_idx] = hidden_states.detach().clone()
        
        # Store attention pattern
        if attention_mask is not None:
            self.attention_states[layer_idx] = self._compute_attention_pattern(hidden_states)
    
    def set_query_anchor(self, query_embedding: torch.Tensor) -> None:
        """
        Set the original query embedding as an anchor for context preservation.
        
        Args:
            query_embedding: Original query embedding
        """
        self.original_query_embedding = query_embedding.detach().clone()
        logger.debug("Query anchor set for context preservation")
    
    def detect_adapter_conflicts(self, layer_adapters: Dict[int, List[str]]) -> Dict[int, float]:
        """
        Detect potential conflicts between adapters in different layers.
        
        Args:
            layer_adapters: Dictionary mapping layer indices to adapter names
            
        Returns:
            Dictionary mapping layer indices to conflict scores
        """
        conflict_scores = {}
        
        for layer_idx, adapters in layer_adapters.items():
            if len(adapters) > 1:
                # Multiple adapters in same layer - potential conflict
                conflict_scores[layer_idx] = 0.8
            elif layer_idx in self.context_cache:
                # Check semantic conflict with previous context
                current_context = self.context_cache[layer_idx]
                
                # Simplified conflict detection based on context similarity
                if self.original_query_embedding is not None:
                    similarity = F.cosine_similarity(
                        current_context.view(-1),
                        self.original_query_embedding.view(-1),
                        dim=0
                    ).item()
                    
                    # High dissimilarity indicates potential conflict
                    conflict_scores[layer_idx] = max(0.0, 1.0 - similarity)
                else:
                    conflict_scores[layer_idx] = 0.0
            else:
                conflict_scores[layer_idx] = 0.0
        
        return conflict_scores
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get context preservation statistics.
        
        Returns:
            Dictionary with context statistics
        """
        stats = {
            'layers_with_context': len(self.context_cache),
            'total_injections': sum(len(layer_stats) for layer_stats in self.injection_stats.values()),
            'average_processing_time': 0.0,
            'layer_statistics': {}
        }
        
        # Calculate layer-specific statistics
        total_time = 0.0
        total_injections = 0
        
        for layer_idx, layer_stats in self.injection_stats.items():
            if layer_stats:
                avg_time = sum(s['processing_time'] for s in layer_stats) / len(layer_stats)
                avg_input_norm = sum(s['input_norm'] for s in layer_stats) / len(layer_stats)
                avg_output_norm = sum(s['output_norm'] for s in layer_stats) / len(layer_stats)
                
                stats['layer_statistics'][layer_idx] = {
                    'injection_count': len(layer_stats),
                    'avg_processing_time': avg_time,
                    'avg_input_norm': avg_input_norm,
                    'avg_output_norm': avg_output_norm
                }
                
                total_time += avg_time * len(layer_stats)
                total_injections += len(layer_stats)
        
        if total_injections > 0:
            stats['average_processing_time'] = total_time / total_injections
        
        return stats
    
    def clear_context(self) -> None:
        """Clear all context cache and statistics."""
        self.context_cache.clear()
        self.attention_states.clear()
        self.injection_stats.clear()
        self.original_query_embedding = None
        logger.debug("Context cache cleared")


class ConversationContextManager:
    """
    Manages context across multiple conversation turns.
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize conversation context manager.
        
        Args:
            max_history: Maximum conversation history to maintain
        """
        self.max_history = max_history
        self.conversation_history = []
        self.context_embeddings = []
        
        logger.info(f"ConversationContextManager initialized with max_history={max_history}")
    
    def maintain_chat_context(self, 
                            new_query: str, 
                            adapter_response: str,
                            active_adapters: List[str]) -> None:
        """
        Maintain chat context across conversation turns.
        
        Args:
            new_query: New user query
            adapter_response: Response from adapter
            active_adapters: List of active adapters
        """
        # Add to conversation history
        self.conversation_history.append({
            'query': new_query,
            'response': adapter_response,
            'adapters_used': active_adapters.copy(),
            'timestamp': time.time()
        })
        
        # Encode conversation context (simplified)
        context_embedding = self._encode_conversation_context(new_query, adapter_response)
        self.context_embeddings.append(context_embedding)
        
        # Prune old context if needed
        if len(self.conversation_history) > self.max_history:
            self._prune_old_context()
    
    def _encode_conversation_context(self, query: str, response: str) -> torch.Tensor:
        """
        Encode conversation context into embedding.
        
        Args:
            query: User query
            response: System response
            
        Returns:
            Context embedding tensor
        """
        # Simplified context encoding (in practice, use sentence transformers)
        combined_text = f"{query} {response}"
        
        # Create a simple hash-based embedding for now
        import hashlib
        text_hash = hashlib.md5(combined_text.encode()).hexdigest()
        
        # Convert to tensor (simplified)
        embedding = torch.tensor([float(int(c, 16)) for c in text_hash[:16]], dtype=torch.float32)
        
        return embedding
    
    def _prune_old_context(self) -> None:
        """Prune old conversation context."""
        # Remove oldest entries
        excess = len(self.conversation_history) - self.max_history
        if excess > 0:
            self.conversation_history = self.conversation_history[excess:]
            self.context_embeddings = self.context_embeddings[excess:]
            
            logger.debug(f"Pruned {excess} old conversation entries")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get conversation summary.
        
        Returns:
            Conversation summary dictionary
        """
        if not self.conversation_history:
            return {'total_turns': 0}
        
        # Calculate statistics
        total_turns = len(self.conversation_history)
        unique_adapters = set()
        
        for entry in self.conversation_history:
            unique_adapters.update(entry['adapters_used'])
        
        recent_adapters = []
        if self.conversation_history:
            recent_adapters = self.conversation_history[-1]['adapters_used']
        
        return {
            'total_turns': total_turns,
            'unique_adapters_used': list(unique_adapters),
            'recent_adapters': recent_adapters,
            'conversation_length': sum(len(entry['query']) + len(entry['response']) 
                                     for entry in self.conversation_history)
        }
