"""
Dimension adapter for cross-model LoRA compatibility.
Handles dimension mismatches between different model architectures.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional
import json
import os

logger = logging.getLogger(__name__)


class DimensionProjector(nn.Module):
    """
    Projects LoRA weights from one dimension to another.
    Allows using adapters trained on different model sizes.
    """
    
    def __init__(self, source_dim: int, target_dim: int, rank: int = 16):
        """
        Initialize dimension projector.
        
        Args:
            source_dim: Source dimension (from original adapter)
            target_dim: Target dimension (for current model)
            rank: Rank for the projection (lower = more compression)
        """
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.rank = min(rank, min(source_dim, target_dim))
        
        # Projection layers
        if source_dim != target_dim:
            # Use SVD-based projection for better preservation
            self.down_proj = nn.Linear(source_dim, self.rank, bias=False)
            self.up_proj = nn.Linear(self.rank, target_dim, bias=False)
            
            # Initialize with identity-like behavior
            self._initialize_projection()
        else:
            self.down_proj = None
            self.up_proj = None
    
    def _initialize_projection(self):
        """Initialize projection to preserve as much information as possible."""
        # Initialize down projection with random orthogonal matrix
        nn.init.orthogonal_(self.down_proj.weight)
        
        # Initialize up projection to approximate identity
        with torch.no_grad():
            # Create pseudo-identity matrix
            if self.target_dim >= self.rank:
                identity_part = torch.eye(self.rank)
                if self.target_dim > self.rank:
                    padding = torch.zeros(self.target_dim - self.rank, self.rank)
                    identity_part = torch.cat([identity_part, padding], dim=0)
                self.up_proj.weight.copy_(identity_part.T)
            else:
                # Target is smaller, use truncated identity
                identity_part = torch.eye(self.target_dim, self.rank)
                self.up_proj.weight.copy_(identity_part.T)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tensor from source to target dimension.
        
        Args:
            x: Input tensor of shape (..., source_dim)
            
        Returns:
            Projected tensor of shape (..., target_dim)
        """
        if self.down_proj is None:
            return x  # No projection needed
        
        # Apply projection
        projected = self.down_proj(x)
        result = self.up_proj(projected)
        
        return result


class DimensionCompatibleAdapter:
    """
    Wrapper for LoRA adapters that handles dimension compatibility.
    """
    
    def __init__(self, adapter_path: str, target_dimensions: Dict[str, Tuple[int, int]]):
        """
        Initialize dimension-compatible adapter.
        
        Args:
            adapter_path: Path to the original adapter
            target_dimensions: Dict mapping module names to (input_dim, output_dim)
        """
        self.adapter_path = adapter_path
        self.target_dimensions = target_dimensions
        self.projectors = {}
        self.adapted_weights = {}
        
        self._load_and_adapt()
    
    def _load_and_adapt(self):
        """Load original adapter and create dimension-compatible version."""
        try:
            # Load metadata
            metadata_path = os.path.join(self.adapter_path, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loading adapter: {metadata.get('name', 'unknown')}")
            logger.info(f"Original base model: {metadata.get('original_base_model', 'unknown')}")
            
            # Load layer weights
            for layer_file in os.listdir(self.adapter_path):
                if layer_file.startswith("layer_") and layer_file.endswith(".pt"):
                    layer_idx = int(layer_file.split("_")[1].split(".")[0])
                    layer_path = os.path.join(self.adapter_path, layer_file)
                    
                    layer_weights = torch.load(layer_path, map_location='cpu')
                    adapted_layer = self._adapt_layer_weights(layer_weights, layer_idx)
                    
                    if adapted_layer:
                        self.adapted_weights[layer_idx] = adapted_layer
            
            logger.info(f"Successfully adapted {len(self.adapted_weights)} layers")
            
        except Exception as e:
            logger.error(f"Failed to load and adapt weights: {e}")
            raise
    
    def _adapt_layer_weights(self, layer_weights: Dict, layer_idx: int) -> Dict:
        """
        Adapt layer weights to target dimensions.
        
        Args:
            layer_weights: Original layer weights
            layer_idx: Layer index
            
        Returns:
            Adapted layer weights
        """
        adapted = {}
        
        for module_name, weights in layer_weights.items():
            if module_name not in self.target_dimensions:
                logger.warning(f"No target dimensions for module {module_name}, skipping")
                continue
            
            target_input_dim, target_output_dim = self.target_dimensions[module_name]
            
            # Get original dimensions
            lora_A = weights['lora_A']  # Shape: (rank, input_dim)
            lora_B = weights['lora_B']  # Shape: (output_dim, rank)
            
            original_input_dim = lora_A.shape[1]
            original_output_dim = lora_B.shape[0]
            rank = lora_A.shape[0]
            
            logger.debug(f"Layer {layer_idx}.{module_name}: "
                        f"{original_input_dim}x{original_output_dim} -> "
                        f"{target_input_dim}x{target_output_dim}")
            
            # Adapt dimensions if needed
            adapted_lora_A = self._adapt_tensor_dimension(
                lora_A, (rank, original_input_dim), (rank, target_input_dim), 
                f"{layer_idx}_{module_name}_A"
            )
            
            adapted_lora_B = self._adapt_tensor_dimension(
                lora_B, (original_output_dim, rank), (target_output_dim, rank),
                f"{layer_idx}_{module_name}_B"
            )
            
            adapted[module_name] = {
                'lora_A': adapted_lora_A,
                'lora_B': adapted_lora_B,
                'rank': rank,
                'alpha': weights.get('alpha', 16)
            }
        
        return adapted
    
    def _adapt_tensor_dimension(self, tensor: torch.Tensor, 
                               original_shape: Tuple[int, int], 
                               target_shape: Tuple[int, int],
                               projector_key: str) -> torch.Tensor:
        """
        Adapt tensor dimensions using projection.
        
        Args:
            tensor: Original tensor
            original_shape: Original (dim1, dim2)
            target_shape: Target (dim1, dim2)
            projector_key: Key for caching projector
            
        Returns:
            Dimension-adapted tensor
        """
        if original_shape == target_shape:
            return tensor
        
        # Only adapt the dimension that changed
        if original_shape[0] != target_shape[0] and original_shape[1] == target_shape[1]:
            # First dimension changed
            if projector_key not in self.projectors:
                self.projectors[projector_key] = DimensionProjector(
                    original_shape[0], target_shape[0]
                )
            
            # Transpose, project, transpose back
            projected = self.projectors[projector_key](tensor.T).T
            
        elif original_shape[1] != target_shape[1] and original_shape[0] == target_shape[0]:
            # Second dimension changed
            if projector_key not in self.projectors:
                self.projectors[projector_key] = DimensionProjector(
                    original_shape[1], target_shape[1]
                )
            
            projected = self.projectors[projector_key](tensor)
            
        else:
            # Both dimensions changed - more complex case
            logger.warning(f"Both dimensions changed for {projector_key}, using simple scaling")
            # Use simple interpolation as fallback
            projected = torch.nn.functional.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        return projected
    
    def get_adapted_weights(self) -> Dict[int, Dict]:
        """Get the dimension-adapted weights."""
        return self.adapted_weights
    
    def save_adapted_weights(self, output_path: str):
        """Save adapted weights to disk."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save adapted layer weights
        for layer_idx, weights in self.adapted_weights.items():
            layer_file = os.path.join(output_path, f"layer_{layer_idx}.pt")
            torch.save(weights, layer_file)
        
        # Create updated metadata
        original_metadata_path = os.path.join(self.adapter_path, "metadata.json")
        with open(original_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update metadata for adapted version
        metadata['name'] = metadata.get('name', 'unknown') + '_adapted'
        metadata['description'] = f"Dimension-adapted version of {metadata.get('name', 'unknown')}"
        metadata['base_model'] = 'deepseek-ai/deepseek-r1-distill-qwen-1.5b'
        metadata['adapted_from'] = self.adapter_path
        metadata['target_dimensions'] = {k: list(v) for k, v in self.target_dimensions.items()}
        
        metadata_file = os.path.join(output_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved adapted weights to {output_path}")


def create_compatible_adapter(source_adapter_path: str, 
                            target_model_name: str = "deepseek-ai/deepseek-r1-distill-qwen-1.5b") -> str:
    """
    Create a dimension-compatible version of an existing adapter.
    
    Args:
        source_adapter_path: Path to source adapter
        target_model_name: Target model name
        
    Returns:
        Path to the adapted adapter
    """
    # Define target dimensions for DeepSeek-R1-1.5B
    target_dimensions = {
        'self_attn.q_proj': (1536, 1536),
        'self_attn.v_proj': (1536, 256),
        'self_attn.k_proj': (1536, 256),
        'self_attn.o_proj': (1536, 1536),
        'mlp.gate_proj': (1536, 8960),
        'mlp.up_proj': (1536, 8960),
        'mlp.down_proj': (8960, 1536)
    }
    
    # Create adapter
    adapter = DimensionCompatibleAdapter(source_adapter_path, target_dimensions)
    
    # Save adapted version
    adapter_name = os.path.basename(source_adapter_path)
    output_path = os.path.join("adapters", f"{adapter_name}_compatible")
    adapter.save_adapted_weights(output_path)
    
    return output_path
