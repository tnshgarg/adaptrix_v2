"""
Middle-layer LoRA injection engine for Adaptrix system.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict
from ..utils.config import config
from .context_preservation import ContextPreservingInjector

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer implementation.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            rank: LoRA rank (bottleneck dimension)
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor
            
        Returns:
            LoRA transformation output
        """
        if not self.enabled:
            return torch.zeros_like(x)
        
        # LoRA computation: x -> A -> dropout -> B -> scale
        result = self.lora_A(x)
        result = self.dropout(result)
        result = self.lora_B(result)
        return result * self.scaling
    
    def enable(self):
        """Enable LoRA layer."""
        self.enabled = True
    
    def disable(self):
        """Disable LoRA layer."""
        self.enabled = False


class LayerInjector:
    """
    Manages dynamic LoRA injection into transformer layers.
    
    Uses PyTorch forward hooks to inject LoRA computations into
    specific modules within transformer layers.
    """
    
    def __init__(self, base_model: nn.Module):
        """
        Initialize layer injector.

        Args:
            base_model: Base transformer model
        """
        self.base_model = base_model
        self.device = next(base_model.parameters()).device

        # Registry of active adapters per layer
        self.active_adapters: Dict[int, Dict[str, str]] = defaultdict(dict)

        # Registry of LoRA layers
        self.lora_layers: Dict[str, LoRALayer] = {}

        # Registry of forward hooks
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}

        # Target modules cache
        self._target_modules_cache: Optional[Dict[str, torch.nn.Module]] = None

        # Context preservation
        self.context_injector = ContextPreservingInjector(base_model)
        self.enable_context_preservation = config.get('injection.enable_context_preservation', True)

        logger.info("LayerInjector initialized with context preservation")
    
    def _get_target_modules(self) -> Dict[str, torch.nn.Module]:
        """
        Get target modules for injection.
        
        Returns:
            Dictionary mapping module paths to modules
        """
        if self._target_modules_cache is not None:
            return self._target_modules_cache
        
        target_modules = {}
        
        # Get transformer layers
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            layers = self.base_model.transformer.h
            layer_prefix = 'transformer.h'
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            layers = self.base_model.model.layers
            layer_prefix = 'model.layers'
        else:
            raise RuntimeError("Could not find transformer layers")
        
        # Extract target modules from each layer
        target_module_names = config.get('injection.target_modules', ['attn.c_attn', 'mlp.c_fc'])
        
        for layer_idx, layer in enumerate(layers):
            for module_name in target_module_names:
                module_path = f"{layer_prefix}.{layer_idx}.{module_name}"
                
                # Navigate to the target module
                module = layer
                for part in module_name.split('.'):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        logger.warning(f"Module {module_name} not found in layer {layer_idx}")
                        module = None
                        break
                
                if module is not None:
                    # Check for standard PyTorch modules
                    is_valid_module = isinstance(module, (nn.Linear, nn.Conv1d))

                    # Also check for transformers Conv1D (used in GPT-2/DialoGPT)
                    try:
                        from transformers.pytorch_utils import Conv1D as TransformersConv1D
                        if isinstance(module, TransformersConv1D):
                            is_valid_module = True
                    except ImportError:
                        pass

                    if is_valid_module:
                        target_modules[module_path] = module
                        logger.debug(f"Found target module: {module_path} ({type(module).__name__})")
        
        self._target_modules_cache = target_modules
        logger.info(f"Found {len(target_modules)} target modules for injection")
        return target_modules
    
    def register_injection_point(self, layer_idx: int, module_name: str) -> bool:
        """
        Register an injection point for a specific layer and module.
        
        Args:
            layer_idx: Target layer index
            module_name: Target module name (e.g., 'self_attn.q_proj')
            
        Returns:
            True if registration successful
        """
        try:
            target_modules = self._get_target_modules()
            
            # Find the target module
            if hasattr(self.base_model, 'transformer'):
                module_path = f"transformer.h.{layer_idx}.{module_name}"
            else:
                module_path = f"model.layers.{layer_idx}.{module_name}"
            
            if module_path not in target_modules:
                logger.error(f"Module {module_path} not found")
                return False
            
            target_module = target_modules[module_path]
            
            # Create hook key
            hook_key = f"{layer_idx}_{module_name}"
            
            # Remove existing hook if present
            if hook_key in self.hooks:
                self.hooks[hook_key].remove()
            
            # Create and register hook
            hook = self._create_injection_hook(layer_idx, module_name, target_module)
            handle = target_module.register_forward_hook(hook)
            self.hooks[hook_key] = handle
            
            logger.info(f"Registered injection point: layer {layer_idx}, module {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register injection point: {e}")
            return False
    
    def _create_injection_hook(self, layer_idx: int, module_name: str, target_module: nn.Module) -> Callable:
        """
        Create a forward hook for LoRA injection.
        
        Args:
            layer_idx: Target layer index
            module_name: Target module name
            target_module: Target module instance
            
        Returns:
            Hook function
        """
        def hook_fn(_module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            """Forward hook function for LoRA injection with context preservation."""
            try:
                # Check if there are active adapters for this layer
                if layer_idx not in self.active_adapters:
                    return output

                # Get input tensor and attention mask
                if isinstance(input, tuple):
                    input_tensor = input[0]
                    # Try to extract attention mask if available
                    attention_mask = None
                    if len(input) > 1 and hasattr(input[1], 'shape'):
                        attention_mask = input[1]
                else:
                    input_tensor = input
                    attention_mask = None

                # Apply LoRA transformations from active adapters
                # Only apply LoRA for the specific module this hook is attached to
                total_lora_output = torch.zeros_like(output)
                lora_applied = False

                for adapter_name in self.active_adapters[layer_idx]:
                    # Only apply LoRA for the specific module this hook is attached to
                    lora_key = f"{adapter_name}_{layer_idx}_{module_name}"

                    if lora_key in self.lora_layers:
                        lora_layer = self.lora_layers[lora_key]
                        if lora_layer.enabled:
                            try:
                                lora_output = lora_layer(input_tensor)

                                # Ensure LoRA output matches expected output dimensions
                                if lora_output.shape == output.shape:
                                    total_lora_output += lora_output
                                    lora_applied = True
                                    logger.debug(f"Successfully applied LoRA {lora_key}: {lora_output.shape}")
                                else:
                                    logger.warning(f"LoRA output shape {lora_output.shape} doesn't match expected {output.shape} for {lora_key}")
                                    logger.warning(f"Skipping LoRA {lora_key} due to dimension mismatch")
                                    continue
                            except Exception as e:
                                logger.warning(f"LoRA computation failed for {lora_key}: {e}")
                                continue

                # Apply context preservation if enabled and LoRA was applied
                if self.enable_context_preservation and lora_applied:
                    try:
                        final_output = self.context_injector.inject_with_context(
                            layer_idx=layer_idx,
                            input_hidden_states=output,
                            adapter_output=total_lora_output,
                            attention_mask=attention_mask
                        )
                    except Exception as e:
                        logger.warning(f"Context preservation failed for layer {layer_idx}: {e}")
                        # Fallback to standard residual connection
                        final_output = output + total_lora_output
                else:
                    # Standard LoRA residual connection
                    if lora_applied:
                        final_output = output + total_lora_output
                    else:
                        final_output = output

                return final_output

            except Exception as e:
                logger.error(f"Error in injection hook: {e}")
                return output
        
        return hook_fn
    
    def inject_adapter(self, layer_idx: int, adapter_name: str, adapter_weights: Dict[str, Any]) -> bool:
        """
        Inject a LoRA adapter into a specific layer.
        
        Args:
            layer_idx: Target layer index
            adapter_name: Name of the adapter
            adapter_weights: Dictionary containing LoRA weights
            
        Returns:
            True if injection successful
        """
        try:
            target_modules = config.get('injection.target_modules', ['attn.c_attn', 'mlp.c_fc'])
            
            for module_name in target_modules:
                # Check if weights exist for this module
                weight_key = f"{module_name}"
                if weight_key not in adapter_weights:
                    continue
                
                module_weights = adapter_weights[weight_key]
                
                # Extract LoRA parameters
                lora_A = module_weights.get('lora_A')
                lora_B = module_weights.get('lora_B')
                rank = module_weights.get('rank', config.get('injection.default_rank', 16))
                alpha = module_weights.get('alpha', config.get('injection.default_alpha', 32))
                
                if lora_A is None or lora_B is None:
                    logger.warning(f"Missing LoRA weights for {adapter_name} layer {layer_idx} module {module_name}")
                    continue
                
                # Create LoRA layer
                in_features = lora_A.shape[1]
                out_features = lora_B.shape[0]
                
                lora_layer = LoRALayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=config.get('injection.dropout', 0.1)
                )
                
                # Load weights
                if isinstance(lora_A, torch.Tensor):
                    lora_layer.lora_A.weight.data = lora_A.detach().clone().to(self.device)
                else:
                    lora_layer.lora_A.weight.data = torch.tensor(lora_A, device=self.device)

                if isinstance(lora_B, torch.Tensor):
                    lora_layer.lora_B.weight.data = lora_B.detach().clone().to(self.device)
                else:
                    lora_layer.lora_B.weight.data = torch.tensor(lora_B, device=self.device)
                
                # Move to device
                lora_layer = lora_layer.to(self.device)
                
                # Register LoRA layer
                lora_key = f"{adapter_name}_{layer_idx}_{module_name}"
                self.lora_layers[lora_key] = lora_layer
                
                # Register injection point if not already registered
                self.register_injection_point(layer_idx, module_name)
            
            # Add to active adapters
            self.active_adapters[layer_idx][adapter_name] = adapter_name
            
            logger.info(f"Successfully injected adapter {adapter_name} into layer {layer_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to inject adapter {adapter_name}: {e}")
            return False
    
    def remove_adapter(self, layer_idx: int, adapter_name: str) -> bool:
        """
        Remove a LoRA adapter from a specific layer.
        
        Args:
            layer_idx: Target layer index
            adapter_name: Name of the adapter to remove
            
        Returns:
            True if removal successful
        """
        try:
            # Remove from active adapters
            if layer_idx in self.active_adapters and adapter_name in self.active_adapters[layer_idx]:
                del self.active_adapters[layer_idx][adapter_name]
            
            # Remove LoRA layers
            target_modules = config.get('injection.target_modules', ['attn.c_attn', 'mlp.c_fc'])
            
            for module_name in target_modules:
                lora_key = f"{adapter_name}_{layer_idx}_{module_name}"
                if lora_key in self.lora_layers:
                    del self.lora_layers[lora_key]
            
            # Clean up empty layer entries
            if layer_idx in self.active_adapters and not self.active_adapters[layer_idx]:
                del self.active_adapters[layer_idx]
            
            logger.info(f"Successfully removed adapter {adapter_name} from layer {layer_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove adapter {adapter_name}: {e}")
            return False
    
    def clear_all_adapters(self) -> None:
        """Remove all adapters and clean up."""
        # Clear all LoRA layers
        self.lora_layers.clear()
        
        # Clear active adapters
        self.active_adapters.clear()
        
        # Remove all hooks
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        
        logger.info("Cleared all adapters and hooks")
    
    def get_active_adapters(self) -> Dict[int, List[str]]:
        """
        Get currently active adapters.
        
        Returns:
            Dictionary mapping layer indices to adapter names
        """
        return {layer_idx: list(adapters.keys()) 
                for layer_idx, adapters in self.active_adapters.items()}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage of LoRA layers.
        
        Returns:
            Dictionary with memory usage information
        """
        total_params = 0
        total_memory_mb = 0
        
        for lora_layer in self.lora_layers.values():
            layer_params = sum(p.numel() for p in lora_layer.parameters())
            total_params += layer_params
            
            # Estimate memory (assuming fp32)
            layer_memory_mb = layer_params * 4 / (1024**2)
            total_memory_mb += layer_memory_mb
        
        return {
            'total_lora_layers': len(self.lora_layers),
            'total_parameters': total_params,
            'memory_mb': total_memory_mb,
            'memory_gb': total_memory_mb / 1024
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_all_adapters()
