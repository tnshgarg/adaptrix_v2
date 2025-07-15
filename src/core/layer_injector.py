"""
Layer injection system for dynamic LoRA adapter loading.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class LayerInjector:
    """Manages injection of LoRA adapters into model layers."""
    
    def __init__(self, model):
        self.model = model
        self.injection_points = {}  # {(layer_idx, module_name): original_module}
        self.active_adapters = {}   # {adapter_name: {layer_idx: {module_name: adapter_data}}}
        self.target_modules = []
        
    def set_target_modules(self, target_modules: List[str]):
        """Set the target modules for injection."""
        self.target_modules = target_modules
        
    def register_injection_point(self, layer_idx: int, module_name: str):
        """Register a point where adapters can be injected."""
        try:
            # Find the module in the model
            module = self._get_module_by_path(layer_idx, module_name)
            if module is not None:
                self.injection_points[(layer_idx, module_name)] = module
                logger.debug(f"Registered injection point: layer {layer_idx}, module {module_name}")
            else:
                logger.warning(f"Could not find module {module_name} in layer {layer_idx}")
        except Exception as e:
            logger.error(f"Failed to register injection point {layer_idx}.{module_name}: {e}")
    
    def _get_module_by_path(self, layer_idx: int, module_name: str):
        """Get a module by its path in the model."""
        try:
            # Common patterns for different model architectures
            layer_patterns = [
                f"model.layers.{layer_idx}",  # Most common
                f"transformer.h.{layer_idx}",  # GPT-style
                f"layers.{layer_idx}",  # Direct access
            ]
            
            layer = None
            for pattern in layer_patterns:
                try:
                    layer = self._get_nested_attr(self.model, pattern)
                    if layer is not None:
                        break
                except:
                    continue
            
            if layer is None:
                return None
            
            # Get the specific module
            return self._get_nested_attr(layer, module_name)
            
        except Exception as e:
            logger.debug(f"Could not get module {module_name} from layer {layer_idx}: {e}")
            return None
    
    def _get_nested_attr(self, obj, attr_path: str):
        """Get nested attribute by dot-separated path."""
        attrs = attr_path.split('.')
        for attr in attrs:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj
    
    def inject_adapter(self, adapter_name: str, layer_idx: int, module_name: str, adapter_data: Dict[str, Any]):
        """Inject an adapter into a specific layer and module."""
        try:
            injection_key = (layer_idx, module_name)
            if injection_key not in self.injection_points:
                logger.warning(f"Injection point {layer_idx}.{module_name} not registered")
                return False

            # Get the actual module from the model
            target_module = self._get_module_by_path(layer_idx, module_name)
            if target_module is None:
                logger.error(f"Could not find module {module_name} in layer {layer_idx}")
                return False

            # Apply LoRA weights to the module
            if not self._apply_lora_weights(target_module, adapter_data, adapter_name):
                logger.error(f"Failed to apply LoRA weights to {layer_idx}.{module_name}")
                return False

            # Store adapter data for tracking
            if adapter_name not in self.active_adapters:
                self.active_adapters[adapter_name] = {}
            if layer_idx not in self.active_adapters[adapter_name]:
                self.active_adapters[adapter_name][layer_idx] = {}

            self.active_adapters[adapter_name][layer_idx][module_name] = {
                'adapter_data': adapter_data,
                'original_module': target_module,
                'applied': True
            }

            logger.debug(f"Successfully injected and applied adapter {adapter_name} to layer {layer_idx}, module {module_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to inject adapter {adapter_name}: {e}")
            return False
    
    def remove_adapter(self, adapter_name: str):
        """Remove an adapter from all injection points."""
        try:
            if adapter_name not in self.active_adapters:
                logger.warning(f"Adapter {adapter_name} not found in active adapters")
                return False

            # Remove LoRA weights from all modules
            for layer_idx, layer_data in self.active_adapters[adapter_name].items():
                for module_name, module_info in layer_data.items():
                    if module_info.get('applied', False):
                        target_module = self._get_module_by_path(layer_idx, module_name)
                        if target_module is not None:
                            self._remove_lora_weights(target_module, adapter_name)

            # Remove from active adapters
            del self.active_adapters[adapter_name]
            logger.debug(f"Removed adapter {adapter_name} and restored original weights")
            return True

        except Exception as e:
            logger.error(f"Failed to remove adapter {adapter_name}: {e}")
            return False
    
    def get_active_adapters(self) -> Dict[str, Any]:
        """Get information about currently active adapters."""
        return {
            adapter_name: {
                "layers": list(adapter_data.keys()),
                "modules": [list(layer_data.keys()) for layer_data in adapter_data.values()]
            }
            for adapter_name, adapter_data in self.active_adapters.items()
        }
    
    def clear_all_adapters(self):
        """Clear all active adapters."""
        self.active_adapters.clear()
        logger.info("Cleared all adapters")
    
    def _apply_lora_weights(self, target_module, adapter_data: Dict[str, Any], adapter_name: str) -> bool:
        """Apply LoRA weights to a target module."""
        try:
            if not hasattr(target_module, 'weight'):
                logger.warning(f"Target module has no weight parameter")
                return False

            # Get LoRA matrices
            lora_A = adapter_data.get('lora_A')
            lora_B = adapter_data.get('lora_B')
            scaling = adapter_data.get('scaling', 1.0)

            if lora_A is None or lora_B is None:
                logger.warning(f"Missing LoRA matrices in adapter data")
                return False

            # Ensure tensors are on the same device as the target module
            device = target_module.weight.device
            dtype = target_module.weight.dtype

            lora_A = lora_A.to(device=device, dtype=dtype)
            lora_B = lora_B.to(device=device, dtype=dtype)

            # Calculate LoRA delta: scaling * B @ A
            with torch.no_grad():
                lora_delta = scaling * (lora_B @ lora_A)

                # Store original weights if not already stored
                if not hasattr(target_module, f'_original_weight_{adapter_name}'):
                    setattr(target_module, f'_original_weight_{adapter_name}', target_module.weight.data.clone())

                # Apply LoRA delta to the module weights
                target_module.weight.data += lora_delta

                # Mark module as having LoRA applied
                if not hasattr(target_module, '_lora_adapters'):
                    target_module._lora_adapters = set()
                target_module._lora_adapters.add(adapter_name)

            logger.debug(f"Applied LoRA weights for adapter {adapter_name}")
            return True

        except Exception as e:
            logger.error(f"Error applying LoRA weights: {e}")
            return False

    def _remove_lora_weights(self, target_module, adapter_name: str) -> bool:
        """Remove LoRA weights from a target module."""
        try:
            if not hasattr(target_module, f'_original_weight_{adapter_name}'):
                logger.debug(f"No original weights stored for adapter {adapter_name} - may already be removed")
                return True  # Consider this success if already removed

            # Restore original weights
            with torch.no_grad():
                original_weight = getattr(target_module, f'_original_weight_{adapter_name}')
                target_module.weight.data = original_weight.clone()

                # Clean up stored weights
                delattr(target_module, f'_original_weight_{adapter_name}')

                # Update LoRA adapter tracking
                if hasattr(target_module, '_lora_adapters'):
                    target_module._lora_adapters.discard(adapter_name)
                    if not target_module._lora_adapters:
                        delattr(target_module, '_lora_adapters')

            logger.debug(f"Removed LoRA weights for adapter {adapter_name}")
            return True

        except Exception as e:
            logger.error(f"Error removing LoRA weights: {e}")
            return False

    def get_injection_points(self) -> List[Tuple[int, str]]:
        """Get all registered injection points."""
        return list(self.injection_points.keys())

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for injected adapters."""
        total_params = 0
        adapter_count = 0

        for adapter_data in self.active_adapters.values():
            adapter_count += 1
            for layer_data in adapter_data.values():
                for module_data in layer_data.values():
                    if isinstance(module_data, dict):
                        if 'lora_A' in module_data and hasattr(module_data['lora_A'], 'numel'):
                            total_params += module_data['lora_A'].numel()
                        if 'lora_B' in module_data and hasattr(module_data['lora_B'], 'numel'):
                            total_params += module_data['lora_B'].numel()

        memory_mb = total_params * 4 / (1024**2)  # Assume fp32

        return {
            'total_parameters': total_params,
            'memory_mb': memory_mb,
            'memory_gb': memory_mb / 1024,
            'active_adapters': adapter_count,
            'injection_points': len(self.injection_points)
        }
