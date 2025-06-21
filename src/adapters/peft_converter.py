"""
PEFT/QLoRA adapter converter for Adaptrix system.

Converts existing HuggingFace PEFT adapters to work with middle-layer injection.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import safetensors
from transformers import AutoConfig

try:
    from peft import PeftConfig, PeftModel
    from peft.utils import get_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. Install with: pip install peft")

from ..utils.config import config
from ..utils.helpers import ensure_directory, safe_json_save, safe_json_load

logger = logging.getLogger(__name__)


class PEFTConverter:
    """
    Converts PEFT/QLoRA adapters to Adaptrix middle-layer format.
    
    Supports:
    - HuggingFace Hub adapters
    - Local PEFT adapters
    - Alpaca-LoRA variants
    - Custom trained adapters
    """
    
    def __init__(self, target_layers: Optional[List[int]] = None):
        """
        Initialize PEFT converter.
        
        Args:
            target_layers: Target layers for middle-layer injection
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required. Install with: pip install peft")
        
        self.target_layers = target_layers or config.get('injection.target_layers', [3, 6, 9])
        self.target_modules = config.get('injection.target_modules', ['attn.c_attn', 'mlp.c_fc'])
        
        logger.info(f"PEFTConverter initialized for layers {self.target_layers}")
    
    def convert_from_hub(self, adapter_id: str, output_dir: str, base_model_name: Optional[str] = None) -> bool:
        """
        Convert a PEFT adapter from HuggingFace Hub.
        
        Args:
            adapter_id: HuggingFace adapter ID (e.g., "microsoft/DialoGPT-medium-lora")
            output_dir: Output directory for converted adapter
            base_model_name: Base model name (auto-detected if None)
            
        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting PEFT adapter from Hub: {adapter_id}")
            
            # Load PEFT config
            peft_config = PeftConfig.from_pretrained(adapter_id)
            
            # Get base model name
            if base_model_name is None:
                base_model_name = peft_config.base_model_name_or_path
            
            logger.info(f"Base model: {base_model_name}")
            logger.info(f"PEFT type: {peft_config.peft_type}")
            
            # Load adapter weights
            adapter_weights = self._load_peft_weights(adapter_id)
            
            # Convert to Adaptrix format
            converted_data = self._convert_peft_to_adaptrix(
                adapter_weights, 
                peft_config, 
                base_model_name
            )
            
            # Save converted adapter
            return self._save_converted_adapter(converted_data, output_dir)
            
        except Exception as e:
            logger.error(f"Failed to convert adapter {adapter_id}: {e}")
            return False
    
    def convert_from_local(self, adapter_path: str, output_dir: str, base_model_name: str) -> bool:
        """
        Convert a local PEFT adapter.
        
        Args:
            adapter_path: Path to local PEFT adapter
            output_dir: Output directory for converted adapter
            base_model_name: Base model name
            
        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting local PEFT adapter: {adapter_path}")
            
            adapter_path = Path(adapter_path)
            
            # Load PEFT config
            config_path = adapter_path / "adapter_config.json"
            if not config_path.exists():
                logger.error(f"PEFT config not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                peft_config_dict = json.load(f)
            
            # Load adapter weights
            adapter_weights = self._load_local_peft_weights(adapter_path)
            
            # Convert to Adaptrix format
            converted_data = self._convert_peft_to_adaptrix(
                adapter_weights, 
                peft_config_dict, 
                base_model_name
            )
            
            # Save converted adapter
            return self._save_converted_adapter(converted_data, output_dir)
            
        except Exception as e:
            logger.error(f"Failed to convert local adapter {adapter_path}: {e}")
            return False
    
    def _load_peft_weights(self, adapter_id: str) -> Dict[str, torch.Tensor]:
        """Load PEFT weights from HuggingFace Hub."""
        try:
            # Try to load using PEFT
            from huggingface_hub import hf_hub_download
            
            # Download adapter weights
            weights_file = hf_hub_download(
                repo_id=adapter_id,
                filename="adapter_model.safetensors",
                cache_dir=config.get('model.cache_dir', './models')
            )
            
            # Load safetensors
            with safetensors.safe_open(weights_file, framework="pt", device="cpu") as f:
                weights = {key: f.get_tensor(key) for key in f.keys()}
            
            return weights
            
        except Exception as e:
            logger.warning(f"Failed to load safetensors, trying pytorch: {e}")
            
            # Fallback to pytorch format
            try:
                weights_file = hf_hub_download(
                    repo_id=adapter_id,
                    filename="adapter_model.bin",
                    cache_dir=config.get('model.cache_dir', './models')
                )
                
                weights = torch.load(weights_file, map_location="cpu")
                return weights
                
            except Exception as e2:
                logger.error(f"Failed to load adapter weights: {e2}")
                raise
    
    def _load_local_peft_weights(self, adapter_path: Path) -> Dict[str, torch.Tensor]:
        """Load PEFT weights from local directory."""
        # Try safetensors first
        safetensors_path = adapter_path / "adapter_model.safetensors"
        if safetensors_path.exists():
            with safetensors.safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                return {key: f.get_tensor(key) for key in f.keys()}
        
        # Fallback to pytorch
        pytorch_path = adapter_path / "adapter_model.bin"
        if pytorch_path.exists():
            return torch.load(pytorch_path, map_location="cpu")
        
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
    
    def _convert_peft_to_adaptrix(self, 
                                  peft_weights: Dict[str, torch.Tensor], 
                                  peft_config: Dict[str, Any], 
                                  base_model_name: str) -> Dict[str, Any]:
        """
        Convert PEFT weights to Adaptrix middle-layer format.
        
        Args:
            peft_weights: PEFT adapter weights
            peft_config: PEFT configuration
            base_model_name: Base model name
            
        Returns:
            Converted adapter data
        """
        logger.info("Converting PEFT weights to Adaptrix format...")
        
        # Extract PEFT parameters
        if isinstance(peft_config, dict):
            rank = peft_config.get('r', 16)
            alpha = peft_config.get('lora_alpha', 32)
            target_modules = peft_config.get('target_modules', [])
        else:
            rank = getattr(peft_config, 'r', 16)
            alpha = getattr(peft_config, 'lora_alpha', 32)
            target_modules = getattr(peft_config, 'target_modules', [])
        
        # Get model architecture info
        model_info = self._get_model_architecture_info(base_model_name)
        total_layers = model_info['num_layers']
        
        # Map PEFT target modules to Adaptrix target modules
        module_mapping = self._create_module_mapping(target_modules, model_info['architecture'])
        
        # Distribute weights across target layers
        converted_weights = {}
        
        for layer_idx in self.target_layers:
            if layer_idx >= total_layers:
                logger.warning(f"Target layer {layer_idx} exceeds model layers ({total_layers})")
                continue
            
            layer_weights = {}
            
            for adaptrix_module in self.target_modules:
                # Find corresponding PEFT weights
                peft_module = module_mapping.get(adaptrix_module)
                if not peft_module:
                    continue
                
                # Look for LoRA A and B weights for this layer and module
                lora_A_key = f"base_model.model.{model_info['layer_prefix']}.{layer_idx}.{peft_module}.lora_A.weight"
                lora_B_key = f"base_model.model.{model_info['layer_prefix']}.{layer_idx}.{peft_module}.lora_B.weight"
                
                # Alternative key patterns
                alt_patterns = [
                    f"{model_info['layer_prefix']}.{layer_idx}.{peft_module}.lora_A.weight",
                    f"model.{model_info['layer_prefix']}.{layer_idx}.{peft_module}.lora_A.weight",
                ]
                
                lora_A = None
                lora_B = None
                
                # Try to find LoRA weights
                for key in peft_weights.keys():
                    if lora_A_key in key or any(pattern in key for pattern in alt_patterns):
                        if "lora_A" in key and str(layer_idx) in key and peft_module in key:
                            lora_A = peft_weights[key]
                        elif "lora_B" in key and str(layer_idx) in key and peft_module in key:
                            lora_B = peft_weights[key]
                
                # If we found both A and B weights, add to layer weights
                if lora_A is not None and lora_B is not None:
                    # Get expected dimensions for this module in the target architecture
                    expected_dims = model_info.get('module_dimensions', {}).get(adaptrix_module)

                    if expected_dims:
                        expected_in, expected_out = expected_dims

                        # Check if dimensions match, if not, adjust
                        if lora_A.shape[1] != expected_in or lora_B.shape[0] != expected_out:
                            logger.warning(f"Dimension mismatch for {adaptrix_module}: "
                                         f"LoRA A: {lora_A.shape}, B: {lora_B.shape}, "
                                         f"Expected: in={expected_in}, out={expected_out}")

                            # Adjust dimensions if needed
                            if lora_A.shape[1] != expected_in:
                                # Resize A matrix input dimension
                                if lora_A.shape[1] < expected_in:
                                    # Pad with zeros
                                    padding = torch.zeros(lora_A.shape[0], expected_in - lora_A.shape[1])
                                    lora_A = torch.cat([lora_A, padding], dim=1)
                                else:
                                    # Truncate
                                    lora_A = lora_A[:, :expected_in]

                            if lora_B.shape[0] != expected_out:
                                # Resize B matrix output dimension
                                if lora_B.shape[0] < expected_out:
                                    # Pad with zeros
                                    padding = torch.zeros(expected_out - lora_B.shape[0], lora_B.shape[1])
                                    lora_B = torch.cat([lora_B, padding], dim=0)
                                else:
                                    # Truncate
                                    lora_B = lora_B[:expected_out, :]

                    layer_weights[adaptrix_module] = {
                        'lora_A': lora_A,
                        'lora_B': lora_B,
                        'rank': rank,
                        'alpha': alpha
                    }
                    logger.debug(f"Converted {peft_module} -> {adaptrix_module} for layer {layer_idx}")
                else:
                    # If original layer doesn't have weights, redistribute from available layers
                    redistributed_weights = self._redistribute_weights(
                        peft_weights, peft_module, layer_idx, rank, alpha
                    )
                    if redistributed_weights:
                        layer_weights[adaptrix_module] = redistributed_weights
            
            if layer_weights:
                converted_weights[layer_idx] = layer_weights
        
        # Create metadata
        metadata = {
            'name': f"converted_peft_adapter",
            'version': '1.0.0',
            'description': f"Converted from PEFT adapter for {base_model_name}",
            'source': 'peft_conversion',
            'base_model': base_model_name,
            'target_layers': list(converted_weights.keys()),
            'target_modules': self.target_modules,
            'rank': rank,
            'alpha': alpha,
            'original_peft_config': peft_config
        }
        
        return {
            'metadata': metadata,
            'weights': converted_weights
        }
    
    def _get_model_architecture_info(self, model_name: str) -> Dict[str, Any]:
        """Get model architecture information."""
        try:
            config_obj = AutoConfig.from_pretrained(model_name)
            
            # Determine architecture type and layer info
            if hasattr(config_obj, 'n_layer'):  # GPT-2 style
                num_layers = config_obj.n_layer
                layer_prefix = "h"
                architecture = "gpt2"
            elif hasattr(config_obj, 'num_hidden_layers'):  # BERT/RoBERTa style
                num_layers = config_obj.num_hidden_layers
                layer_prefix = "layers"
                architecture = "bert"
            else:
                # Default fallback
                num_layers = 12
                layer_prefix = "h"
                architecture = "unknown"
            
            # Get module dimensions using architecture registry
            from ..models.architecture_registry import architecture_registry
            arch = architecture_registry.get_architecture(model_name)
            module_dimensions = arch.get_module_dimensions(config_obj)

            return {
                'num_layers': num_layers,
                'layer_prefix': layer_prefix,
                'architecture': architecture,
                'model_type': getattr(config_obj, 'model_type', 'unknown'),
                'module_dimensions': module_dimensions
            }
            
        except Exception as e:
            logger.warning(f"Could not load model config for {model_name}: {e}")
            return {
                'num_layers': 12,
                'layer_prefix': "h",
                'architecture': "unknown",
                'model_type': 'unknown'
            }
    
    def _create_module_mapping(self, peft_target_modules: List[str], architecture: str) -> Dict[str, str]:
        """Create mapping from Adaptrix modules to PEFT modules."""
        mapping = {}
        
        # Common mappings based on architecture
        if architecture in ["gpt2", "unknown"]:
            mapping = {
                'attn.c_attn': 'attn.c_attn',
                'attn.c_proj': 'attn.c_proj',
                'mlp.c_fc': 'mlp.c_fc',
                'mlp.c_proj': 'mlp.c_proj'
            }
        elif architecture == "bert":
            mapping = {
                'attn.c_attn': 'attention.self.query',  # Map to query for simplicity
                'mlp.c_fc': 'intermediate.dense'
            }
        
        # Filter by what's actually in PEFT target modules
        filtered_mapping = {}
        for adaptrix_module, peft_module in mapping.items():
            if any(peft_module in target for target in peft_target_modules):
                filtered_mapping[adaptrix_module] = peft_module
        
        return filtered_mapping
    
    def _redistribute_weights(self, 
                            peft_weights: Dict[str, torch.Tensor], 
                            module_name: str, 
                            target_layer: int, 
                            rank: int, 
                            alpha: float) -> Optional[Dict[str, Any]]:
        """
        Redistribute weights from available layers to target layer.
        
        This is used when the original PEFT adapter doesn't have weights
        for the specific target layer we want to inject into.
        """
        # Find any available weights for this module
        available_weights = []
        
        for key, weight in peft_weights.items():
            if module_name in key and "lora_A" in key:
                layer_match = self._extract_layer_number(key)
                if layer_match is not None:
                    # Find corresponding B weight
                    b_key = key.replace("lora_A", "lora_B")
                    if b_key in peft_weights:
                        available_weights.append({
                            'layer': layer_match,
                            'lora_A': weight,
                            'lora_B': peft_weights[b_key]
                        })
        
        if not available_weights:
            return None
        
        # Use the closest layer or average if multiple layers
        if len(available_weights) == 1:
            weights = available_weights[0]
        else:
            # Find closest layer
            closest = min(available_weights, key=lambda x: abs(x['layer'] - target_layer))
            weights = closest
        
        return {
            'lora_A': weights['lora_A'].clone(),
            'lora_B': weights['lora_B'].clone(),
            'rank': rank,
            'alpha': alpha
        }
    
    def _extract_layer_number(self, key: str) -> Optional[int]:
        """Extract layer number from weight key."""
        import re
        match = re.search(r'\.(\d+)\.', key)
        return int(match.group(1)) if match else None
    
    def _save_converted_adapter(self, converted_data: Dict[str, Any], output_dir: str) -> bool:
        """Save converted adapter to Adaptrix format."""
        try:
            output_path = Path(output_dir)
            ensure_directory(output_path)
            
            metadata = converted_data['metadata']
            weights = converted_data['weights']
            
            # Save metadata
            metadata_path = output_path / "metadata.json"
            if not safe_json_save(metadata, metadata_path):
                return False
            
            # Save weights for each layer
            for layer_idx, layer_weights in weights.items():
                layer_file = output_path / f"layer_{layer_idx}.pt"
                torch.save(layer_weights, layer_file)
            
            logger.info(f"Successfully saved converted adapter to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save converted adapter: {e}")
            return False
