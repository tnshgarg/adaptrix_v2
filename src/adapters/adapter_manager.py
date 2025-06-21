"""
Adapter management system for Adaptrix.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from ..utils.config import config
from ..utils.helpers import ensure_directory, safe_json_load, safe_json_save, validate_adapter_structure

logger = logging.getLogger(__name__)


class AdapterManager:
    """
    Manages LoRA adapters for the Adaptrix system.
    
    Handles loading, saving, validation, and metadata management
    for LoRA adapters stored in the adapter library.
    """
    
    def __init__(self, adapter_dir: Optional[str] = None):
        """
        Initialize adapter manager.
        
        Args:
            adapter_dir: Directory to store adapters
        """
        self.adapter_dir = Path(adapter_dir or config.get('adapters.storage_path', './adapters'))
        ensure_directory(self.adapter_dir)
        
        # Cache for loaded adapters
        self._adapter_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"AdapterManager initialized with directory: {self.adapter_dir}")
    
    def load_adapter(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Load an adapter from storage.
        
        Args:
            adapter_name: Name of the adapter to load
            
        Returns:
            Adapter data dictionary or None if not found
        """
        # Check cache first
        if adapter_name in self._adapter_cache:
            logger.debug(f"Loading adapter {adapter_name} from cache")
            return self._adapter_cache[adapter_name]
        
        adapter_path = self.adapter_dir / adapter_name
        
        if not adapter_path.exists():
            logger.error(f"Adapter {adapter_name} not found at {adapter_path}")
            return None
        
        try:
            # Load metadata
            metadata_path = adapter_path / "metadata.json"
            metadata = safe_json_load(metadata_path)
            
            if metadata is None:
                logger.error(f"Failed to load metadata for adapter {adapter_name}")
                return None
            
            # Validate metadata
            validation_errors = validate_adapter_structure(metadata)
            if validation_errors:
                logger.error(f"Invalid adapter metadata for {adapter_name}: {validation_errors}")
                return None
            
            # Load weights for each target layer
            adapter_data = {
                'metadata': metadata,
                'weights': {}
            }
            
            target_layers = metadata.get('target_layers', [])
            target_modules = metadata.get('target_modules', ['self_attn.q_proj', 'mlp.c_fc'])
            
            for layer_idx in target_layers:
                layer_file = adapter_path / f"layer_{layer_idx}.pt"
                
                if layer_file.exists():
                    try:
                        layer_weights = torch.load(layer_file, map_location='cpu')
                        
                        # Validate layer weights structure
                        layer_data = {}
                        for module_name in target_modules:
                            if module_name in layer_weights:
                                module_weights = layer_weights[module_name]
                                
                                # Ensure required LoRA components exist
                                if 'lora_A' in module_weights and 'lora_B' in module_weights:
                                    layer_data[module_name] = {
                                        'lora_A': module_weights['lora_A'],
                                        'lora_B': module_weights['lora_B'],
                                        'rank': module_weights.get('rank', metadata.get('rank', 16)),
                                        'alpha': module_weights.get('alpha', metadata.get('alpha', 32))
                                    }
                                else:
                                    logger.warning(f"Missing LoRA weights for {adapter_name} layer {layer_idx} module {module_name}")
                        
                        if layer_data:
                            adapter_data['weights'][layer_idx] = layer_data
                        
                    except Exception as e:
                        logger.error(f"Failed to load weights for layer {layer_idx}: {e}")
                        continue
                else:
                    logger.warning(f"Layer file not found: {layer_file}")
            
            if not adapter_data['weights']:
                logger.error(f"No valid weights found for adapter {adapter_name}")
                return None
            
            # Cache the adapter
            self._adapter_cache[adapter_name] = adapter_data
            
            logger.info(f"Successfully loaded adapter {adapter_name}")
            return adapter_data
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return None
    
    def save_adapter(self, adapter_name: str, weights: Dict[int, Dict[str, Any]], metadata: Dict[str, Any]) -> bool:
        """
        Save an adapter to storage.
        
        Args:
            adapter_name: Name of the adapter
            weights: Dictionary mapping layer indices to weight dictionaries
            metadata: Adapter metadata
            
        Returns:
            True if save successful
        """
        try:
            adapter_path = self.adapter_dir / adapter_name
            ensure_directory(adapter_path)
            
            # Add creation timestamp to metadata
            metadata['created_date'] = datetime.now().isoformat()
            metadata['name'] = adapter_name
            
            # Validate metadata
            validation_errors = validate_adapter_structure(metadata)
            if validation_errors:
                logger.error(f"Invalid adapter metadata: {validation_errors}")
                return False
            
            # Save metadata
            metadata_path = adapter_path / "metadata.json"
            if not safe_json_save(metadata, metadata_path):
                logger.error(f"Failed to save metadata for {adapter_name}")
                return False
            
            # Save weights for each layer
            for layer_idx, layer_weights in weights.items():
                layer_file = adapter_path / f"layer_{layer_idx}.pt"
                
                try:
                    torch.save(layer_weights, layer_file)
                    logger.debug(f"Saved weights for layer {layer_idx}")
                except Exception as e:
                    logger.error(f"Failed to save weights for layer {layer_idx}: {e}")
                    return False
            
            # Update cache
            adapter_data = {
                'metadata': metadata,
                'weights': weights
            }
            self._adapter_cache[adapter_name] = adapter_data
            
            logger.info(f"Successfully saved adapter {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save adapter {adapter_name}: {e}")
            return False
    
    def list_adapters(self) -> List[str]:
        """
        List all available adapters.
        
        Returns:
            List of adapter names
        """
        adapters = []
        
        try:
            for item in self.adapter_dir.iterdir():
                if item.is_dir():
                    metadata_file = item / "metadata.json"
                    if metadata_file.exists():
                        adapters.append(item.name)
            
            adapters.sort()
            logger.debug(f"Found {len(adapters)} adapters")
            return adapters
            
        except Exception as e:
            logger.error(f"Failed to list adapters: {e}")
            return []
    
    def get_adapter_metadata(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Adapter metadata or None if not found
        """
        # Check cache first
        if adapter_name in self._adapter_cache:
            return self._adapter_cache[adapter_name]['metadata'].copy()
        
        adapter_path = self.adapter_dir / adapter_name / "metadata.json"
        return safe_json_load(adapter_path)
    
    def delete_adapter(self, adapter_name: str) -> bool:
        """
        Delete an adapter from storage.
        
        Args:
            adapter_name: Name of the adapter to delete
            
        Returns:
            True if deletion successful
        """
        try:
            adapter_path = self.adapter_dir / adapter_name
            
            if not adapter_path.exists():
                logger.warning(f"Adapter {adapter_name} not found")
                return False
            
            # Remove from cache
            if adapter_name in self._adapter_cache:
                del self._adapter_cache[adapter_name]
            
            # Delete directory and all contents
            import shutil
            shutil.rmtree(adapter_path)
            
            logger.info(f"Successfully deleted adapter {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete adapter {adapter_name}: {e}")
            return False
    
    def validate_adapter(self, adapter_data: Dict[str, Any]) -> bool:
        """
        Validate adapter data structure.
        
        Args:
            adapter_data: Adapter data to validate
            
        Returns:
            True if valid
        """
        try:
            # Validate metadata
            if 'metadata' not in adapter_data:
                logger.error("Missing metadata in adapter data")
                return False
            
            metadata = adapter_data['metadata']
            validation_errors = validate_adapter_structure(metadata)
            
            if validation_errors:
                logger.error(f"Metadata validation errors: {validation_errors}")
                return False
            
            # Validate weights structure
            if 'weights' not in adapter_data:
                logger.error("Missing weights in adapter data")
                return False
            
            weights = adapter_data['weights']
            target_layers = metadata.get('target_layers', [])
            target_modules = metadata.get('target_modules', ['self_attn.q_proj', 'mlp.c_fc'])
            
            for layer_idx in target_layers:
                if layer_idx not in weights:
                    logger.error(f"Missing weights for layer {layer_idx}")
                    return False
                
                layer_weights = weights[layer_idx]
                
                for module_name in target_modules:
                    if module_name in layer_weights:
                        module_weights = layer_weights[module_name]
                        
                        # Check for required LoRA components
                        if 'lora_A' not in module_weights or 'lora_B' not in module_weights:
                            logger.error(f"Missing LoRA weights for layer {layer_idx} module {module_name}")
                            return False
                        
                        # Validate tensor shapes
                        lora_A = module_weights['lora_A']
                        lora_B = module_weights['lora_B']
                        
                        if not isinstance(lora_A, torch.Tensor) or not isinstance(lora_B, torch.Tensor):
                            logger.error(f"LoRA weights must be tensors for layer {layer_idx} module {module_name}")
                            return False
                        
                        # Check rank consistency
                        rank = module_weights.get('rank', metadata.get('rank', 16))
                        if lora_A.shape[0] != rank or lora_B.shape[1] != rank:
                            logger.error(f"Rank mismatch for layer {layer_idx} module {module_name}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about an adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Adapter information dictionary
        """
        metadata = self.get_adapter_metadata(adapter_name)
        if metadata is None:
            return None
        
        adapter_path = self.adapter_dir / adapter_name
        
        # Calculate storage size
        total_size = 0
        file_count = 0
        
        for file_path in adapter_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        info = {
            'name': adapter_name,
            'metadata': metadata,
            'storage_size_bytes': total_size,
            'storage_size_mb': total_size / (1024**2),
            'file_count': file_count,
            'path': str(adapter_path),
            'cached': adapter_name in self._adapter_cache
        }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the adapter cache."""
        self._adapter_cache.clear()
        logger.info("Adapter cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the adapter cache.
        
        Returns:
            Cache information dictionary
        """
        return {
            'cached_adapters': list(self._adapter_cache.keys()),
            'cache_size': len(self._adapter_cache),
            'max_cache_size': config.get('adapters.cache_size', 3)
        }
