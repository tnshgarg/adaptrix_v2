"""
Configuration management for Adaptrix system.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AdaptrixConfig:
    """Configuration manager for Adaptrix system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "configs" / "default.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not found."""
        return {
            "model": {
                "name": "microsoft/DialoGPT-medium",
                "device": "auto",
                "precision": "fp16"
            },
            "injection": {
                "target_layers": [6, 12, 18],
                "target_modules": ["self_attn.q_proj", "mlp.c_fc"],
                "default_rank": 16,
                "default_alpha": 32
            },
            "adapters": {
                "cache_size": 3,
                "storage_path": "./adapters"
            },
            "performance": {
                "max_memory_gb": 8,
                "batch_size": 1,
                "max_length": 512
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.name')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses current config path.
        """
        save_path = path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
            """Recursively update nested dictionaries."""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        deep_update(self._config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.set(key, value)


# Global configuration instance
config = AdaptrixConfig()
