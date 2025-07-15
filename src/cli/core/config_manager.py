"""
Configuration manager for the Adaptrix CLI.

This module provides a class for managing CLI configuration,
including loading, saving, and accessing configuration values.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "models": {
        "directory": "~/.adaptrix/models",
        "max_size_gb": 10,
        "default_model": "qwen/qwen3-1.7b",
        "auto_download": True
    },
    "adapters": {
        "directory": "~/.adaptrix/adapters",
        "registry_url": "https://adaptrix.ai/api/adapters",
        "auto_discover": True
    },
    "rag": {
        "directory": "~/.adaptrix/rag",
        "vector_store_type": "faiss",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    "inference": {
        "device": "auto",
        "max_memory": "4GB",
        "precision": "fp16",
        "max_tokens": 1024,
        "temperature": 0.7
    },
    "logging": {
        "directory": "~/.adaptrix/logs",
        "level": "INFO",
        "max_files": 10
    },
    "ui": {
        "color": True,
        "progress_bars": True,
        "rich_output": True
    }
}

class ConfigManager:
    """
    Configuration manager for the Adaptrix CLI.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files (default: ~/.adaptrix)
        """
        # Set up configuration directory
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".adaptrix"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up configuration file paths
        self.global_config_path = self.config_dir / "config.yaml"
        self.project_config_path = Path.cwd() / ".adaptrix" / "config.yaml"
        
        # Initialize configuration
        self.config = DEFAULT_CONFIG.copy()
        
        # Expand paths
        self._expand_paths()
    
    def _expand_paths(self):
        """Expand all paths in the configuration."""
        # Expand model directory
        models_dir = self.config["models"]["directory"]
        self.config["models"]["directory"] = os.path.expanduser(models_dir)
        
        # Expand adapter directory
        adapters_dir = self.config["adapters"]["directory"]
        self.config["adapters"]["directory"] = os.path.expanduser(adapters_dir)
        
        # Expand RAG directory
        rag_dir = self.config["rag"]["directory"]
        self.config["rag"]["directory"] = os.path.expanduser(rag_dir)
        
        # Expand logging directory
        log_dir = self.config["logging"]["directory"]
        self.config["logging"]["directory"] = os.path.expanduser(log_dir)
    
    def load_default_config(self):
        """Load default configuration."""
        # Load global configuration if it exists
        if self.global_config_path.exists():
            self.load_config(self.global_config_path)
        
        # Load project configuration if it exists
        if self.project_config_path.exists():
            self.load_config(self.project_config_path)
    
    def load_config(self, config_path: str):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                # Update configuration recursively
                self._update_config_recursive(self.config, config_data)
                
                # Re-expand paths
                self._expand_paths()
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
    
    def _update_config_recursive(self, target: Dict, source: Dict):
        """
        Update configuration recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, config_path: Optional[str] = None):
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save configuration (default: global config path)
        """
        config_path = Path(config_path) if config_path else self.global_config_path
        
        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: New value
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        # Re-expand paths if necessary
        if "directory" in key_path:
            self._expand_paths()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def reset(self):
        """Reset configuration to defaults."""
        self.config = DEFAULT_CONFIG.copy()
        self._expand_paths()
