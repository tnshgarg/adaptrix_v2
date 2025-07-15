"""
Adapter manager for the Adaptrix CLI.

This module provides functionality for downloading, installing, and managing adapters.
"""

import os
import sys
import json
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import adapter managers with error handling
try:
    from src.adapters.adapter_manager import AdapterManager
    ADAPTER_MANAGER_AVAILABLE = True
except ImportError:
    AdapterManager = None
    ADAPTER_MANAGER_AVAILABLE = False

try:
    from src.core.universal_adapter_manager import UniversalAdapterManager
    UNIVERSAL_ADAPTER_MANAGER_AVAILABLE = True
except ImportError:
    UniversalAdapterManager = None
    UNIVERSAL_ADAPTER_MANAGER_AVAILABLE = False

# Check if any adapter manager is available
ADAPTER_SYSTEM_AVAILABLE = ADAPTER_MANAGER_AVAILABLE or UNIVERSAL_ADAPTER_MANAGER_AVAILABLE

if not ADAPTER_SYSTEM_AVAILABLE:
    logger = logging.getLogger("adapter_manager")
    logger.warning("No adapter managers available, using mock implementation")

    # Mock implementation for when adapter managers are not available
    class MockAdapterManager:
        def __init__(self, *args, **kwargs):
            pass

        def list_adapters(self):
            return []

        def load_adapter(self, *args, **kwargs):
            return None

    AdapterManager = MockAdapterManager
    UniversalAdapterManager = MockAdapterManager
from src.cli.utils.logging import get_logger
from src.cli.utils.progress import download_with_progress, ProgressBar

logger = get_logger("adapter_manager")

class CLIAdapterManager:
    """
    Manages adapter downloading, installation, and management for the CLI.
    """
    
    def __init__(self, config_manager):
        """
        Initialize CLI adapter manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.adapters_dir = Path(self.config.get("adapters.directory"))
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core adapter manager if available
        if ADAPTER_MANAGER_AVAILABLE:
            try:
                self.core_adapter_manager = AdapterManager(str(self.adapters_dir))
            except Exception as e:
                logger.warning(f"Failed to initialize AdapterManager: {e}")
                self.core_adapter_manager = None
        elif UNIVERSAL_ADAPTER_MANAGER_AVAILABLE:
            try:
                # UniversalAdapterManager requires a base model, so we'll initialize it later
                self.core_adapter_manager = None
            except Exception as e:
                logger.warning(f"Failed to initialize UniversalAdapterManager: {e}")
                self.core_adapter_manager = None
        else:
            self.core_adapter_manager = None
        
        # Load builtin adapters registry
        self.builtin_adapters = self._load_builtin_adapters()
        
        logger.info(f"CLIAdapterManager initialized with directory: {self.adapters_dir}")
    
    def _load_builtin_adapters(self) -> List[Dict[str, Any]]:
        """Load builtin adapters from configuration."""
        return self.config.get("adapters.builtin_adapters", [])
    
    def list_available_adapters(self) -> List[Dict[str, Any]]:
        """
        List all available adapters (builtin + marketplace).
        
        Returns:
            List of adapter information dictionaries
        """
        adapters = []
        
        # Add builtin adapters
        for adapter in self.builtin_adapters:
            adapter_data = {
                "name": adapter["name"],
                "domain": adapter.get("domain", "general"),
                "description": adapter.get("description", ""),
                "version": "builtin",
                "installed": self.is_adapter_installed(adapter["name"]),
                "source": "builtin"
            }
            adapters.append(adapter_data)
        
        # TODO: Add marketplace adapters
        # This would fetch from the adapter registry URL
        
        return adapters
    
    def list_installed_adapters(self) -> List[Dict[str, Any]]:
        """
        List all installed adapters.
        
        Returns:
            List of installed adapter information dictionaries
        """
        adapters = []
        
        for adapter_dir in self.adapters_dir.iterdir():
            if adapter_dir.is_dir():
                adapter_name = adapter_dir.name
                
                # Load adapter metadata
                metadata_path = adapter_dir / "metadata.json"
                metadata = {}
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading metadata for {adapter_name}: {e}")
                
                adapter_data = {
                    "name": adapter_name,
                    "domain": metadata.get("domain", "unknown"),
                    "description": metadata.get("description", ""),
                    "version": metadata.get("version", "unknown"),
                    "installed": True,
                    "path": str(adapter_dir),
                    "target_layers": metadata.get("target_layers", []),
                    "target_modules": metadata.get("target_modules", [])
                }
                adapters.append(adapter_data)
        
        return adapters
    
    def is_adapter_installed(self, adapter_name: str) -> bool:
        """
        Check if an adapter is installed.
        
        Args:
            adapter_name: Adapter name
        
        Returns:
            True if adapter is installed, False otherwise
        """
        adapter_path = self.adapters_dir / adapter_name
        return adapter_path.exists() and adapter_path.is_dir()
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an adapter.
        
        Args:
            adapter_name: Adapter name
        
        Returns:
            Adapter information dictionary or None if not found
        """
        # Check if adapter is installed
        if self.is_adapter_installed(adapter_name):
            adapter_path = self.adapters_dir / adapter_name
            metadata_path = adapter_path / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["installed"] = True
                    metadata["local_path"] = str(adapter_path)
                    return metadata
                except Exception as e:
                    logger.error(f"Error loading adapter metadata: {e}")
        
        # Check builtin adapters
        for adapter in self.builtin_adapters:
            if adapter["name"] == adapter_name:
                adapter_info = adapter.copy()
                adapter_info["installed"] = self.is_adapter_installed(adapter_name)
                adapter_info["source"] = "builtin"
                return adapter_info
        
        return None
    
    def install_adapter(self, adapter_name: str) -> bool:
        """
        Install an adapter from the builtin collection or marketplace.
        
        Args:
            adapter_name: Adapter name
        
        Returns:
            True if installation successful, False otherwise
        """
        try:
            # Check if it's a builtin adapter
            builtin_adapter = None
            for adapter in self.builtin_adapters:
                if adapter["name"] == adapter_name:
                    builtin_adapter = adapter
                    break
            
            if builtin_adapter:
                return self._install_builtin_adapter(adapter_name, builtin_adapter)
            else:
                # TODO: Install from marketplace
                logger.error(f"Marketplace installation not yet implemented for {adapter_name}")
                return False
        
        except Exception as e:
            logger.error(f"Error installing adapter {adapter_name}: {e}")
            return False
    
    def _install_builtin_adapter(self, adapter_name: str, adapter_info: Dict[str, Any]) -> bool:
        """
        Install a builtin adapter.
        
        Args:
            adapter_name: Adapter name
            adapter_info: Adapter information
        
        Returns:
            True if installation successful, False otherwise
        """
        try:
            # Create adapter directory
            adapter_dir = self.adapters_dir / adapter_name
            adapter_dir.mkdir(parents=True, exist_ok=True)
            
            # For builtin adapters, we'll create a placeholder structure
            # In a real implementation, these would be downloaded from a repository
            
            # Create metadata
            metadata = {
                "name": adapter_name,
                "description": adapter_info.get("description", ""),
                "domain": adapter_info.get("domain", "general"),
                "version": "1.0.0",
                "target_layers": [6, 12, 18],  # Default middle layers
                "target_modules": [
                    "self_attn.q_proj",
                    "self_attn.k_proj", 
                    "self_attn.v_proj",
                    "mlp.gate_proj"
                ],
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1
                },
                "source": "builtin",
                "install_date": str(Path().cwd())
            }
            
            # Save metadata
            metadata_path = adapter_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create placeholder weight files for each layer
            for layer_idx in metadata["target_layers"]:
                layer_file = adapter_dir / f"layer_{layer_idx}.pt"
                
                # Create empty file (in real implementation, this would be actual weights)
                layer_file.touch()
            
            logger.info(f"Successfully installed builtin adapter {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing builtin adapter {adapter_name}: {e}")
            
            # Clean up on failure
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            
            return False
    
    def install_from_path(self, adapter_name: str, source_path: str) -> bool:
        """
        Install an adapter from a local path.
        
        Args:
            adapter_name: Adapter name
            source_path: Path to adapter directory
        
        Returns:
            True if installation successful, False otherwise
        """
        try:
            source_path = Path(source_path)
            
            if not source_path.exists():
                logger.error(f"Source path does not exist: {source_path}")
                return False
            
            # Validate adapter structure
            is_valid, errors = self.validate_adapter_structure(str(source_path))
            if not is_valid:
                logger.error(f"Invalid adapter structure: {errors}")
                return False
            
            # Create adapter directory
            adapter_dir = self.adapters_dir / adapter_name
            
            # Remove existing if it exists
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            
            # Copy adapter
            shutil.copytree(source_path, adapter_dir)
            
            logger.info(f"Successfully installed adapter {adapter_name} from {source_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing adapter from path: {e}")
            return False
    
    def uninstall_adapter(self, adapter_name: str) -> bool:
        """
        Uninstall an adapter.
        
        Args:
            adapter_name: Adapter name
        
        Returns:
            True if uninstallation successful, False otherwise
        """
        try:
            adapter_dir = self.adapters_dir / adapter_name
            
            if not adapter_dir.exists():
                logger.warning(f"Adapter {adapter_name} is not installed")
                return True
            
            # Remove adapter directory
            shutil.rmtree(adapter_dir)
            
            logger.info(f"Successfully uninstalled adapter {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uninstalling adapter {adapter_name}: {e}")
            return False
    
    def validate_adapter_structure(self, adapter_path: str) -> Tuple[bool, List[str]]:
        """
        Validate adapter directory structure.
        
        Args:
            adapter_path: Path to adapter directory
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        adapter_path = Path(adapter_path)
        
        # Check if directory exists
        if not adapter_path.exists():
            errors.append("Adapter directory does not exist")
            return False, errors
        
        if not adapter_path.is_dir():
            errors.append("Adapter path is not a directory")
            return False, errors
        
        # Check for metadata file
        metadata_path = adapter_path / "metadata.json"
        if not metadata_path.exists():
            errors.append("metadata.json file is missing")
        else:
            # Validate metadata structure
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                required_fields = ["name", "target_layers", "target_modules"]
                for field in required_fields:
                    if field not in metadata:
                        errors.append(f"Missing required field in metadata: {field}")
                
            except Exception as e:
                errors.append(f"Invalid metadata.json: {e}")
        
        # Check for weight files (at least one layer file should exist)
        layer_files = list(adapter_path.glob("layer_*.pt"))
        if not layer_files:
            errors.append("No layer weight files found (layer_*.pt)")
        
        return len(errors) == 0, errors
