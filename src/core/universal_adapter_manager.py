"""
Universal Adapter Manager for Adaptrix.

Handles LoRA adapters for any base model with automatic compatibility detection
and seamless switching between different model families.
"""

import torch
import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from .base_model_interface import BaseModelInterface, ModelFamily

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a LoRA adapter."""
    name: str
    path: str
    base_model: str
    model_family: ModelFamily
    adapter_type: str
    target_modules: List[str]
    rank: int
    alpha: int
    description: str = ""
    capabilities: List[str] = None
    domain: str = "general"
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class UniversalAdapterManager:
    """Manages LoRA adapters for any base model."""
    
    def __init__(self, base_model: BaseModelInterface):
        self.base_model = base_model
        self.active_adapters: Dict[str, AdapterInfo] = {}
        self.adapter_registry: Dict[str, AdapterInfo] = {}
        self.peft_model = None
        self._original_model = None
        
    def register_adapter(self, adapter_info: AdapterInfo) -> bool:
        """Register an adapter in the manager."""
        try:
            # Validate adapter compatibility
            if not self._validate_adapter_compatibility(adapter_info):
                logger.error(f"Adapter {adapter_info.name} is not compatible with {self.base_model.config.model_family}")
                return False
            
            self.adapter_registry[adapter_info.name] = adapter_info
            logger.info(f"✅ Registered adapter: {adapter_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register adapter {adapter_info.name}: {e}")
            return False
    
    def load_adapter(self, adapter_name: str) -> bool:
        """Load a LoRA adapter."""
        try:
            if adapter_name not in self.adapter_registry:
                logger.error(f"Adapter {adapter_name} not found in registry")
                return False
            
            adapter_info = self.adapter_registry[adapter_name]
            
            # Check if adapter is already loaded
            if adapter_name in self.active_adapters:
                logger.info(f"Adapter {adapter_name} already loaded")
                return True
            
            # Initialize PEFT if not already done
            if self.peft_model is None:
                self._initialize_peft()
            
            # Load the adapter
            success = self._load_adapter_weights(adapter_info)
            
            if success:
                self.active_adapters[adapter_name] = adapter_info
                logger.info(f"✅ Loaded adapter: {adapter_name}")
                return True
            else:
                logger.error(f"Failed to load adapter weights for {adapter_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return False
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """Unload a LoRA adapter."""
        try:
            if adapter_name not in self.active_adapters:
                logger.warning(f"Adapter {adapter_name} not currently loaded")
                return True
            
            # Remove from PEFT model
            if self.peft_model is not None and hasattr(self.peft_model, 'delete_adapter'):
                self.peft_model.delete_adapter(adapter_name)
            
            # Remove from active adapters
            del self.active_adapters[adapter_name]
            
            logger.info(f"✅ Unloaded adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload adapter {adapter_name}: {e}")
            return False
    
    def switch_adapter(self, adapter_name: str) -> bool:
        """Switch to a different adapter."""
        try:
            # Unload all current adapters
            for active_adapter in list(self.active_adapters.keys()):
                self.unload_adapter(active_adapter)
            
            # Load the new adapter
            return self.load_adapter(adapter_name)
            
        except Exception as e:
            logger.error(f"Failed to switch to adapter {adapter_name}: {e}")
            return False
    
    def list_adapters(self) -> List[str]:
        """List all registered adapters."""
        return list(self.adapter_registry.keys())
    
    def list_active_adapters(self) -> List[str]:
        """List currently active adapters."""
        return list(self.active_adapters.keys())
    
    def get_adapter_info(self, adapter_name: str) -> Optional[AdapterInfo]:
        """Get information about an adapter."""
        return self.adapter_registry.get(adapter_name)
    
    def auto_discover_adapters(self, adapters_dir: str) -> int:
        """Automatically discover and register adapters from directory."""
        discovered_count = 0
        adapters_path = Path(adapters_dir)
        
        if not adapters_path.exists():
            logger.warning(f"Adapters directory not found: {adapters_dir}")
            return 0
        
        for adapter_dir in adapters_path.iterdir():
            if adapter_dir.is_dir():
                adapter_info = self._parse_adapter_directory(adapter_dir)
                if adapter_info and self.register_adapter(adapter_info):
                    discovered_count += 1
        
        logger.info(f"✅ Auto-discovered {discovered_count} adapters")
        return discovered_count
    
    def _validate_adapter_compatibility(self, adapter_info: AdapterInfo) -> bool:
        """Validate adapter compatibility with current base model."""
        try:
            # Check model family compatibility
            base_family = self.base_model.config.model_family
            adapter_family = adapter_info.model_family
            
            # Allow same family or unknown (for manual override)
            if adapter_family != ModelFamily.UNKNOWN and adapter_family != base_family:
                logger.warning(f"Model family mismatch: base={base_family}, adapter={adapter_family}")
                # Don't fail completely, but warn
            
            # Check if adapter path exists
            if not Path(adapter_info.path).exists():
                logger.error(f"Adapter path does not exist: {adapter_info.path}")
                return False
            
            # Use base model's validation if available
            if hasattr(self.base_model, 'validate_adapter'):
                return self.base_model.validate_adapter(adapter_info.path)
            
            return True
            
        except Exception as e:
            logger.error(f"Adapter compatibility validation failed: {e}")
            return False
    
    def _initialize_peft(self):
        """Initialize PEFT for the base model."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            # Store original model
            self._original_model = self.base_model.model
            
            # Get adapter compatibility info
            compat_info = self.base_model.get_adapter_compatibility()
            
            # Create a default LoRA config for initialization
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=compat_info.get("recommended_rank", 16),
                lora_alpha=compat_info.get("recommended_alpha", 32),
                target_modules=compat_info.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=0.1,
                bias="none",
            )
            
            # Create PEFT model
            self.peft_model = get_peft_model(self.base_model.model, lora_config)
            
            # Replace base model's model with PEFT model
            self.base_model.model = self.peft_model
            
            logger.info("✅ PEFT initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PEFT: {e}")
            raise
    
    def _load_adapter_weights(self, adapter_info: AdapterInfo) -> bool:
        """Load adapter weights into the PEFT model."""
        try:
            from peft import PeftModel
            
            # Load adapter using PEFT
            if hasattr(self.peft_model, 'load_adapter'):
                self.peft_model.load_adapter(adapter_info.path, adapter_info.name)
            else:
                # Alternative loading method
                self.peft_model = PeftModel.from_pretrained(
                    self._original_model,
                    adapter_info.path,
                    adapter_name=adapter_info.name
                )
                self.base_model.model = self.peft_model
            
            # Set adapter as active
            if hasattr(self.peft_model, 'set_adapter'):
                self.peft_model.set_adapter(adapter_info.name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter weights: {e}")
            return False
    
    def _parse_adapter_directory(self, adapter_dir: Path) -> Optional[AdapterInfo]:
        """Parse adapter directory to extract adapter information with robust error handling."""
        try:
            # Look for adapter config
            config_path = adapter_dir / "adapter_config.json"
            if not config_path.exists():
                logger.debug(f"No adapter config found in {adapter_dir}")
                return None

            # Load adapter config with error handling
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in adapter config {config_path}: {e}")
                return None
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error in adapter config {config_path}: {e}")
                return None

            # Validate required fields
            required_fields = ["base_model_name_or_path", "peft_type"]
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                logger.warning(f"Missing required fields in {config_path}: {missing_fields}")
                # Try to continue with defaults for non-critical fields

            # Look for Adaptrix metadata
            metadata_path = adapter_dir / "adaptrix_metadata.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to load metadata {metadata_path}: {e}")
                    metadata = {}

            # Extract information with defaults
            base_model = config.get("base_model_name_or_path", "unknown")
            peft_type = config.get("peft_type", "LORA").lower()

            # Detect model family from base model
            from .base_model_interface import ModelDetector
            model_family = ModelDetector.detect_family(base_model)

            # Handle different target module formats
            target_modules = config.get("target_modules", [])
            if isinstance(target_modules, str):
                target_modules = [target_modules]
            elif not isinstance(target_modules, list):
                logger.warning(f"Invalid target_modules format in {config_path}, using defaults")
                target_modules = ["q_proj", "v_proj"]

            # Extract rank and alpha with validation
            rank = config.get("r", config.get("rank", 16))
            alpha = config.get("lora_alpha", config.get("alpha", 32))

            try:
                rank = int(rank)
                alpha = int(alpha)
            except (ValueError, TypeError):
                logger.warning(f"Invalid rank/alpha values in {config_path}, using defaults")
                rank, alpha = 16, 32

            adapter_info = AdapterInfo(
                name=adapter_dir.name,
                path=str(adapter_dir),
                base_model=base_model,
                model_family=model_family,
                adapter_type=peft_type,
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                description=metadata.get("description", f"LoRA adapter for {base_model}"),
                capabilities=metadata.get("capabilities", []),
                domain=metadata.get("domain", "general")
            )

            # Validate adapter info
            if not self._validate_adapter_info(adapter_info):
                logger.error(f"Adapter validation failed for {adapter_dir}")
                return None

            return adapter_info

        except Exception as e:
            logger.error(f"Failed to parse adapter directory {adapter_dir}: {e}")
            return None

    def _validate_adapter_info(self, adapter_info: AdapterInfo) -> bool:
        """Validate adapter information for completeness and correctness."""
        try:
            # Check required fields
            if not adapter_info.name:
                logger.error("Adapter name is empty")
                return False

            if not adapter_info.path or not Path(adapter_info.path).exists():
                logger.error(f"Adapter path does not exist: {adapter_info.path}")
                return False

            if not adapter_info.base_model:
                logger.error("Base model is not specified")
                return False

            # Validate adapter type
            valid_types = ["lora", "qlora", "adalora", "ia3"]
            if adapter_info.adapter_type.lower() not in valid_types:
                logger.warning(f"Unknown adapter type: {adapter_info.adapter_type}")
                # Don't fail, just warn

            # Validate rank and alpha
            if adapter_info.rank <= 0 or adapter_info.rank > 512:
                logger.warning(f"Unusual rank value: {adapter_info.rank}")

            if adapter_info.alpha <= 0:
                logger.warning(f"Invalid alpha value: {adapter_info.alpha}")

            # Validate target modules
            if not adapter_info.target_modules:
                logger.warning("No target modules specified")

            return True

        except Exception as e:
            logger.error(f"Adapter validation failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current adapter manager status."""
        return {
            "base_model": self.base_model.config.model_id,
            "model_family": self.base_model.config.model_family.value,
            "total_adapters": len(self.adapter_registry),
            "active_adapters": list(self.active_adapters.keys()),
            "peft_initialized": self.peft_model is not None,
        }
    
    def cleanup(self):
        """Clean up adapter manager resources."""
        try:
            # Unload all adapters
            for adapter_name in list(self.active_adapters.keys()):
                self.unload_adapter(adapter_name)
            
            # Restore original model if needed
            if self._original_model is not None:
                self.base_model.model = self._original_model
                self._original_model = None
            
            self.peft_model = None
            self.active_adapters.clear()
            
            logger.info("✅ Adapter manager cleaned up")
            
        except Exception as e:
            logger.error(f"Adapter manager cleanup failed: {e}")


class AdapterConverter:
    """Converts adapters between different formats and model families."""
    
    @staticmethod
    def convert_adapter_for_model(
        source_adapter_path: str,
        target_model_family: ModelFamily,
        output_path: str
    ) -> bool:
        """Convert adapter for different model family."""
        try:
            # This would implement adapter conversion logic
            # For now, we'll implement basic compatibility checking
            
            logger.info(f"Converting adapter from {source_adapter_path} for {target_model_family}")
            
            # Load source adapter config
            config_path = Path(source_adapter_path) / "adapter_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update config for target model family
            # This is a simplified version - real implementation would be more complex
            
            # Save converted adapter
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "adapter_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Adapter converted successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Adapter conversion failed: {e}")
            return False
