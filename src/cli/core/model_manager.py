"""
Model manager for the Adaptrix CLI.

This module provides functionality for downloading, managing, and running models.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import engines with error handling
try:
    from src.inference.optimized_engine import OptimizedAdaptrixEngine
    OPTIMIZED_ENGINE_AVAILABLE = True
except ImportError:
    OptimizedAdaptrixEngine = None
    OPTIMIZED_ENGINE_AVAILABLE = False

try:
    from src.core.modular_engine import ModularAdaptrixEngine
    MODULAR_ENGINE_AVAILABLE = True
except ImportError:
    ModularAdaptrixEngine = None
    MODULAR_ENGINE_AVAILABLE = False

# Check if any engine is available
ENGINE_AVAILABLE = OPTIMIZED_ENGINE_AVAILABLE or MODULAR_ENGINE_AVAILABLE

if not ENGINE_AVAILABLE:
    logger = logging.getLogger("model_manager")
    logger.warning("No Adaptrix engines available, using mock implementation")

    # Mock implementation for when engines are not available
    class MockEngine:
        def __init__(self, **kwargs):
            pass

        def initialize(self):
            return False

        def generate(self, *args, **kwargs):
            return "Mock response: Engine not available"

    OptimizedAdaptrixEngine = MockEngine
    ModularAdaptrixEngine = MockEngine
from src.cli.utils.logging import get_logger
from src.cli.utils.progress import download_with_progress, ProgressBar

logger = get_logger("model_manager")

class ModelManager:
    """
    Manages model downloading, caching, and execution for the CLI.
    """
    
    def __init__(self, config_manager):
        """
        Initialize model manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.models_dir = Path(self.config.get("models.directory"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model registry
        self.registry = self._load_model_registry()
        
        # Cache for loaded engines
        self._engine_cache: Dict[str, ModularAdaptrixEngine] = {}
        
        logger.info(f"ModelManager initialized with directory: {self.models_dir}")
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load the model registry from configuration."""
        try:
            # Load from CLI config directory
            registry_path = Path(__file__).parent.parent / "config" / "models_registry.yaml"
            
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Model registry not found, using empty registry")
                return {"models": {}, "model_families": {}}
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            return {"models": {}, "model_families": {}}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models from the registry.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_id, model_info in self.registry.get("models", {}).items():
            model_data = {
                "name": model_id,
                "description": model_info.get("description", ""),
                "parameters": f"{model_info.get('parameters', 0)}B",
                "size_gb": f"{model_info.get('size_gb', 0)}GB",
                "architecture": model_info.get("architecture", "unknown"),
                "downloaded": self.is_model_downloaded(model_id),
                "license": model_info.get("license", "unknown")
            }
            models.append(model_data)
        
        return models
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded models.
        
        Returns:
            List of downloaded model information dictionaries
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_id = model_dir.name
                
                # Get model info from registry if available
                model_info = self.registry.get("models", {}).get(model_id, {})
                
                model_data = {
                    "name": model_id,
                    "description": model_info.get("description", ""),
                    "parameters": f"{model_info.get('parameters', 0)}B",
                    "size_gb": f"{model_info.get('size_gb', 0)}GB",
                    "architecture": model_info.get("architecture", "unknown"),
                    "downloaded": True,
                    "path": str(model_dir)
                }
                models.append(model_data)
        
        return models
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """
        Check if a model is downloaded.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if model is downloaded, False otherwise
        """
        model_path = self.models_dir / model_id.replace("/", "--")
        return model_path.exists() and model_path.is_dir()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model information dictionary or None if not found
        """
        # Check registry first
        if model_id in self.registry.get("models", {}):
            model_info = self.registry["models"][model_id].copy()
            model_info["downloaded"] = self.is_model_downloaded(model_id)
            
            if model_info["downloaded"]:
                model_path = self.models_dir / model_id.replace("/", "--")
                model_info["local_path"] = str(model_path)
            
            return model_info
        
        # If not in registry, check if it's downloaded
        if self.is_model_downloaded(model_id):
            return {
                "name": model_id,
                "description": "Custom model",
                "downloaded": True,
                "local_path": str(self.models_dir / model_id.replace("/", "--"))
            }
        
        return None
    
    def download_model(self, model_id: str) -> bool:
        """
        Download a model from HuggingFace Hub.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create model directory
            model_dir = self.models_dir / model_id.replace("/", "--")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading model {model_id} to {model_dir}")
            
            # Get list of files to download
            api = HfApi()
            repo_files = api.list_repo_files(model_id)
            
            # Filter for essential files
            essential_files = [
                f for f in repo_files 
                if f.endswith(('.bin', '.safetensors', '.json', '.txt', '.md'))
                and not f.startswith('.')
            ]
            
            # Download files with progress
            with ProgressBar(f"Downloading {model_id}", len(essential_files)) as progress:
                for file_path in essential_files:
                    try:
                        # Download file
                        downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=file_path,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False
                        )
                        
                        progress.update(1, f"Downloaded {file_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to download {file_path}: {e}")
                        continue
            
            # Save model metadata
            model_info = self.registry.get("models", {}).get(model_id, {})
            metadata = {
                "model_id": model_id,
                "download_date": str(Path().cwd()),
                "local_path": str(model_dir),
                **model_info
            }
            
            metadata_path = model_dir / "adaptrix_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully downloaded model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            
            # Clean up on failure
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            
            return False
    
    def run_model(self, model_id: str, prompt: str, **kwargs) -> str:
        """
        Run inference with a model.
        
        Args:
            model_id: Model identifier
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response
        """
        try:
            # Get or create engine
            if model_id not in self._engine_cache:
                # Get model path
                model_path = self.models_dir / model_id.replace("/", "--")

                if not model_path.exists():
                    raise ValueError(f"Model {model_id} not found. Please download it first.")

                # Initialize engine - prefer OptimizedAdaptrixEngine
                if OPTIMIZED_ENGINE_AVAILABLE:
                    engine = OptimizedAdaptrixEngine(
                        model_id=str(model_path),
                        device=self.config.get("inference.device", "auto"),
                        adapters_dir=self.config.get("adapters.directory"),
                        enable_auto_selection=True,
                        use_vllm=True,
                        enable_caching=True
                    )
                elif MODULAR_ENGINE_AVAILABLE:
                    engine = ModularAdaptrixEngine(
                        model_id=str(model_path),
                        device=self.config.get("inference.device", "auto"),
                        adapters_dir=self.config.get("adapters.directory")
                    )
                else:
                    raise RuntimeError("No Adaptrix engine available")

                # Initialize engine
                if not engine.initialize():
                    raise RuntimeError(f"Failed to initialize engine for model {model_id}")

                self._engine_cache[model_id] = engine
            
            # Get engine
            engine = self._engine_cache[model_id]
            
            # Set generation parameters
            generation_config = {
                "max_tokens": kwargs.get("max_tokens", self.config.get("inference.max_tokens", 1024)),
                "temperature": kwargs.get("temperature", self.config.get("inference.temperature", 0.7)),
                "top_p": kwargs.get("top_p", self.config.get("inference.top_p", 0.9)),
                "top_k": kwargs.get("top_k", self.config.get("inference.top_k", 50)),
            }
            
            # Generate response
            response = engine.generate(prompt, **generation_config)
            
            return response
            
        except Exception as e:
            logger.error(f"Error running model {model_id}: {e}")
            raise
    
    def cleanup_cache(self):
        """Clean up cached engines."""
        for engine in self._engine_cache.values():
            try:
                # Clean up engine resources if needed
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up engine: {e}")
        
        self._engine_cache.clear()
        logger.info("Model cache cleaned up")
