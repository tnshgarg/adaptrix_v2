"""
Engine manager for the Adaptrix CLI.

This module provides functionality for creating and managing Adaptrix engine instances.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.utils.logging import get_logger

# Import modules with error handling
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

try:
    from src.moe.moe_engine import MoEAdaptrixEngine
    MOE_ENGINE_AVAILABLE = True
except ImportError:
    MoEAdaptrixEngine = None
    MOE_ENGINE_AVAILABLE = False

try:
    from src.composition.adapter_composer import AdapterComposer
    ADAPTER_COMPOSER_AVAILABLE = True
except ImportError:
    AdapterComposer = None
    ADAPTER_COMPOSER_AVAILABLE = False

try:
    from src.rag.vector_store import FAISSVectorStore
    RAG_AVAILABLE = True
except ImportError:
    FAISSVectorStore = None
    RAG_AVAILABLE = False

logger = get_logger("engine_manager")

class EngineManager:
    """
    Manages Adaptrix engine instances for the CLI.
    """
    
    def __init__(self, config_manager):
        """
        Initialize engine manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.models_dir = Path(self.config.get("models.directory"))
        self.adapters_dir = Path(self.config.get("adapters.directory"))
        self.rag_dir = Path(self.config.get("rag.directory"))
        
        # Cache for engine instances
        self._engine_cache: Dict[str, Any] = {}

        logger.info(f"EngineManager initialized (Optimized: {OPTIMIZED_ENGINE_AVAILABLE}, Modular: {MODULAR_ENGINE_AVAILABLE}, MoE: {MOE_ENGINE_AVAILABLE})")
    
    def create_engine(self, model_id: str, adapters: Optional[List[str]] = None,
                    rag_collection: Optional[str] = None, composition_strategy: str = "sequential") -> Optional[Any]:
        """
        Create a new engine instance.
        
        Args:
            model_id: Model identifier
            adapters: List of adapter names
            rag_collection: RAG collection name
            composition_strategy: Adapter composition strategy
        
        Returns:
            Engine instance or None if creation failed
        """
        try:
            # Generate cache key
            cache_key = f"{model_id}_{'-'.join(adapters or [])}_{rag_collection or 'no_rag'}_{composition_strategy}"
            
            # Check cache
            if cache_key in self._engine_cache:
                logger.info(f"Using cached engine for {cache_key}")
                return self._engine_cache[cache_key]
            
            # Normalize model path
            model_path = self._get_model_path(model_id)
            
            if not model_path.exists():
                logger.error(f"Model not found: {model_id}")
                return None
            
            # Create engine
            logger.info(f"Creating engine for model {model_id}")
            
            # Determine engine type - prefer OptimizedAdaptrixEngine for production
            if OPTIMIZED_ENGINE_AVAILABLE:
                # Use optimized engine for best performance
                rag_path = None
                if rag_collection:
                    rag_path = str(self.rag_dir / rag_collection)

                engine = OptimizedAdaptrixEngine(
                    model_id=str(model_path),
                    device=self.config.get("inference.device", "auto"),
                    adapters_dir=str(self.adapters_dir),
                    enable_auto_selection=True,
                    rag_vector_store_path=rag_path,
                    enable_rag=bool(rag_collection),
                    use_vllm=True,
                    enable_caching=True
                )
            elif len(adapters or []) > 1 and composition_strategy == "conditional" and MOE_ENGINE_AVAILABLE:
                # Use MoE engine for conditional composition
                engine = MoEAdaptrixEngine(
                    model_id=str(model_path),
                    device=self.config.get("inference.device", "auto"),
                    adapters_dir=str(self.adapters_dir)
                )
            elif MODULAR_ENGINE_AVAILABLE:
                # Use modular engine for other strategies
                engine = ModularAdaptrixEngine(
                    model_id=str(model_path),
                    device=self.config.get("inference.device", "auto"),
                    adapters_dir=str(self.adapters_dir)
                )
            else:
                logger.error("No engine implementation available")
                return None
            
            # Initialize engine
            if not engine.initialize():
                logger.error(f"Failed to initialize engine for {model_id}")
                return None
            
            # Add adapters
            if adapters:
                if composition_strategy == "conditional" and MOE_ENGINE_AVAILABLE:
                    # For MoE engine, add adapters to router
                    if hasattr(engine, 'add_adapter_to_router'):
                        for adapter_name in adapters:
                            engine.add_adapter_to_router(adapter_name)
                    else:
                        logger.warning("Engine does not support adapter routing")
                elif ADAPTER_COMPOSER_AVAILABLE:
                    # For modular engine, compose adapters
                    adapter_paths = [str(self.adapters_dir / adapter_name) for adapter_name in adapters]

                    # Create adapter composer (simplified for CLI)
                    # Note: This is a simplified implementation
                    # In a full implementation, we would use the actual AdapterComposer
                    logger.info(f"Loading adapters: {adapters} with strategy: {composition_strategy}")

                    # For now, just load adapters individually
                    for adapter_name in adapters:
                        if hasattr(engine, 'load_adapter'):
                            engine.load_adapter(adapter_name)
                        else:
                            logger.warning(f"Engine does not support adapter loading: {adapter_name}")
                else:
                    logger.warning("Adapter composition not available, skipping adapter loading")
            
            # Add RAG
            if rag_collection and RAG_AVAILABLE:
                rag_path = self.rag_dir / rag_collection

                if rag_path.exists():
                    try:
                        # Initialize vector store
                        vector_store = FAISSVectorStore.load(str(rag_path))

                        # Add RAG to engine if supported
                        if hasattr(engine, 'add_rag'):
                            engine.add_rag(vector_store)
                        else:
                            logger.warning("Engine does not support RAG integration")
                    except Exception as e:
                        logger.warning(f"Failed to load RAG collection {rag_collection}: {e}")
                else:
                    logger.warning(f"RAG collection not found: {rag_collection}")
            elif rag_collection:
                logger.warning("RAG functionality not available")
            
            # Cache engine
            self._engine_cache[cache_key] = engine
            
            return engine
            
        except Exception as e:
            logger.error(f"Error creating engine: {e}")
            return None
    
    def get_engine(self, cache_key: str) -> Optional[Any]:
        """
        Get a cached engine instance.

        Args:
            cache_key: Engine cache key

        Returns:
            Engine instance or None if not found
        """
        return self._engine_cache.get(cache_key)
    
    def cleanup(self):
        """Clean up all engine instances."""
        for engine in self._engine_cache.values():
            try:
                # Clean up engine resources if needed
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up engine: {e}")
        
        self._engine_cache.clear()
        logger.info("Engine cache cleaned up")
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get local path for a model."""
        # Check if it's a direct path
        if os.path.exists(model_id):
            return Path(model_id)
        
        # Check if it's a model ID
        model_path = self.models_dir / model_id.replace("/", "--")
        
        if model_path.exists():
            return model_path
        
        # Try HuggingFace format
        return self.models_dir / model_id
