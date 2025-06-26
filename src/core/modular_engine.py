"""
Modular Adaptrix Engine - Universal support for any base model and adapters.

This engine provides a plug-and-play architecture that can work with any
base LLM model and corresponding LoRA adapters.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_model_interface import (
    BaseModelInterface, ModelFactory, GenerationConfig, 
    get_optimal_generation_config, ModelFamily
)
from .universal_adapter_manager import UniversalAdapterManager, AdapterInfo
from .prompt_templates import PromptTemplateManager, ResponseFormatter

logger = logging.getLogger(__name__)


class ModularAdaptrixEngine:
    """
    Modular Adaptrix Engine with universal base model support.
    
    Features:
    - Plug-and-play base model support (Qwen, Phi, LLaMA, Mistral, etc.)
    - Universal LoRA adapter management
    - Automatic model family detection
    - Optimized generation parameters per model family
    - Domain-specific prompt engineering
    """
    
    def __init__(self, model_id: str, device: str = "cpu", adapters_dir: str = "adapters"):
        """
        Initialize Modular Adaptrix Engine.
        
        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-1.7B")
            device: Device to run on ('cpu', 'cuda', 'auto')
            adapters_dir: Directory containing LoRA adapters
        """
        self.model_id = model_id
        self.device = device
        self.adapters_dir = adapters_dir
        
        # Core components
        self.base_model: Optional[BaseModelInterface] = None
        self.adapter_manager: Optional[UniversalAdapterManager] = None
        
        # State tracking
        self._initialized = False
        self._lock = threading.Lock()
        
        # Conversation state
        self.conversation_history: List[str] = []
        self.use_context_by_default = False
        
        logger.info(f"Initialized Modular Adaptrix Engine for {model_id}")
    
    def initialize(self) -> bool:
        """Initialize the engine with the specified base model."""
        try:
            logger.info(f"🚀 Initializing Modular Adaptrix Engine...")
            logger.info(f"   Model: {self.model_id}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Adapters: {self.adapters_dir}")
            
            # Create base model instance
            self.base_model = ModelFactory.create_model(self.model_id, self.device)
            
            if not self.base_model:
                logger.error(f"Failed to create model instance for {self.model_id}")
                return False
            
            # Initialize base model
            if not self.base_model.initialize():
                logger.error(f"Failed to initialize base model")
                return False
            
            # Initialize adapter manager
            self.adapter_manager = UniversalAdapterManager(self.base_model)
            
            # Auto-discover adapters
            if Path(self.adapters_dir).exists():
                discovered = self.adapter_manager.auto_discover_adapters(self.adapters_dir)
                logger.info(f"✅ Discovered {discovered} adapters")
            
            self._initialized = True
            
            # Log system info
            model_info = self.base_model.get_model_info()
            logger.info(f"✅ Engine initialized successfully!")
            logger.info(f"   Model Family: {model_info['model_family']}")
            logger.info(f"   Context Length: {model_info['context_length']}")
            logger.info(f"   Parameters: {model_info.get('total_parameters', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = None,
        task_type: str = "general",
        use_context: bool = None,
        **kwargs
    ) -> str:
        """
        Generate text using the current model and adapter.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            task_type: Type of task (general, code, math, creative)
            use_context: Whether to use conversation context
            **kwargs: Additional generation parameters
        """
        if not self._initialized:
            return "Error: Engine not initialized"
        
        try:
            with self._lock:
                # Apply domain-specific prompt engineering
                enhanced_prompt = self._apply_prompt_engineering(prompt)
                
                # Apply conversation context if requested
                if use_context or (use_context is None and self.use_context_by_default):
                    enhanced_prompt = self._apply_conversation_context(enhanced_prompt)
                
                # Get optimal generation config
                generation_config = get_optimal_generation_config(
                    self.base_model.get_model_family(), 
                    task_type
                )
                
                # Override with user parameters
                if max_length:
                    generation_config.max_new_tokens = max_length
                
                for key, value in kwargs.items():
                    if hasattr(generation_config, key):
                        setattr(generation_config, key, value)
                
                # Generate response
                start_time = time.time()
                raw_response = self.base_model.generate(enhanced_prompt, generation_config)
                generation_time = time.time() - start_time
                
                # Post-process response
                final_response = self._post_process_response(raw_response, task_type)
                
                # Update conversation history
                self.conversation_history.extend([prompt, final_response])
                
                # Keep only recent history
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                logger.debug(f"Generated response in {generation_time:.2f}s")
                return final_response
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: Generation failed - {str(e)}"
    
    def load_adapter(self, adapter_name: str) -> bool:
        """Load a LoRA adapter."""
        if not self._initialized or not self.adapter_manager:
            logger.error("Engine not initialized")
            return False
        
        return self.adapter_manager.load_adapter(adapter_name)
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """Unload a LoRA adapter."""
        if not self._initialized or not self.adapter_manager:
            logger.error("Engine not initialized")
            return False
        
        return self.adapter_manager.unload_adapter(adapter_name)
    
    def switch_adapter(self, adapter_name: str) -> bool:
        """Switch to a different adapter."""
        if not self._initialized or not self.adapter_manager:
            logger.error("Engine not initialized")
            return False
        
        return self.adapter_manager.switch_adapter(adapter_name)
    
    def list_adapters(self) -> List[str]:
        """List all available adapters."""
        if not self._initialized or not self.adapter_manager:
            return []
        
        return self.adapter_manager.list_adapters()
    
    def list_active_adapters(self) -> List[str]:
        """List currently active adapters."""
        if not self._initialized or not self.adapter_manager:
            return []
        
        return self.adapter_manager.list_active_adapters()
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an adapter."""
        if not self._initialized or not self.adapter_manager:
            return None
        
        adapter_info = self.adapter_manager.get_adapter_info(adapter_name)
        if adapter_info:
            return {
                "name": adapter_info.name,
                "base_model": adapter_info.base_model,
                "model_family": adapter_info.model_family.value,
                "adapter_type": adapter_info.adapter_type,
                "domain": adapter_info.domain,
                "capabilities": adapter_info.capabilities,
                "description": adapter_info.description,
                "rank": adapter_info.rank,
                "alpha": adapter_info.alpha,
            }
        return None
    
    def register_adapter(
        self,
        name: str,
        path: str,
        description: str = "",
        capabilities: List[str] = None,
        domain: str = "general"
    ) -> bool:
        """Register a new adapter."""
        if not self._initialized or not self.adapter_manager:
            logger.error("Engine not initialized")
            return False
        
        try:
            # Auto-detect adapter info
            adapter_info = self.adapter_manager._parse_adapter_directory(Path(path))
            
            if adapter_info:
                # Override with provided info
                adapter_info.name = name
                adapter_info.description = description or adapter_info.description
                adapter_info.capabilities = capabilities or adapter_info.capabilities
                adapter_info.domain = domain
                
                return self.adapter_manager.register_adapter(adapter_info)
            else:
                logger.error(f"Could not parse adapter at {path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register adapter: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self._initialized,
            "model_id": self.model_id,
            "device": self.device,
        }
        
        if self._initialized and self.base_model:
            status.update({
                "model_info": self.base_model.get_model_info(),
                "adapter_status": self.adapter_manager.get_status() if self.adapter_manager else {},
                "memory_usage": self.base_model.get_memory_usage() if hasattr(self.base_model, 'get_memory_usage') else {},
            })
        
        return status
    
    def _apply_prompt_engineering(self, prompt: str) -> str:
        """Apply domain-specific prompt engineering."""
        try:
            # Get current adapter info for domain detection
            current_adapter = None
            if self.adapter_manager and self.adapter_manager.list_active_adapters():
                adapter_name = self.adapter_manager.list_active_adapters()[0]
                adapter_info = self.adapter_manager.get_adapter_info(adapter_name)
                if adapter_info:
                    current_adapter = adapter_info.name
            
            # Use prompt template manager
            structured_prompt = PromptTemplateManager.get_structured_prompt(
                task=prompt,
                adapter_name=current_adapter
            )
            
            return structured_prompt
            
        except Exception as e:
            logger.warning(f"Prompt engineering failed: {e}")
            return prompt
    
    def _apply_conversation_context(self, prompt: str) -> str:
        """Apply conversation context to prompt."""
        if not self.conversation_history:
            return prompt
        
        # Get recent context (last 3 exchanges)
        recent_history = self.conversation_history[-6:]  # 3 exchanges = 6 entries
        
        context_parts = []
        for i in range(0, len(recent_history), 2):
            if i + 1 < len(recent_history):
                user_msg = recent_history[i]
                assistant_msg = recent_history[i + 1]
                context_parts.append(f"User: {user_msg}")
                context_parts.append(f"Assistant: {assistant_msg}")
        
        context = "\n".join(context_parts)
        return f"{context}\nUser: {prompt}\nAssistant:"
    
    def _post_process_response(self, response: str, task_type: str) -> str:
        """Post-process generated response."""
        try:
            # Determine domain from task type or active adapter
            domain = "general"
            if task_type in ["code", "programming"]:
                domain = "programming"
            elif task_type in ["math", "mathematics"]:
                domain = "mathematics"
            elif task_type in ["news", "journalism"]:
                domain = "journalism"
            
            # Use response formatter
            formatted_response = ResponseFormatter.format_response(response, domain)
            
            return formatted_response
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return response.strip()
    
    def cleanup(self):
        """Clean up engine resources."""
        try:
            if self.adapter_manager:
                self.adapter_manager.cleanup()
            
            if self.base_model:
                self.base_model.cleanup()
            
            self.conversation_history.clear()
            self._initialized = False
            
            logger.info("✅ Modular engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Engine cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
