"""
Main Adaptrix engine that coordinates all components.
"""

import logging
import time
import torch
from typing import Dict, List, Optional, Any
from ..models.base_model import BaseModelManager
from ..injection.layer_injector import LayerInjector
from ..adapters.adapter_manager import AdapterManager
from ..core.dynamic_loader import DynamicLoader
from ..utils.config import config
from ..utils.helpers import timer, get_memory_info

logger = logging.getLogger(__name__)


class AdaptrixEngine:
    """
    Main engine for the Adaptrix system.
    
    Coordinates all components and provides a unified interface
    for dynamic LoRA adapter injection and inference.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        """
        Initialize Adaptrix engine.
        
        Args:
            model_name: Name of the base model to use
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize components
        self.base_model_manager = BaseModelManager(model_name, device)
        self.adapter_manager = AdapterManager()
        
        # These will be initialized after model loading
        self.layer_injector: Optional[LayerInjector] = None
        self.dynamic_loader: Optional[DynamicLoader] = None
        
        # State tracking
        self._initialized = False
        self._model_loaded = False
        
        logger.info("AdaptrixEngine initialized")

    @property
    def tokenizer(self):
        """Get the tokenizer from the base model manager."""
        return self.base_model_manager.tokenizer if self.base_model_manager else None

    def initialize(self) -> bool:
        """
        Initialize the engine by loading the base model and setting up injection.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            logger.info("Engine already initialized")
            return True
        
        try:
            with timer("Engine initialization"):
                # Load base model
                logger.info("Loading base model...")
                model = self.base_model_manager.load_model()
                self._model_loaded = True
                
                # Initialize layer injector
                logger.info("Initializing layer injector...")
                self.layer_injector = LayerInjector(model)
                
                # Initialize dynamic loader
                logger.info("Initializing dynamic loader...")
                self.dynamic_loader = DynamicLoader(self.layer_injector, self.adapter_manager)
                
                # Register injection points for target layers
                target_layers = config.get('injection.target_layers', [6, 12, 18])
                target_modules = config.get('injection.target_modules', ['self_attn.q_proj', 'mlp.c_fc'])
                
                for layer_idx in target_layers:
                    for module_name in target_modules:
                        self.layer_injector.register_injection_point(layer_idx, module_name)
                
                self._initialized = True
                logger.info("Engine initialization completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            return False
    
    def load_adapter(self, adapter_name: str, layer_indices: Optional[List[int]] = None) -> bool:
        """
        Load an adapter into the model.
        
        Args:
            adapter_name: Name of the adapter to load
            layer_indices: Specific layers to inject into (None for default)
            
        Returns:
            True if loading successful
        """
        if not self._check_initialized():
            return False
        
        return self.dynamic_loader.load_adapter(adapter_name, layer_indices)
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """
        Unload an adapter from the model.
        
        Args:
            adapter_name: Name of the adapter to unload
            
        Returns:
            True if unloading successful
        """
        if not self._check_initialized():
            return False
        
        return self.dynamic_loader.unload_adapter(adapter_name)
    
    def switch_adapter(self, old_name: str, new_name: str) -> bool:
        """
        Switch from one adapter to another.
        
        Args:
            old_name: Name of adapter to unload
            new_name: Name of adapter to load
            
        Returns:
            True if switch successful
        """
        if not self._check_initialized():
            return False
        
        return self.dynamic_loader.switch_adapter(old_name, new_name)
    
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """
        Generate text using the current model configuration.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self._check_initialized():
            return ""
        
        try:
            with timer("Text generation"):
                # Tokenize input
                tokenizer = self.base_model_manager.tokenizer
                model = self.base_model_manager.model
                
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                inputs = inputs.to(self.base_model_manager.device)
                
                # Set generation parameters
                generation_kwargs = {
                    'max_length': max_length,
                    'do_sample': kwargs.get('do_sample', True),
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'pad_token_id': tokenizer.eos_token_id,
                    **kwargs
                }
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(inputs, **generation_kwargs)
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove input prompt from output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return generated_text
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return ""
    
    def query(self, text: str, adapter_name: Optional[str] = None, **kwargs) -> str:
        """
        Query the model with optional adapter specification.
        
        Args:
            text: Input text/prompt
            adapter_name: Specific adapter to use (None for current)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        if not self._check_initialized():
            return ""
        
        # Switch to specific adapter if requested
        current_adapters = self.get_loaded_adapters()
        adapter_switched = False
        
        try:
            if adapter_name and adapter_name not in current_adapters:
                # Load the requested adapter
                if self.load_adapter(adapter_name):
                    adapter_switched = True
                else:
                    logger.warning(f"Failed to load adapter {adapter_name}, using current configuration")
            
            # Generate response
            response = self.generate(text, **kwargs)
            
            return response
            
        finally:
            # Optionally unload the adapter if it was temporarily loaded
            if adapter_switched and config.get('adapters.auto_cleanup', True):
                self.unload_adapter(adapter_name)
    
    def list_adapters(self) -> List[str]:
        """
        List all available adapters.
        
        Returns:
            List of adapter names
        """
        return self.adapter_manager.list_adapters()
    
    def get_loaded_adapters(self) -> List[str]:
        """
        Get currently loaded adapters.
        
        Returns:
            List of loaded adapter names
        """
        if not self._check_initialized():
            return []
        
        return list(self.dynamic_loader.get_loaded_adapters().keys())
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Adapter information dictionary
        """
        return self.adapter_manager.get_adapter_info(adapter_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status dictionary
        """
        status = {
            'initialized': self._initialized,
            'model_loaded': self._model_loaded,
            'model_name': self.model_name,
            'device': str(self.base_model_manager.device) if self._model_loaded else None,
        }
        
        if self._model_loaded:
            status['model_info'] = self.base_model_manager.get_model_info()
        
        if self._initialized:
            status['loaded_adapters'] = self.get_loaded_adapters()
            status['available_adapters'] = self.list_adapters()
            status['memory_usage'] = self.dynamic_loader.get_memory_usage()
            status['active_injection_points'] = self.layer_injector.get_active_adapters()
        
        # Add system memory info
        status['system_memory'] = get_memory_info()
        
        return status
    
    def cleanup(self) -> None:
        """Clean up resources and unload everything."""
        try:
            if self.dynamic_loader:
                self.dynamic_loader.clear_all()
            
            if self.layer_injector:
                self.layer_injector.clear_all_adapters()
            
            if self.base_model_manager:
                self.base_model_manager.unload_model()
            
            self._initialized = False
            self._model_loaded = False
            
            logger.info("Engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _check_initialized(self) -> bool:
        """Check if engine is properly initialized."""
        if not self._initialized:
            logger.error("Engine not initialized. Call initialize() first.")
            return False
        return True
    
    def benchmark_adapter(self, adapter_name: str, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Benchmark an adapter's performance.
        
        Args:
            adapter_name: Name of the adapter to benchmark
            test_prompts: List of test prompts
            
        Returns:
            Benchmark results
        """
        if not self._check_initialized():
            return {}
        
        results = {
            'adapter_name': adapter_name,
            'test_prompts': len(test_prompts),
            'results': []
        }
        
        try:
            # Load adapter
            load_start = time.time()
            if not self.load_adapter(adapter_name):
                return {'error': f'Failed to load adapter {adapter_name}'}
            load_time = time.time() - load_start
            
            # Run tests
            for i, prompt in enumerate(test_prompts):
                start_time = time.time()
                response = self.generate(prompt, max_length=50)
                end_time = time.time()
                
                results['results'].append({
                    'prompt_index': i,
                    'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                    'response': response[:100] + '...' if len(response) > 100 else response,
                    'generation_time': end_time - start_time,
                    'response_length': len(response)
                })
            
            # Calculate statistics
            generation_times = [r['generation_time'] for r in results['results']]
            results['statistics'] = {
                'load_time': load_time,
                'avg_generation_time': sum(generation_times) / len(generation_times),
                'min_generation_time': min(generation_times),
                'max_generation_time': max(generation_times),
                'total_time': sum(generation_times)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            return {'error': str(e)}
        
        finally:
            # Clean up
            self.unload_adapter(adapter_name)
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
