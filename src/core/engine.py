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
from ..composition.adapter_composer import AdapterComposer, CompositionStrategy
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
        self.adapter_composer: Optional[AdapterComposer] = None
        
        # State tracking
        self._initialized = False
        self._model_loaded = False

        # Conversation memory
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        self.use_conversation_context = True

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

                # Initialize adapter composer
                logger.info("Initializing adapter composer...")
                self.adapter_composer = AdapterComposer(self.layer_injector, self.adapter_manager)
                
                # Detect model architecture and set appropriate modules
                target_layers, target_modules = self._detect_model_architecture(model)

                # Set the detected target modules in the layer injector
                self.layer_injector.set_target_modules(target_modules)

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
    
    def generate(self, prompt: str, max_length: int = 100, use_context: bool = None, **kwargs) -> str:
        """
        Generate text using the current model configuration.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length (total tokens, not new tokens)
            use_context: Whether to use conversation context (None for default)
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._check_initialized():
            return ""

        try:
            with timer("Text generation"):
                # Determine if we should use context
                if use_context is None:
                    use_context = self.use_conversation_context

                # Build context-aware prompt
                if use_context and self.conversation_history:
                    context_prompt = self._build_context_prompt(prompt)
                else:
                    context_prompt = prompt

                # Tokenize input with proper attention mask
                tokenizer = self.base_model_manager.tokenizer
                model = self.base_model_manager.model

                # Proper tokenization with attention mask
                inputs = tokenizer(context_prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.base_model_manager.device) for k, v in inputs.items()}

                # Calculate max_new_tokens from max_length
                input_length = inputs['input_ids'].shape[1]
                max_new_tokens = max(1, max_length - input_length)

                # Set optimized generation parameters for Phi-2 compatibility
                generation_kwargs = {
                    'max_new_tokens': max_new_tokens,  # Generate new tokens, not total length
                    'do_sample': kwargs.get('do_sample', True),
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }

                # Add compatible generation parameters for Phi-2
                if kwargs.get('do_sample', True):
                    # Only add sampling parameters if do_sample is True
                    if 'temperature' in kwargs:
                        generation_kwargs['temperature'] = kwargs['temperature']
                    if 'top_p' in kwargs:
                        generation_kwargs['top_p'] = kwargs['top_p']
                    if 'top_k' in kwargs:
                        generation_kwargs['top_k'] = kwargs['top_k']

                # Add other compatible parameters
                if 'repetition_penalty' in kwargs:
                    generation_kwargs['repetition_penalty'] = kwargs['repetition_penalty']

                # Generate with attention mask
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        **generation_kwargs
                    )

                # Decode only the new tokens (exclude input)
                generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

                # Post-process for better quality
                response = self._post_process_response(generated_text.strip())

                # Add to conversation history
                if use_context:
                    self._add_to_history(prompt, response)

                return response

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

            # Add composition system status
            if self.adapter_composer:
                status['composition_stats'] = self.adapter_composer.get_composition_stats()

        # Add system memory info
        status['system_memory'] = get_memory_info()

        return status

    # Revolutionary Multi-Adapter Composition Methods

    def compose_adapters(self,
                        adapters: List[str],
                        strategy: Optional[CompositionStrategy] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        🚀 REVOLUTIONARY FEATURE: Compose multiple adapters for enhanced intelligence.

        This is the core innovation that sets Adaptrix apart - the ability to combine
        multiple specialized adapters in sophisticated ways to create emergent capabilities.

        Args:
            adapters: List of adapter names to compose
            strategy: Composition strategy (auto-selected if None)
            **kwargs: Additional composition parameters

        Returns:
            Dictionary with composition results and metadata
        """
        if not self._check_initialized():
            return {'error': 'Engine not initialized'}

        if not self.adapter_composer:
            return {'error': 'Adapter composer not available'}

        try:
            # Load all requested adapters
            loaded_adapters = []
            for adapter_name in adapters:
                if self.load_adapter(adapter_name):
                    loaded_adapters.append(adapter_name)
                else:
                    logger.warning(f"Failed to load adapter {adapter_name}")

            if not loaded_adapters:
                return {'error': 'No adapters could be loaded'}

            # Perform composition
            result = self.adapter_composer.compose_adapters(loaded_adapters, strategy, kwargs)

            return {
                'success': True,
                'strategy': result.strategy_used.value,
                'adapters_used': loaded_adapters,
                'confidence': result.confidence_scores,
                'weights': result.composition_weights,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            }

        except Exception as e:
            logger.error(f"Failed to compose adapters: {e}")
            return {'error': str(e)}

    def generate_with_composition(self,
                                 prompt: str,
                                 adapters: List[str],
                                 strategy: Optional[CompositionStrategy] = None,
                                 max_length: int = 100,
                                 temperature: float = 0.7,
                                 **kwargs) -> str:
        """
        🚀 Generate text using multi-adapter composition for enhanced capabilities.

        This method showcases the revolutionary power of Adaptrix by combining
        multiple specialized adapters during text generation.

        Args:
            prompt: Input prompt
            adapters: List of adapters to compose
            strategy: Composition strategy
            max_length: Maximum generation length
            temperature: Generation temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text with enhanced capabilities
        """
        if not self._check_initialized():
            return "Error: Engine not initialized"

        try:
            # Set up composition
            composition_result = self.compose_adapters(adapters, strategy, **kwargs)

            if not composition_result.get('success'):
                logger.warning(f"Composition failed: {composition_result.get('error')}")
                # Fallback to single adapter or base model
                if adapters:
                    self.load_adapter(adapters[0])
                return self.generate(prompt, max_length, temperature)

            # Generate with composed adapters
            logger.info(f"Generating with {len(adapters)} composed adapters using {composition_result['strategy']} strategy")

            # The actual generation happens with all adapters active
            # The composition system has already set up the optimal combination
            generated_text = self.generate(prompt, max_length, temperature)

            # Add composition metadata to the response
            composition_info = f"\n\n[Composed using {composition_result['strategy']} strategy with {len(adapters)} adapters]"

            return generated_text + composition_info

        except Exception as e:
            logger.error(f"Failed to generate with composition: {e}")
            return f"Error: {str(e)}"

    def get_composition_recommendations(self,
                                      available_adapters: Optional[List[str]] = None,
                                      task_context: Optional[str] = None) -> Dict[str, Any]:
        """
        🚀 Get intelligent recommendations for adapter composition.

        Args:
            available_adapters: List of available adapters (None for all)
            task_context: Context about the task to optimize for

        Returns:
            Dictionary with composition recommendations
        """
        if not self._check_initialized() or not self.adapter_composer:
            return {'error': 'System not ready'}

        try:
            if available_adapters is None:
                available_adapters = self.list_adapters()

            recommendations = {}

            # Get recommendations for different numbers of adapters
            for num_adapters in [2, 3, 4, 5]:
                if len(available_adapters) >= num_adapters:
                    # Select top adapters (could be enhanced with more sophisticated selection)
                    selected_adapters = available_adapters[:num_adapters]

                    # Get strategy recommendation
                    recommended_strategy = self.adapter_composer.recommend_composition_strategy(selected_adapters)

                    recommendations[f"{num_adapters}_adapters"] = {
                        'adapters': selected_adapters,
                        'strategy': recommended_strategy.value,
                        'expected_benefits': self._describe_composition_benefits(selected_adapters, recommended_strategy)
                    }

            return {
                'success': True,
                'recommendations': recommendations,
                'total_available_adapters': len(available_adapters),
                'composition_stats': self.adapter_composer.get_composition_stats()
            }

        except Exception as e:
            logger.error(f"Failed to get composition recommendations: {e}")
            return {'error': str(e)}

    def _describe_composition_benefits(self,
                                     adapters: List[str],
                                     strategy: CompositionStrategy) -> List[str]:
        """Describe the expected benefits of a composition."""
        benefits = []

        if len(adapters) >= 2:
            benefits.append("Enhanced reasoning through multi-adapter collaboration")

        if strategy == CompositionStrategy.PARALLEL:
            benefits.append("Simultaneous processing for comprehensive analysis")
        elif strategy == CompositionStrategy.SEQUENTIAL:
            benefits.append("Step-by-step refinement through adapter pipeline")
        elif strategy == CompositionStrategy.HIERARCHICAL:
            benefits.append("Structured processing with specialized stages")
        elif strategy == CompositionStrategy.ATTENTION:
            benefits.append("Dynamic weighting based on context relevance")

        # Add adapter-specific benefits
        for adapter_name in adapters:
            adapter_data = self.adapter_manager.load_adapter(adapter_name)
            if adapter_data and 'metadata' in adapter_data:
                description = adapter_data['metadata'].get('description', '')
                if 'math' in description.lower():
                    benefits.append("Mathematical reasoning enhancement")
                elif 'code' in description.lower():
                    benefits.append("Programming and logic improvement")

        return benefits
    
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
    
    def _detect_model_architecture(self, model) -> tuple[List[int], List[str]]:
        """
        Detect model architecture and return appropriate target layers and modules.

        Args:
            model: The loaded model

        Returns:
            Tuple of (target_layers, target_modules)
        """
        try:
            # Get model config
            config_obj = getattr(model, 'config', None)
            if config_obj is None:
                logger.warning("No model config found, using default GPT-2 settings")
                return [6, 12, 18], ['attn.c_attn', 'mlp.c_fc']

            model_type = getattr(config_obj, 'model_type', 'unknown')
            num_layers = getattr(config_obj, 'num_hidden_layers', getattr(config_obj, 'n_layer', 24))

            logger.info(f"Detected model type: {model_type}, layers: {num_layers}")

            # Calculate target layers based on model size
            if num_layers <= 12:
                target_layers = [3, 6, 9]
            elif num_layers <= 24:
                target_layers = [6, 12, 18]
            else:
                # For larger models, spread across more layers
                step = num_layers // 4
                target_layers = [step, step * 2, step * 3]

            # Determine target modules based on architecture
            if model_type in ['qwen2', 'qwen']:
                # Qwen/DeepSeek architecture - USE ALL MODULES FOR MAXIMUM ADAPTER POWER
                target_modules = [
                    'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.o_proj',
                    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
                ]
                logger.info("Using Qwen2/DeepSeek module names - ALL 7 MODULES FOR MAXIMUM POWER")
            elif model_type in ['llama', 'mistral']:
                # LLaMA/Mistral architecture - USE ALL MODULES FOR MAXIMUM ADAPTER POWER
                target_modules = [
                    'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.o_proj',
                    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
                ]
                logger.info("Using LLaMA/Mistral module names - ALL 7 MODULES FOR MAXIMUM POWER")
            elif model_type in ['gpt2', 'gpt_neox']:
                # GPT-2 style architecture
                target_modules = ['attn.c_attn', 'mlp.c_fc']
                logger.info("Using GPT-2 module names")
            elif model_type == 'phi':
                # Phi-2 architecture - CORRECT MODULE NAMES
                target_modules = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.dense', 'mlp.fc1', 'mlp.fc2']
                logger.info("Using Phi-2 module names - CORRECTED")
            else:
                # Try to detect by examining the model structure
                logger.info(f"Unknown model type {model_type}, attempting auto-detection")
                target_modules = self._auto_detect_modules(model)

            logger.info(f"Selected target layers: {target_layers}")
            logger.info(f"Selected target modules: {target_modules}")

            return target_layers, target_modules

        except Exception as e:
            logger.error(f"Error detecting model architecture: {e}")
            # Fallback to default
            return [6, 12, 18], ['attn.c_attn', 'mlp.c_fc']

    def _auto_detect_modules(self, model) -> List[str]:
        """
        Auto-detect module names by examining the model structure.

        Args:
            model: The loaded model

        Returns:
            List of target module names
        """
        try:
            # Try to find the first transformer layer
            first_layer = None

            # Common patterns for finding layers
            if hasattr(model, 'layers') and len(model.layers) > 0:
                first_layer = model.layers[0]
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                first_layer = model.transformer.h[0]
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                first_layer = model.model.layers[0]

            if first_layer is None:
                logger.warning("Could not find transformer layers, using default modules")
                return ['attn.c_attn', 'mlp.c_fc']

            # Examine the layer structure
            modules = []
            for name, _ in first_layer.named_children():
                if 'attn' in name.lower():
                    # Check attention submodules
                    attn_module = getattr(first_layer, name)
                    for sub_name, _ in attn_module.named_children():
                        if 'q_proj' in sub_name:
                            modules.append(f"{name}.q_proj")
                        elif 'v_proj' in sub_name:
                            modules.append(f"{name}.v_proj")
                        elif 'c_attn' in sub_name:
                            modules.append(f"{name}.c_attn")
                elif 'mlp' in name.lower():
                    # Check MLP submodules
                    mlp_module = getattr(first_layer, name)
                    for sub_name, _ in mlp_module.named_children():
                        if 'gate_proj' in sub_name:
                            modules.append(f"{name}.gate_proj")
                        elif 'c_fc' in sub_name:
                            modules.append(f"{name}.c_fc")

            if modules:
                logger.info(f"Auto-detected modules: {modules}")
                return modules[:3]  # Limit to 3 modules for stability
            else:
                logger.warning("No suitable modules found, using default")
                return ['attn.c_attn', 'mlp.c_fc']

        except Exception as e:
            logger.error(f"Error in auto-detection: {e}")
            return ['attn.c_attn', 'mlp.c_fc']

    def _check_initialized(self) -> bool:
        """Check if engine is properly initialized."""
        if not self._initialized:
            logger.error("Engine not initialized. Call initialize() first.")
            return False
        return True

    def _build_context_prompt(self, current_prompt: str) -> str:
        """
        Build a context-aware prompt using conversation history.

        Args:
            current_prompt: The current user input

        Returns:
            Context-enhanced prompt
        """
        if not self.conversation_history:
            return current_prompt

        # For single queries without context keywords, don't use context
        context_keywords = ['what did i', 'my name', 'i told you', 'remember', 'earlier', 'before']
        if not any(keyword in current_prompt.lower() for keyword in context_keywords):
            return current_prompt

        # Build context from recent history
        context_parts = []

        # Add recent exchanges (limit to avoid token overflow)
        recent_history = self.conversation_history[-2:]  # Last 2 exchanges only

        for exchange in recent_history:
            context_parts.append(f"Previous: {exchange['user']} -> {exchange['assistant'][:100]}...")

        # Add current prompt
        context_parts.append(f"Current question: {current_prompt}")

        return "\n".join(context_parts)

    def _add_to_history(self, user_input: str, assistant_response: str) -> None:
        """
        Add an exchange to conversation history.

        Args:
            user_input: User's input
            assistant_response: Assistant's response
        """
        exchange = {
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': time.time()
        }

        self.conversation_history.append(exchange)

        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    def set_conversation_context(self, enabled: bool) -> None:
        """Enable or disable conversation context."""
        self.use_conversation_context = enabled
        logger.info(f"Conversation context {'enabled' if enabled else 'disabled'}")

    def _post_process_response(self, response: str) -> str:
        """
        Post-process the generated response for better quality.

        Args:
            response: Raw generated response

        Returns:
            Cleaned and improved response
        """
        if not response:
            return response

        # Remove common artifacts
        response = response.strip()

        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and sentences[-1].strip() and len(sentences[-1].strip()) < 10:
            # Last sentence is very short, likely incomplete
            response = '.'.join(sentences[:-1]) + '.'

        # Remove repetitive patterns
        response = self._remove_repetitive_patterns(response)

        # Clean up formatting
        response = self._clean_formatting(response)

        # Ensure proper capitalization
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]

        return response

    def _remove_repetitive_patterns(self, text: str) -> str:
        """Remove repetitive patterns from text."""
        import re

        # Remove repeated phrases (3+ words repeated)
        words = text.split()
        if len(words) < 6:
            return text

        # Check for repeated 3-word patterns
        for i in range(len(words) - 5):
            phrase = ' '.join(words[i:i+3])
            next_phrase = ' '.join(words[i+3:i+6])
            if phrase == next_phrase:
                # Found repetition, truncate
                return ' '.join(words[:i+3])

        return text

    def _clean_formatting(self, text: str) -> str:
        """Clean up text formatting."""
        import re

        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)

        # Remove trailing incomplete words
        if text.endswith(' '):
            text = text.rstrip()

        return text
    
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
