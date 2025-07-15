"""
Optimized Inference Engine for Adaptrix.

This module provides an enhanced engine that integrates vLLM, quantization,
and caching for maximum performance while maintaining compatibility with
the existing MoE and RAG systems.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..moe.moe_engine import MoEAdaptrixEngine
from .vllm_engine import VLLMInferenceEngine, VLLMConfig, VLLMModelAdapter
from .quantization import QuantizationManager, QuantizationConfig, QuantizationType
from .caching import CacheManager, create_default_cache_manager
from ..core.base_model_interface import ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)


class OptimizedAdaptrixEngine(MoEAdaptrixEngine):
    """
    Optimized Adaptrix Engine with vLLM, quantization, and caching.
    
    This engine extends the MoE engine with:
    - vLLM for high-performance inference
    - Model quantization for memory efficiency
    - Multi-level caching for speed
    - Automatic optimization selection
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        adapters_dir: str = "adapters",
        classifier_path: str = "models/classifier",
        enable_auto_selection: bool = True,
        rag_vector_store_path: Optional[str] = None,
        enable_rag: bool = False,
        # vLLM configuration
        use_vllm: bool = True,
        vllm_config: Optional[VLLMConfig] = None,
        # Quantization configuration
        quantization_config: Optional[QuantizationConfig] = None,
        # Caching configuration
        enable_caching: bool = True,
        cache_manager: Optional[CacheManager] = None,
        # Performance settings
        max_batch_size: int = 32,
        enable_async: bool = False
    ):
        """
        Initialize optimized Adaptrix engine.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to run on ("auto", "cpu", "cuda", etc.)
            adapters_dir: Directory containing LoRA adapters
            classifier_path: Path to trained task classifier
            enable_auto_selection: Whether to enable automatic adapter selection
            rag_vector_store_path: Path to RAG vector store
            enable_rag: Whether to enable RAG functionality
            use_vllm: Whether to use vLLM for inference
            vllm_config: vLLM configuration
            quantization_config: Quantization configuration
            enable_caching: Whether to enable caching
            cache_manager: Custom cache manager
            max_batch_size: Maximum batch size for inference
            enable_async: Whether to enable async inference
        """
        # Initialize base MoE engine
        super().__init__(
            model_id=model_id,
            device=device,
            adapters_dir=adapters_dir,
            classifier_path=classifier_path,
            enable_auto_selection=enable_auto_selection,
            rag_vector_store_path=rag_vector_store_path,
            enable_rag=enable_rag
        )
        
        # Optimization settings
        self.use_vllm = use_vllm
        self.vllm_config = vllm_config
        self.quantization_config = quantization_config
        self.enable_caching = enable_caching
        self.max_batch_size = max_batch_size
        self.enable_async = enable_async
        
        # Optimization components
        self.vllm_engine: Optional[VLLMInferenceEngine] = None
        self.quantization_manager: Optional[QuantizationManager] = None
        self.cache_manager: Optional[CacheManager] = None
        
        # State tracking
        self.vllm_initialized = False
        self.optimization_enabled = False
        
        # Use provided cache manager or create default
        if enable_caching:
            self.cache_manager = cache_manager or create_default_cache_manager()
        
        logger.info(f"Initialized optimized Adaptrix engine with vLLM: {use_vllm}, caching: {enable_caching}")
    
    def initialize(self) -> bool:
        """Initialize the optimized engine with all components."""
        try:
            logger.info("🚀 Initializing optimized Adaptrix engine")
            
            # Initialize quantization manager
            if self.quantization_config:
                self.quantization_manager = QuantizationManager()
                if not self.quantization_manager.validate_config(self.quantization_config):
                    logger.warning("Invalid quantization config, disabling quantization")
                    self.quantization_config = None
            
            # Initialize vLLM if enabled and available
            if self.use_vllm:
                if not self._initialize_vllm():
                    logger.warning("vLLM initialization failed, falling back to standard engine")
                    self.use_vllm = False
            
            # Initialize base MoE engine if vLLM not used
            if not self.vllm_initialized:
                if not super().initialize():
                    logger.error("Failed to initialize base MoE engine")
                    return False
            
            self.optimization_enabled = True
            logger.info("✅ Optimized Adaptrix engine initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Optimized engine initialization failed: {e}")
            return False
    
    def _initialize_vllm(self) -> bool:
        """Initialize vLLM engine."""
        try:
            # Create vLLM config if not provided
            if self.vllm_config is None:
                self.vllm_config = VLLMConfig(
                    model_id=self.model_id,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.9,
                    max_num_seqs=self.max_batch_size,
                    enable_lora=True,
                    enable_prefix_caching=True
                )
                
                # Apply quantization if configured
                if self.quantization_config and self.quantization_manager:
                    quant_param = self.quantization_manager.get_vllm_quantization_param(
                        self.quantization_config
                    )
                    if quant_param:
                        self.vllm_config.quantization = quant_param
            
            # Initialize vLLM engine
            self.vllm_engine = VLLMInferenceEngine(self.vllm_config)
            
            if self.vllm_engine.initialize(async_mode=self.enable_async):
                self.vllm_initialized = True
                logger.info("✅ vLLM engine initialized successfully")
                return True
            else:
                logger.error("❌ vLLM engine initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"vLLM initialization error: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        task_type: str = "auto",
        use_context: bool = None,
        adapter_name: str = None,
        use_rag: bool = None,
        rag_top_k: int = 3,
        use_cache: bool = None,
        **kwargs
    ) -> str:
        """
        Generate text with full optimization pipeline.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            task_type: Task type ('auto' for automatic selection)
            use_context: Whether to use conversation context
            adapter_name: Specific adapter to use
            use_rag: Whether to use RAG
            rag_top_k: Number of documents to retrieve for RAG
            use_cache: Whether to use response caching
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.optimization_enabled:
            return "Error: Optimized engine not initialized"
        
        try:
            # Determine cache usage
            should_use_cache = use_cache if use_cache is not None else self.enable_caching
            
            # Check cache first
            if should_use_cache and self.cache_manager:
                generation_config = {
                    "max_length": max_length,
                    "task_type": task_type,
                    "adapter_name": adapter_name,
                    "use_rag": use_rag,
                    "rag_top_k": rag_top_k,
                    **kwargs
                }
                
                cached_response = self.cache_manager.response_cache.get_response(
                    prompt, generation_config, adapter_name
                )
                
                if cached_response:
                    logger.debug("Cache hit for prompt")
                    return cached_response
            
            # Use vLLM if available and no adapter switching needed
            if (self.vllm_initialized and 
                (adapter_name is None or not self.enable_auto_selection)):
                
                response = self._generate_with_vllm(
                    prompt, max_length, task_type, use_rag, rag_top_k, **kwargs
                )
            else:
                # Use MoE engine for adapter selection and RAG
                response = super().generate(
                    prompt=prompt,
                    max_length=max_length,
                    task_type=task_type,
                    use_context=use_context,
                    adapter_name=adapter_name,
                    use_rag=use_rag,
                    rag_top_k=rag_top_k,
                    **kwargs
                )
            
            # Cache the response
            if should_use_cache and self.cache_manager and response:
                generation_config = {
                    "max_length": max_length,
                    "task_type": task_type,
                    "adapter_name": adapter_name,
                    "use_rag": use_rag,
                    "rag_top_k": rag_top_k,
                    **kwargs
                }
                
                self.cache_manager.response_cache.cache_response(
                    prompt, generation_config, response, adapter_name
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Optimized generation failed: {e}")
            return f"Error: Optimized generation failed - {str(e)}"
    
    def _generate_with_vllm(
        self,
        prompt: str,
        max_length: int,
        task_type: str,
        use_rag: bool,
        rag_top_k: int,
        **kwargs
    ) -> str:
        """Generate text using vLLM engine."""
        try:
            # Handle RAG if enabled
            final_prompt = prompt
            if use_rag and self.rag_initialized:
                retrieval_results = self.retriever.retrieve(
                    prompt, top_k=rag_top_k, score_threshold=0.0
                )
                
                if retrieval_results:
                    rag_context = self.retriever.create_context(
                        retrieval_results, max_context_length=1500
                    )
                    final_prompt = f"Context:\n{rag_context}\n\nQuestion: {prompt}"
            
            # Create generation config
            generation_config = GenerationConfig(
                max_new_tokens=max_length or 150,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                stop_sequences=kwargs.get('stop_sequences', [])
            )
            
            # Generate with vLLM
            if self.enable_async:
                # Note: This would need to be called from an async context
                # For now, use sync generation
                response = self.vllm_engine.generate(final_prompt, generation_config)
            else:
                response = self.vllm_engine.generate(final_prompt, generation_config)
            
            return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if not self.optimization_enabled:
            return ["Error: Optimized engine not initialized"] * len(prompts)
        
        try:
            if self.vllm_initialized and len(prompts) <= self.max_batch_size:
                # Use vLLM for batch generation
                generation_config = GenerationConfig(
                    max_new_tokens=max_length or 150,
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    top_k=kwargs.get('top_k', 50)
                )
                
                responses = self.vllm_engine.generate(prompts, generation_config)
                return responses if isinstance(responses, list) else [responses]
            else:
                # Fall back to sequential generation
                responses = []
                for prompt in prompts:
                    response = self.generate(prompt, max_length, **kwargs)
                    responses.append(response)
                return responses
                
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [f"Error: Batch generation failed - {str(e)}"] * len(prompts)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization system status."""
        base_status = self.get_moe_status()
        
        optimization_status = {
            "vllm_enabled": self.use_vllm,
            "vllm_initialized": self.vllm_initialized,
            "quantization_enabled": self.quantization_config is not None,
            "caching_enabled": self.enable_caching,
            "async_enabled": self.enable_async,
            "max_batch_size": self.max_batch_size,
            "optimization_enabled": self.optimization_enabled
        }
        
        # Add vLLM stats
        if self.vllm_engine:
            optimization_status["vllm_stats"] = self.vllm_engine.get_stats()
        
        # Add quantization info
        if self.quantization_config:
            optimization_status["quantization_config"] = {
                "type": self.quantization_config.quantization_type.value,
                "load_in_8bit": self.quantization_config.load_in_8bit,
                "load_in_4bit": self.quantization_config.load_in_4bit
            }
        
        # Add cache stats
        if self.cache_manager:
            optimization_status["cache_stats"] = self.cache_manager.get_global_stats()
        
        base_status["optimization"] = optimization_status
        return base_status
    
    def cleanup(self):
        """Clean up optimized engine resources."""
        try:
            # Clean up vLLM
            if self.vllm_engine:
                self.vllm_engine.cleanup()
                self.vllm_engine = None
            
            # Clean up cache manager
            if self.cache_manager:
                self.cache_manager.cleanup()
                self.cache_manager = None
            
            # Clean up base engine
            super().cleanup()
            
            self.optimization_enabled = False
            logger.info("✅ Optimized engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Optimized engine cleanup failed: {e}")
