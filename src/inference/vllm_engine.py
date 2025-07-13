"""
vLLM Inference Engine for Adaptrix.

This module provides optimized inference using vLLM for high-throughput
and low-latency text generation with advanced caching and batching.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import time

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None

from ..core.base_model_interface import BaseModelInterface, ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine."""
    model_id: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm", "fp8"
    dtype: str = "auto"
    seed: int = 42
    trust_remote_code: bool = True
    enable_lora: bool = True
    max_lora_rank: int = 64
    enable_prefix_caching: bool = True
    disable_log_stats: bool = False


class VLLMInferenceEngine:
    """
    vLLM-based inference engine for optimized text generation.
    
    Provides high-throughput, low-latency inference with advanced features:
    - Continuous batching
    - PagedAttention for memory efficiency
    - LoRA adapter support
    - Prefix caching
    - Quantization support
    """
    
    def __init__(self, config: VLLMConfig):
        """
        Initialize vLLM inference engine.
        
        Args:
            config: vLLM configuration
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install with: pip install vllm"
            )
        
        self.config = config
        self.llm: Optional[LLM] = None
        self.async_engine: Optional[AsyncLLMEngine] = None
        self.is_initialized = False
        self.is_async = False
        
        # Performance tracking
        self.generation_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
            "avg_tokens_per_second": 0.0,
            "avg_latency": 0.0
        }
        
        logger.info(f"Initialized vLLM engine for {config.model_id}")
    
    def initialize(self, async_mode: bool = False) -> bool:
        """
        Initialize the vLLM engine.
        
        Args:
            async_mode: Whether to use async engine for concurrent requests
            
        Returns:
            Success status
        """
        try:
            logger.info(f"🚀 Initializing vLLM engine: {self.config.model_id}")
            
            if async_mode:
                self._initialize_async_engine()
            else:
                self._initialize_sync_engine()
            
            self.is_async = async_mode
            self.is_initialized = True
            
            logger.info("✅ vLLM engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vLLM engine: {e}")
            return False
    
    def _initialize_sync_engine(self):
        """Initialize synchronous vLLM engine."""
        self.llm = LLM(
            model=self.config.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            seed=self.config.seed,
            trust_remote_code=self.config.trust_remote_code,
            enable_lora=self.config.enable_lora,
            max_lora_rank=self.config.max_lora_rank,
            enable_prefix_caching=self.config.enable_prefix_caching,
            disable_log_stats=self.config.disable_log_stats
        )
    
    def _initialize_async_engine(self):
        """Initialize asynchronous vLLM engine."""
        engine_args = AsyncEngineArgs(
            model=self.config.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            seed=self.config.seed,
            trust_remote_code=self.config.trust_remote_code,
            enable_lora=self.config.enable_lora,
            max_lora_rank=self.config.max_lora_rank,
            enable_prefix_caching=self.config.enable_prefix_caching,
            disable_log_stats=self.config.disable_log_stats
        )
        self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        generation_config: GenerationConfig,
        lora_request: Optional[Any] = None
    ) -> Union[str, List[str]]:
        """
        Generate text using vLLM (synchronous).
        
        Args:
            prompts: Single prompt or list of prompts
            generation_config: Generation configuration
            lora_request: Optional LoRA request for adapter
            
        Returns:
            Generated text(s)
        """
        if not self.is_initialized or self.is_async:
            raise RuntimeError("Engine not initialized or in async mode")
        
        # Convert single prompt to list
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        try:
            start_time = time.time()
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                max_tokens=generation_config.max_new_tokens,
                stop=generation_config.stop_sequences,
                repetition_penalty=getattr(generation_config, 'repetition_penalty', 1.0),
                length_penalty=getattr(generation_config, 'length_penalty', 1.0),
                seed=getattr(generation_config, 'seed', None)
            )
            
            # Generate
            outputs = self.llm.generate(
                prompts, 
                sampling_params,
                lora_request=lora_request
            )
            
            # Extract generated text
            results = []
            total_tokens = 0
            
            for output in outputs:
                generated_text = output.outputs[0].text
                results.append(generated_text)
                total_tokens += len(output.outputs[0].token_ids)
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_stats(len(prompts), total_tokens, generation_time)
            
            # Return single string if input was single string
            return results[0] if is_single else results
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return "Error: vLLM generation failed" if is_single else ["Error: vLLM generation failed"] * len(prompts)
    
    async def generate_async(
        self, 
        prompts: Union[str, List[str]], 
        generation_config: GenerationConfig,
        lora_request: Optional[Any] = None
    ) -> Union[str, List[str]]:
        """
        Generate text using vLLM (asynchronous).
        
        Args:
            prompts: Single prompt or list of prompts
            generation_config: Generation configuration
            lora_request: Optional LoRA request for adapter
            
        Returns:
            Generated text(s)
        """
        if not self.is_initialized or not self.is_async:
            raise RuntimeError("Engine not initialized or not in async mode")
        
        # Convert single prompt to list
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        try:
            start_time = time.time()
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                max_tokens=generation_config.max_new_tokens,
                stop=generation_config.stop_sequences,
                repetition_penalty=getattr(generation_config, 'repetition_penalty', 1.0),
                length_penalty=getattr(generation_config, 'length_penalty', 1.0),
                seed=getattr(generation_config, 'seed', None)
            )
            
            # Generate async
            results = []
            total_tokens = 0
            
            for i, prompt in enumerate(prompts):
                request_id = f"request_{i}_{int(time.time() * 1000)}"
                
                # Add request to engine
                await self.async_engine.add_request(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    lora_request=lora_request
                )
                
                # Get result
                async for request_output in self.async_engine.generate(request_id):
                    if request_output.finished:
                        generated_text = request_output.outputs[0].text
                        results.append(generated_text)
                        total_tokens += len(request_output.outputs[0].token_ids)
                        break
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_stats(len(prompts), total_tokens, generation_time)
            
            # Return single string if input was single string
            return results[0] if is_single else results
            
        except Exception as e:
            logger.error(f"vLLM async generation failed: {e}")
            return "Error: vLLM async generation failed" if is_single else ["Error: vLLM async generation failed"] * len(prompts)
    
    def _update_stats(self, num_requests: int, total_tokens: int, generation_time: float):
        """Update generation statistics."""
        self.generation_stats["total_requests"] += num_requests
        self.generation_stats["total_tokens_generated"] += total_tokens
        self.generation_stats["total_time"] += generation_time
        
        # Calculate averages
        if self.generation_stats["total_time"] > 0:
            self.generation_stats["avg_tokens_per_second"] = (
                self.generation_stats["total_tokens_generated"] / 
                self.generation_stats["total_time"]
            )
        
        if self.generation_stats["total_requests"] > 0:
            self.generation_stats["avg_latency"] = (
                self.generation_stats["total_time"] / 
                self.generation_stats["total_requests"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            **self.generation_stats,
            "config": {
                "model_id": self.config.model_id,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len,
                "quantization": self.config.quantization,
                "enable_lora": self.config.enable_lora,
                "enable_prefix_caching": self.config.enable_prefix_caching
            },
            "is_initialized": self.is_initialized,
            "is_async": self.is_async
        }
    
    def cleanup(self):
        """Clean up vLLM engine resources."""
        try:
            if self.llm is not None:
                # vLLM handles cleanup automatically
                self.llm = None
            
            if self.async_engine is not None:
                # Async engine cleanup
                self.async_engine = None
            
            self.is_initialized = False
            logger.info("✅ vLLM engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"vLLM cleanup failed: {e}")


class VLLMModelAdapter(BaseModelInterface):
    """
    Adapter to make vLLM engine compatible with BaseModelInterface.
    
    This allows vLLM to be used as a drop-in replacement for other models
    in the Adaptrix system.
    """
    
    def __init__(self, model_config: ModelConfig, vllm_config: VLLMConfig):
        """
        Initialize vLLM model adapter.
        
        Args:
            model_config: Base model configuration
            vllm_config: vLLM-specific configuration
        """
        super().__init__(model_config)
        self.vllm_config = vllm_config
        self.vllm_engine: Optional[VLLMInferenceEngine] = None
    
    def initialize(self) -> bool:
        """Initialize the vLLM model adapter."""
        try:
            self.vllm_engine = VLLMInferenceEngine(self.vllm_config)
            success = self.vllm_engine.initialize(async_mode=False)
            
            if success:
                self._initialized = True
                logger.info("✅ vLLM model adapter initialized")
            
            return success
            
        except Exception as e:
            logger.error(f"vLLM model adapter initialization failed: {e}")
            return False
    
    def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate text using vLLM."""
        if not self._initialized or not self.vllm_engine:
            return "Error: vLLM model adapter not initialized"
        
        try:
            result = self.vllm_engine.generate(prompt, generation_config)
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return f"Error: vLLM generation failed - {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information."""
        base_info = {
            "model_id": self.config.model_id,
            "model_family": self.config.model_family.value,
            "device": str(self.config.device),
            "inference_engine": "vLLM"
        }
        
        if self.vllm_engine:
            vllm_stats = self.vllm_engine.get_stats()
            base_info.update({
                "vllm_config": vllm_stats["config"],
                "generation_stats": {
                    k: v for k, v in vllm_stats.items() 
                    if k not in ["config", "is_initialized", "is_async"]
                }
            })
        
        return base_info
    
    def get_adapter_compatibility(self) -> Dict[str, Any]:
        """Get vLLM adapter compatibility information."""
        return {
            "supports_lora": self.vllm_config.enable_lora,
            "max_lora_rank": self.vllm_config.max_lora_rank,
            "supports_dynamic_loading": True,
            "supports_batching": True,
            "supports_prefix_caching": self.vllm_config.enable_prefix_caching
        }
    
    def validate_adapter(self, adapter_path: str) -> bool:
        """Validate adapter compatibility with vLLM."""
        # vLLM has its own adapter validation
        # For now, assume compatible if LoRA is enabled
        return self.vllm_config.enable_lora
    
    def cleanup(self):
        """Clean up vLLM model adapter."""
        if self.vllm_engine:
            self.vllm_engine.cleanup()
            self.vllm_engine = None
        
        self._initialized = False
        logger.info("✅ vLLM model adapter cleaned up")
