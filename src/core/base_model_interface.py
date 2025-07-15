"""
Abstract Base Model Interface for Adaptrix.

This module provides a standardized interface for any LLM base model,
enabling plug-and-play functionality with different models and adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelFamily(Enum):
    """Supported model families."""
    QWEN = "qwen"
    PHI = "phi"
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    GEMMA = "gemma"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """Configuration for a base model."""
    model_id: str
    model_family: ModelFamily
    context_length: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    device: str = "cpu"
    torch_dtype: torch.dtype = torch.float32
    trust_remote_code: bool = True
    
    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    min_new_tokens: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None


class BaseModelInterface(ABC):
    """Abstract interface for all base models in Adaptrix."""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._device = model_config.device
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate text from the model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        pass
    
    @abstractmethod
    def get_adapter_compatibility(self) -> Dict[str, Any]:
        """Get adapter compatibility information."""
        pass
    
    @abstractmethod
    def validate_adapter(self, adapter_path: str) -> bool:
        """Validate if an adapter is compatible with this model."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass
    
    # Common utility methods
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized
    
    def get_device(self) -> str:
        """Get current device."""
        return self._device
    
    def get_context_length(self) -> int:
        """Get model context length."""
        return self.config.context_length
    
    def get_model_family(self) -> ModelFamily:
        """Get model family."""
        return self.config.model_family


class ModelRegistry:
    """Registry for managing different model implementations."""
    
    _models: Dict[str, type] = {}
    _configs: Dict[str, ModelConfig] = {}
    
    @classmethod
    def register_model(cls, model_id: str, model_class: type, model_config: ModelConfig):
        """Register a new model implementation."""
        cls._models[model_id] = model_class
        cls._configs[model_id] = model_config
        logger.info(f"Registered model: {model_id}")
    
    @classmethod
    def get_model(cls, model_id: str, device: str = "cpu") -> Optional[BaseModelInterface]:
        """Get a model instance by ID."""
        if model_id not in cls._models:
            logger.error(f"Model {model_id} not registered")
            return None
        
        config = cls._configs[model_id]
        config.device = device  # Override device
        
        model_class = cls._models[model_id]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        return cls._configs.get(model_id)


class ModelDetector:
    """Automatically detect model family and configuration."""
    
    FAMILY_PATTERNS = {
        ModelFamily.QWEN: ["qwen", "qwen2", "qwen3"],
        ModelFamily.PHI: ["phi", "phi-2", "phi-3"],
        ModelFamily.LLAMA: ["llama", "llama2", "llama3"],
        ModelFamily.MISTRAL: ["mistral", "mixtral"],
        ModelFamily.DEEPSEEK: ["deepseek"],
        ModelFamily.GEMMA: ["gemma"],
    }
    
    @classmethod
    def detect_family(cls, model_id: str) -> ModelFamily:
        """Detect model family from model ID."""
        model_id_lower = model_id.lower()
        
        for family, patterns in cls.FAMILY_PATTERNS.items():
            if any(pattern in model_id_lower for pattern in patterns):
                return family
        
        return ModelFamily.UNKNOWN
    
    @classmethod
    def create_config_from_model_id(cls, model_id: str, device: str = "cpu") -> ModelConfig:
        """Create a model config by detecting from model ID."""
        family = cls.detect_family(model_id)
        
        # Default configurations for different families
        family_defaults = {
            ModelFamily.QWEN: {
                "context_length": 32768,
                "vocab_size": 151936,
                "hidden_size": 1536,
                "num_layers": 28,
                "num_attention_heads": 12,
                "default_max_tokens": 1024,
            },
            ModelFamily.PHI: {
                "context_length": 2048,
                "vocab_size": 51200,
                "hidden_size": 2560,
                "num_layers": 32,
                "num_attention_heads": 32,
                "default_max_tokens": 512,
            },
            ModelFamily.LLAMA: {
                "context_length": 4096,
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "default_max_tokens": 1024,
            },
            ModelFamily.MISTRAL: {
                "context_length": 8192,
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "default_max_tokens": 1024,
            },
        }
        
        defaults = family_defaults.get(family, family_defaults[ModelFamily.QWEN])
        
        return ModelConfig(
            model_id=model_id,
            model_family=family,
            device=device,
            **defaults
        )


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(
        model_id: str,
        device: str = "cpu",
        use_vllm: bool = False,
        vllm_config: Optional[Any] = None,
        quantization_config: Optional[Any] = None,
        **kwargs
    ) -> BaseModelInterface:
        """
        Create a model instance with automatic detection.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run the model on
            use_vllm: Whether to use vLLM for inference
            vllm_config: vLLM configuration
            quantization_config: Quantization configuration
            **kwargs: Additional model configuration
        """

        # First try registry
        model = ModelRegistry.get_model(model_id, device)
        if model:
            return model

        # Auto-detect and create config
        config = ModelDetector.create_config_from_model_id(model_id, device)

        # Use vLLM if requested and available
        if use_vllm:
            try:
                from ..inference.vllm_engine import VLLMModelAdapter, VLLMConfig
                from ..inference.quantization import QuantizationManager

                # Create vLLM config
                if vllm_config is None:
                    vllm_config = VLLMConfig(
                        model_id=model_id,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.9,
                        enable_lora=True,
                        enable_prefix_caching=True
                    )

                    # Apply quantization if configured
                    if quantization_config:
                        quant_manager = QuantizationManager()
                        quant_param = quant_manager.get_vllm_quantization_param(quantization_config)
                        if quant_param:
                            vllm_config.quantization = quant_param

                return VLLMModelAdapter(config, vllm_config)

            except ImportError:
                logger.warning("vLLM not available, falling back to standard model")

        # Import and create appropriate implementation
        if config.model_family == ModelFamily.QWEN:
            from .models.qwen_model import QwenModel
            return QwenModel(config)
        elif config.model_family == ModelFamily.PHI:
            # Use generic model since phi_model doesn't exist
            from .models.generic_model import GenericModel
            return GenericModel(config)
        elif config.model_family == ModelFamily.LLAMA:
            # Use generic model since llama_model doesn't exist
            from .models.generic_model import GenericModel
            return GenericModel(config)
        elif config.model_family == ModelFamily.MISTRAL:
            # Use generic model since mistral_model doesn't exist
            from .models.generic_model import GenericModel
            return GenericModel(config)
        else:
            # Fallback to generic implementation
            from .models.generic_model import GenericModel
            return GenericModel(config)


# Utility functions
def get_optimal_generation_config(model_family: ModelFamily, task_type: str = "general") -> GenerationConfig:
    """Get optimal generation configuration for model family and task."""
    
    base_configs = {
        ModelFamily.QWEN: GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        ),
        ModelFamily.PHI: GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
        ),
        ModelFamily.LLAMA: GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        ),
    }
    
    config = base_configs.get(model_family, GenerationConfig())
    
    # Task-specific adjustments
    if task_type == "code":
        config.temperature = 0.3
        config.top_p = 0.95
        config.max_new_tokens = 1024
    elif task_type == "math":
        config.temperature = 0.1
        config.top_p = 0.9
        config.max_new_tokens = 512
    elif task_type == "creative":
        config.temperature = 0.9
        config.top_p = 0.95
        config.max_new_tokens = 1024
    
    return config
