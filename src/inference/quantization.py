"""
Quantization Support for Adaptrix.

This module provides model quantization capabilities for memory efficiency
and faster inference, supporting various quantization methods.
"""

import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    bnb = None

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quantization_type: QuantizationType
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[list] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    
    # AWQ specific
    awq_bits: int = 4
    awq_group_size: int = 128
    
    # GPTQ specific
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.quantization_type == QuantizationType.INT8:
            self.load_in_8bit = True
        elif self.quantization_type == QuantizationType.INT4:
            self.load_in_4bit = True


class QuantizationManager:
    """
    Manages model quantization for memory efficiency and speed.
    
    Supports multiple quantization methods:
    - BitsAndBytes (4-bit, 8-bit)
    - AWQ (Activation-aware Weight Quantization)
    - GPTQ (Gradient-based Post-training Quantization)
    - SqueezeLLM
    - FP8 (for supported hardware)
    """
    
    def __init__(self):
        """Initialize quantization manager."""
        self.supported_methods = self._detect_supported_methods()
        logger.info(f"Quantization manager initialized. Supported methods: {list(self.supported_methods.keys())}")
    
    def _detect_supported_methods(self) -> Dict[str, bool]:
        """Detect which quantization methods are available."""
        methods = {
            "bitsandbytes": BITSANDBYTES_AVAILABLE,
            "awq": self._check_awq_support(),
            "gptq": self._check_gptq_support(),
            "squeezellm": self._check_squeezellm_support(),
            "fp8": self._check_fp8_support()
        }
        return methods
    
    def _check_awq_support(self) -> bool:
        """Check if AWQ is available."""
        try:
            import awq
            return True
        except ImportError:
            return False
    
    def _check_gptq_support(self) -> bool:
        """Check if GPTQ is available."""
        try:
            import auto_gptq
            return True
        except ImportError:
            return False
    
    def _check_squeezellm_support(self) -> bool:
        """Check if SqueezeLLM is available."""
        try:
            import squeezellm
            return True
        except ImportError:
            return False
    
    def _check_fp8_support(self) -> bool:
        """Check if FP8 quantization is supported."""
        # FP8 requires specific hardware (H100, etc.)
        if torch.cuda.is_available():
            try:
                # Check for FP8 support
                device_capability = torch.cuda.get_device_capability()
                # FP8 is supported on compute capability 8.9+ (H100)
                return device_capability[0] >= 8 and device_capability[1] >= 9
            except:
                return False
        return False
    
    def create_quantization_config(
        self, 
        quantization_type: Union[str, QuantizationType],
        **kwargs
    ) -> QuantizationConfig:
        """
        Create quantization configuration.
        
        Args:
            quantization_type: Type of quantization to use
            **kwargs: Additional configuration parameters
            
        Returns:
            QuantizationConfig object
        """
        if isinstance(quantization_type, str):
            quantization_type = QuantizationType(quantization_type.lower())
        
        # Validate support
        method_name = self._get_method_name(quantization_type)
        if method_name and not self.supported_methods.get(method_name, False):
            logger.warning(f"Quantization method {quantization_type.value} not supported, falling back to none")
            quantization_type = QuantizationType.NONE
        
        config = QuantizationConfig(quantization_type=quantization_type, **kwargs)
        
        logger.info(f"Created quantization config: {quantization_type.value}")
        return config
    
    def _get_method_name(self, quantization_type: QuantizationType) -> Optional[str]:
        """Get the method name for a quantization type."""
        mapping = {
            QuantizationType.INT8: "bitsandbytes",
            QuantizationType.INT4: "bitsandbytes",
            QuantizationType.AWQ: "awq",
            QuantizationType.GPTQ: "gptq",
            QuantizationType.SQUEEZELLM: "squeezellm",
            QuantizationType.FP8: "fp8"
        }
        return mapping.get(quantization_type)
    
    def get_model_loading_kwargs(self, config: QuantizationConfig) -> Dict[str, Any]:
        """
        Get model loading kwargs for quantization.
        
        Args:
            config: Quantization configuration
            
        Returns:
            Dictionary of kwargs for model loading
        """
        kwargs = {}
        
        if config.quantization_type == QuantizationType.NONE:
            return kwargs
        
        elif config.quantization_type in [QuantizationType.INT8, QuantizationType.INT4]:
            if not BITSANDBYTES_AVAILABLE:
                logger.warning("BitsAndBytes not available, skipping quantization")
                return kwargs
            
            if config.load_in_8bit:
                kwargs.update({
                    "load_in_8bit": True,
                    "llm_int8_threshold": config.llm_int8_threshold,
                    "llm_int8_skip_modules": config.llm_int8_skip_modules,
                    "llm_int8_enable_fp32_cpu_offload": config.llm_int8_enable_fp32_cpu_offload
                })
            
            elif config.load_in_4bit:
                kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": config.bnb_4bit_compute_dtype,
                    "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
                    "bnb_4bit_quant_type": config.bnb_4bit_quant_type
                })
        
        elif config.quantization_type == QuantizationType.AWQ:
            kwargs.update({
                "quantization_config": {
                    "bits": config.awq_bits,
                    "group_size": config.awq_group_size,
                    "quant_method": "awq"
                }
            })
        
        elif config.quantization_type == QuantizationType.GPTQ:
            kwargs.update({
                "quantization_config": {
                    "bits": config.gptq_bits,
                    "group_size": config.gptq_group_size,
                    "desc_act": config.gptq_desc_act,
                    "quant_method": "gptq"
                }
            })
        
        elif config.quantization_type == QuantizationType.SQUEEZELLM:
            kwargs.update({
                "quantization_config": {
                    "quant_method": "squeezellm"
                }
            })
        
        elif config.quantization_type == QuantizationType.FP8:
            kwargs.update({
                "torch_dtype": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
            })
        
        return kwargs
    
    def get_vllm_quantization_param(self, config: QuantizationConfig) -> Optional[str]:
        """
        Get vLLM quantization parameter.
        
        Args:
            config: Quantization configuration
            
        Returns:
            vLLM quantization parameter string
        """
        if config.quantization_type == QuantizationType.NONE:
            return None
        elif config.quantization_type == QuantizationType.AWQ:
            return "awq"
        elif config.quantization_type == QuantizationType.GPTQ:
            return "gptq"
        elif config.quantization_type == QuantizationType.SQUEEZELLM:
            return "squeezellm"
        elif config.quantization_type == QuantizationType.FP8:
            return "fp8"
        else:
            # BitsAndBytes quantization not directly supported by vLLM
            logger.warning(f"Quantization type {config.quantization_type.value} not supported by vLLM")
            return None
    
    def estimate_memory_savings(self, config: QuantizationConfig, base_memory_gb: float) -> Dict[str, Any]:
        """
        Estimate memory savings from quantization.
        
        Args:
            config: Quantization configuration
            base_memory_gb: Base model memory usage in GB
            
        Returns:
            Memory savings estimation
        """
        savings_factor = 1.0
        
        if config.quantization_type == QuantizationType.INT8:
            savings_factor = 0.5  # ~50% memory reduction
        elif config.quantization_type == QuantizationType.INT4:
            savings_factor = 0.25  # ~75% memory reduction
        elif config.quantization_type in [QuantizationType.AWQ, QuantizationType.GPTQ]:
            if config.quantization_type == QuantizationType.AWQ:
                bits = config.awq_bits
            else:
                bits = config.gptq_bits
            savings_factor = bits / 16.0  # Assuming FP16 baseline
        elif config.quantization_type == QuantizationType.SQUEEZELLM:
            savings_factor = 0.25  # Similar to 4-bit
        elif config.quantization_type == QuantizationType.FP8:
            savings_factor = 0.5  # ~50% memory reduction
        
        quantized_memory = base_memory_gb * savings_factor
        memory_saved = base_memory_gb - quantized_memory
        
        return {
            "base_memory_gb": base_memory_gb,
            "quantized_memory_gb": quantized_memory,
            "memory_saved_gb": memory_saved,
            "memory_reduction_percent": (memory_saved / base_memory_gb) * 100,
            "quantization_type": config.quantization_type.value
        }
    
    def get_supported_methods(self) -> Dict[str, bool]:
        """Get dictionary of supported quantization methods."""
        return self.supported_methods.copy()
    
    def validate_config(self, config: QuantizationConfig) -> bool:
        """
        Validate quantization configuration.
        
        Args:
            config: Quantization configuration to validate
            
        Returns:
            True if configuration is valid
        """
        try:
            # Check if method is supported
            method_name = self._get_method_name(config.quantization_type)
            if method_name and not self.supported_methods.get(method_name, False):
                logger.error(f"Quantization method {config.quantization_type.value} not supported")
                return False
            
            # Validate BitsAndBytes config
            if config.quantization_type in [QuantizationType.INT8, QuantizationType.INT4]:
                if not BITSANDBYTES_AVAILABLE:
                    logger.error("BitsAndBytes not available for int8/int4 quantization")
                    return False
                
                if config.load_in_8bit and config.load_in_4bit:
                    logger.error("Cannot use both 8-bit and 4-bit quantization")
                    return False
            
            # Validate AWQ config
            if config.quantization_type == QuantizationType.AWQ:
                if config.awq_bits not in [4, 8]:
                    logger.error(f"AWQ bits must be 4 or 8, got {config.awq_bits}")
                    return False
            
            # Validate GPTQ config
            if config.quantization_type == QuantizationType.GPTQ:
                if config.gptq_bits not in [2, 3, 4, 8]:
                    logger.error(f"GPTQ bits must be 2, 3, 4, or 8, got {config.gptq_bits}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Quantization config validation failed: {e}")
            return False


# Convenience functions
def create_int8_config(**kwargs) -> QuantizationConfig:
    """Create 8-bit quantization configuration."""
    return QuantizationConfig(quantization_type=QuantizationType.INT8, **kwargs)


def create_int4_config(**kwargs) -> QuantizationConfig:
    """Create 4-bit quantization configuration."""
    return QuantizationConfig(quantization_type=QuantizationType.INT4, **kwargs)


def create_awq_config(bits: int = 4, group_size: int = 128, **kwargs) -> QuantizationConfig:
    """Create AWQ quantization configuration."""
    return QuantizationConfig(
        quantization_type=QuantizationType.AWQ,
        awq_bits=bits,
        awq_group_size=group_size,
        **kwargs
    )


def create_gptq_config(bits: int = 4, group_size: int = 128, desc_act: bool = False, **kwargs) -> QuantizationConfig:
    """Create GPTQ quantization configuration."""
    return QuantizationConfig(
        quantization_type=QuantizationType.GPTQ,
        gptq_bits=bits,
        gptq_group_size=group_size,
        gptq_desc_act=desc_act,
        **kwargs
    )
