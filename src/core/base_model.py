"""
Base model management for Adaptrix.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseModelManager:
    """Manages the base model loading and configuration."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def load_model(self):
        """Load the base model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_type": getattr(self.model.config, 'model_type', 'unknown')
        }
