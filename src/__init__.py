"""
Adaptrix: Middle-Layer LoRA Injection System

A revolutionary AI system that enhances small language models by dynamically 
injecting specialized LoRA adapters into middle transformer layers.
"""

__version__ = "0.1.0"
__author__ = "Adaptrix Team"
__description__ = "Middle-Layer LoRA Injection System for Enhanced Language Models"

from .core.engine import AdaptrixEngine
from .core.base_model import BaseModelManager
from .adapters.adapter_manager import AdapterManager
from .core.layer_injector import LayerInjector
from .core.dynamic_loader import DynamicLoader
from .composition.adapter_composer import AdapterComposer, CompositionStrategy

__all__ = [
    "AdaptrixEngine",
    "BaseModelManager",
    "AdapterManager",
    "LayerInjector",
    "DynamicLoader",
    "AdapterComposer",
    "CompositionStrategy"
]
