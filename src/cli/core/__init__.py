"""
Core CLI functionality for Adaptrix.

This package contains the core managers and utilities that handle
the integration with the Adaptrix system components.
"""

from .engine_manager import EngineManager
from .model_manager import ModelManager
from .adapter_manager import CLIAdapterManager
from .rag_manager import RAGManager
from .config_manager import ConfigManager

__all__ = [
    "EngineManager",
    "ModelManager", 
    "CLIAdapterManager",
    "RAGManager",
    "ConfigManager"
]
