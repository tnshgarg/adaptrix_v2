"""
CLI Commands for Adaptrix.

This package contains all the command groups and individual commands
for the Adaptrix CLI interface.
"""

from .models import models_group
from .adapters import adapters_group
from .rag import rag_group
from .build import build_group
from .inference import run_command, chat_command
from .config import config_group

__all__ = [
    "models_group",
    "adapters_group", 
    "rag_group",
    "build_group",
    "run_command",
    "chat_command",
    "config_group"
]
