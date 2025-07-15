"""
Adaptrix CLI - Command Line Interface for the Adaptrix System

A powerful CLI tool that provides access to the Adaptrix modular AI system,
allowing users to run models, manage adapters, integrate RAG, and build custom AI solutions.
"""

__version__ = "1.0.0"
__author__ = "Adaptrix Team"
__description__ = "Command Line Interface for Adaptrix Modular AI System"

from .main import cli

__all__ = ["cli"]
