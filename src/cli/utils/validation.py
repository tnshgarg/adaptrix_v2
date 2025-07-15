"""
Validation utilities for the Adaptrix CLI.

This module provides functions for validating user inputs
such as model names, adapter names, and file paths.
"""

import re
import os
from pathlib import Path
from typing import Optional, List

def validate_model_name(model_name: str) -> bool:
    """
    Validate a model name.
    
    Args:
        model_name: Model name to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not model_name:
        return False
    
    # Check for valid HuggingFace model name format
    # Format: organization/model-name or just model-name
    pattern = r'^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)?$'
    return bool(re.match(pattern, model_name))

def validate_adapter_name(adapter_name: str) -> bool:
    """
    Validate an adapter name.
    
    Args:
        adapter_name: Adapter name to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not adapter_name:
        return False
    
    # Check for valid adapter name format
    # Should be alphanumeric with underscores and hyphens
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, adapter_name))

def validate_path(path: str, must_exist: bool = False, must_be_file: bool = False, must_be_dir: bool = False) -> bool:
    """
    Validate a file or directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file
        must_be_dir: Whether the path must be a directory
    
    Returns:
        True if valid, False otherwise
    """
    if not path:
        return False
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        return False
    
    if must_be_file and path_obj.exists() and not path_obj.is_file():
        return False
    
    if must_be_dir and path_obj.exists() and not path_obj.is_dir():
        return False
    
    return True

def validate_config_value(key: str, value: str) -> bool:
    """
    Validate a configuration value.
    
    Args:
        key: Configuration key
        value: Configuration value
    
    Returns:
        True if valid, False otherwise
    """
    # Define validation rules for different config keys
    validation_rules = {
        'models_dir': lambda v: validate_path(v, must_be_dir=True),
        'adapters_dir': lambda v: validate_path(v, must_be_dir=True),
        'rag_dir': lambda v: validate_path(v, must_be_dir=True),
        'device': lambda v: v in ['auto', 'cpu', 'cuda', 'mps'],
        'log_level': lambda v: v.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        'max_memory': lambda v: v.isdigit() and int(v) > 0,
    }
    
    if key in validation_rules:
        return validation_rules[key](value)
    
    # Default validation - non-empty string
    return bool(value.strip())

def validate_model_size(model_name: str, max_params: float = 3.0) -> bool:
    """
    Validate that a model is under the parameter limit.
    
    Args:
        model_name: Model name to check
        max_params: Maximum parameters in billions (default: 3.0B)
    
    Returns:
        True if model is under limit, False otherwise
    """
    # Model size mapping (in billions of parameters)
    model_sizes = {
        'qwen/qwen3-1.7b': 1.7,
        'microsoft/phi-2': 2.7,
        'microsoft/phi-3-mini': 3.8,  # Over limit
        'meta-llama/llama-2-7b': 7.0,  # Over limit
        'mistralai/mistral-7b': 7.0,  # Over limit
    }
    
    # Normalize model name
    normalized_name = model_name.lower()
    
    # Check if we have size information
    if normalized_name in model_sizes:
        return model_sizes[normalized_name] <= max_params
    
    # If we don't have size info, check for common patterns
    if any(size in normalized_name for size in ['7b', '8b', '13b', '30b', '70b']):
        return False  # These are definitely over 3B
    
    if any(size in normalized_name for size in ['1b', '2b', '3b']):
        return True  # These are likely under 3B
    
    # Default to allowing if we can't determine size
    return True

def get_validation_error_message(validation_type: str, value: str) -> str:
    """
    Get a descriptive error message for validation failures.
    
    Args:
        validation_type: Type of validation that failed
        value: Value that failed validation
    
    Returns:
        Error message string
    """
    messages = {
        'model_name': f"Invalid model name '{value}'. Must be in format 'organization/model' or 'model'.",
        'adapter_name': f"Invalid adapter name '{value}'. Must contain only letters, numbers, underscores, and hyphens.",
        'path': f"Invalid path '{value}'.",
        'config_value': f"Invalid configuration value '{value}'.",
        'model_size': f"Model '{value}' exceeds the 3B parameter limit.",
    }
    
    return messages.get(validation_type, f"Invalid value '{value}'.")
