"""
Helper utilities for Adaptrix system.
"""

import os
import json
import torch
import psutil
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_device(preferred_device: str = "auto") -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        preferred_device: Preferred device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device object
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(preferred_device)
        logger.info(f"Using specified device: {device}")
    
    return device


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory information in GB
    """
    memory_info = {}
    
    # System memory
    system_memory = psutil.virtual_memory()
    memory_info['system_total'] = system_memory.total / (1024**3)
    memory_info['system_available'] = system_memory.available / (1024**3)
    memory_info['system_used'] = system_memory.used / (1024**3)
    memory_info['system_percent'] = system_memory.percent
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_stats()
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / (1024**3)
        memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / (1024**3)
    
    return memory_info


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Union[int, float]]:
    """
    Calculate model size and parameter count.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (assuming fp32)
    model_size_mb = param_count * 4 / (1024**2)
    
    return {
        'total_params': param_count,
        'trainable_params': trainable_params,
        'non_trainable_params': param_count - trainable_params,
        'size_mb': model_size_mb,
        'size_gb': model_size_mb / 1024
    }


def create_hash(data: Union[str, bytes, Dict]) -> str:
    """
    Create SHA256 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_load(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Safely load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading JSON file {file_path}: {e}")
        return None


def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Safely save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager for timing operations.
    
    Args:
        description: Description of the operation being timed
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"{description} completed in {duration:.3f} seconds")


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def validate_adapter_structure(adapter_data: Dict[str, Any]) -> List[str]:
    """
    Validate adapter data structure.
    
    Args:
        adapter_data: Adapter data to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_fields = ['name', 'version', 'target_layers', 'rank', 'alpha']
    for field in required_fields:
        if field not in adapter_data:
            errors.append(f"Missing required field: {field}")
    
    if 'target_layers' in adapter_data:
        if not isinstance(adapter_data['target_layers'], list):
            errors.append("target_layers must be a list")
        elif not all(isinstance(layer, int) for layer in adapter_data['target_layers']):
            errors.append("All target_layers must be integers")
    
    if 'rank' in adapter_data:
        if not isinstance(adapter_data['rank'], int) or adapter_data['rank'] <= 0:
            errors.append("rank must be a positive integer")
    
    if 'alpha' in adapter_data:
        if not isinstance(adapter_data['alpha'], (int, float)) or adapter_data['alpha'] <= 0:
            errors.append("alpha must be a positive number")
    
    return errors


def cleanup_old_files(directory: Union[str, Path], max_age_days: int = 7) -> int:
    """
    Clean up old files in directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age of files to keep
        
    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    deleted_count = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
    
    return deleted_count
