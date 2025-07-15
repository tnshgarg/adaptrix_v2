"""
Configuration for Adaptrix API.

This module provides configuration management for the FastAPI application.
"""

import os
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field


class APIConfig(BaseSettings):
    """Configuration for Adaptrix API."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="ADAPTRIX_HOST")
    port: int = Field(default=8000, env="ADAPTRIX_PORT")
    workers: int = Field(default=1, env="ADAPTRIX_WORKERS")
    reload: bool = Field(default=False, env="ADAPTRIX_RELOAD")
    log_level: str = Field(default="info", env="ADAPTRIX_LOG_LEVEL")
    
    # Model settings
    model_id: str = Field(default="Qwen/Qwen3-1.7B", env="ADAPTRIX_MODEL_ID")
    device: str = Field(default="auto", env="ADAPTRIX_DEVICE")
    
    # Directory settings
    adapters_dir: str = Field(default="adapters", env="ADAPTRIX_ADAPTERS_DIR")
    classifier_path: str = Field(default="models/classifier", env="ADAPTRIX_CLASSIFIER_PATH")
    rag_vector_store_path: Optional[str] = Field(default="models/rag_vector_store", env="ADAPTRIX_RAG_VECTOR_STORE_PATH")
    
    # Feature flags
    enable_auto_selection: bool = Field(default=True, env="ADAPTRIX_ENABLE_AUTO_SELECTION")
    enable_rag: bool = Field(default=True, env="ADAPTRIX_ENABLE_RAG")
    use_optimized_engine: bool = Field(default=True, env="ADAPTRIX_USE_OPTIMIZED_ENGINE")
    use_vllm: bool = Field(default=False, env="ADAPTRIX_USE_VLLM")  # Disabled by default for compatibility
    enable_quantization: bool = Field(default=False, env="ADAPTRIX_ENABLE_QUANTIZATION")
    enable_caching: bool = Field(default=True, env="ADAPTRIX_ENABLE_CACHING")
    enable_async: bool = Field(default=False, env="ADAPTRIX_ENABLE_ASYNC")
    
    # Performance settings
    max_batch_size: int = Field(default=32, env="ADAPTRIX_MAX_BATCH_SIZE")
    max_concurrent_requests: int = Field(default=100, env="ADAPTRIX_MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=300, env="ADAPTRIX_REQUEST_TIMEOUT")  # seconds
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, env="ADAPTRIX_ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="ADAPTRIX_RATE_LIMIT_REQUESTS")  # requests per minute
    rate_limit_window: int = Field(default=60, env="ADAPTRIX_RATE_LIMIT_WINDOW")  # seconds
    
    # Authentication
    enable_auth: bool = Field(default=False, env="ADAPTRIX_ENABLE_AUTH")
    api_key: Optional[str] = Field(default=None, env="ADAPTRIX_API_KEY")
    jwt_secret: Optional[str] = Field(default=None, env="ADAPTRIX_JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="ADAPTRIX_JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="ADAPTRIX_JWT_EXPIRATION")  # seconds
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="ADAPTRIX_CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["*"], env="ADAPTRIX_CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="ADAPTRIX_CORS_HEADERS")
    
    # Security
    trusted_hosts: List[str] = Field(default=["*"], env="ADAPTRIX_TRUSTED_HOSTS")
    max_request_size: int = Field(default=10 * 1024 * 1024, env="ADAPTRIX_MAX_REQUEST_SIZE")  # 10MB
    
    # Logging
    log_requests: bool = Field(default=True, env="ADAPTRIX_LOG_REQUESTS")
    log_responses: bool = Field(default=False, env="ADAPTRIX_LOG_RESPONSES")
    log_file: Optional[str] = Field(default=None, env="ADAPTRIX_LOG_FILE")
    
    # Development
    debug: bool = Field(default=False, env="ADAPTRIX_DEBUG")
    docs_url: Optional[str] = Field(default="/docs", env="ADAPTRIX_DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="ADAPTRIX_REDOC_URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="ADAPTRIX_OPENAPI_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
_api_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Get the global API configuration instance."""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config


def update_api_config(**kwargs) -> APIConfig:
    """Update the global API configuration."""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(_api_config, key):
                setattr(_api_config, key, value)
    return _api_config


def load_config_from_file(config_file: str) -> APIConfig:
    """Load configuration from a file."""
    import json
    import yaml
    from pathlib import Path
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return update_api_config(**config_data)


def get_environment_info() -> dict:
    """Get information about the current environment."""
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if key.startswith("ADAPTRIX_")
        }
    }


def validate_config(config: APIConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []
    
    # Check required paths
    if config.enable_auto_selection and not os.path.exists(config.classifier_path):
        warnings.append(f"Classifier path does not exist: {config.classifier_path}")
    
    if config.enable_rag and config.rag_vector_store_path and not os.path.exists(config.rag_vector_store_path):
        warnings.append(f"RAG vector store path does not exist: {config.rag_vector_store_path}")
    
    if not os.path.exists(config.adapters_dir):
        warnings.append(f"Adapters directory does not exist: {config.adapters_dir}")
    
    # Check authentication settings
    if config.enable_auth and not config.api_key and not config.jwt_secret:
        warnings.append("Authentication enabled but no API key or JWT secret provided")
    
    # Check vLLM compatibility
    if config.use_vllm:
        try:
            import vllm
        except ImportError:
            warnings.append("vLLM requested but not installed")
    
    # Check quantization compatibility
    if config.enable_quantization:
        try:
            import bitsandbytes
        except ImportError:
            warnings.append("Quantization enabled but bitsandbytes not installed")
    
    # Check device compatibility
    if config.device.startswith("cuda") and not torch_cuda_available():
        warnings.append(f"CUDA device requested but not available: {config.device}")
    
    return warnings


def torch_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def print_config_summary(config: APIConfig):
    """Print a summary of the current configuration."""
    print("🔧 Adaptrix API Configuration Summary")
    print("=" * 50)
    print(f"Server: {config.host}:{config.port}")
    print(f"Model: {config.model_id}")
    print(f"Device: {config.device}")
    print(f"Optimized Engine: {config.use_optimized_engine}")
    print(f"vLLM: {config.use_vllm}")
    print(f"Quantization: {config.enable_quantization}")
    print(f"Caching: {config.enable_caching}")
    print(f"Auto Selection: {config.enable_auto_selection}")
    print(f"RAG: {config.enable_rag}")
    print(f"Rate Limiting: {config.enable_rate_limiting}")
    print(f"Authentication: {config.enable_auth}")
    print(f"Debug: {config.debug}")
    print("=" * 50)
    
    # Print warnings
    warnings = validate_config(config)
    if warnings:
        print("⚠️ Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("=" * 50)
