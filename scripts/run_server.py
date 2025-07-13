#!/usr/bin/env python3
"""
Adaptrix Server Runner.

This script provides an easy way to start the Adaptrix API server with
various configuration options.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import run_server
from src.api.config import get_api_config, print_config_summary, validate_config


def setup_logging(log_level: str = "info", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def check_prerequisites():
    """Check if all prerequisites are met."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required directories
    required_dirs = ["src", "adapters", "models"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            issues.append(f"Required directory missing: {dir_name}")
    
    # Check configuration
    config = get_api_config()
    config_warnings = validate_config(config)
    
    if issues:
        print("❌ Prerequisites not met:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if config_warnings:
        print("⚠️ Configuration warnings:")
        for warning in config_warnings:
            print(f"  - {warning}")
        print()
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Adaptrix API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with default settings
  %(prog)s --host 0.0.0.0 --port 8080  # Custom host and port
  %(prog)s --dev                    # Development mode with reload
  %(prog)s --workers 4              # Multi-worker production mode
  %(prog)s --config                 # Show configuration and exit
        """
    )
    
    # Server options
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from config)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode (enables reload, debug logging)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=None,
        help="Logging level (default: from config)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (default: console only)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration and exit"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites and exit"
    )
    
    # Model options
    parser.add_argument(
        "--model-id",
        help="Override model ID"
    )
    
    parser.add_argument(
        "--device",
        help="Override device (cpu, cuda, auto)"
    )
    
    # Feature flags
    parser.add_argument(
        "--enable-vllm",
        action="store_true",
        help="Enable vLLM optimization"
    )
    
    parser.add_argument(
        "--disable-rag",
        action="store_true",
        help="Disable RAG functionality"
    )
    
    parser.add_argument(
        "--disable-moe",
        action="store_true",
        help="Disable MoE adapter selection"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_api_config()
    
    # Apply command line overrides
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.log_level:
        config.log_level = args.log_level
    if args.model_id:
        config.model_id = args.model_id
    if args.device:
        config.device = args.device
    if args.enable_vllm:
        config.use_vllm = True
    if args.disable_rag:
        config.enable_rag = False
    if args.disable_moe:
        config.enable_auto_selection = False
    
    # Development mode
    if args.dev:
        config.reload = True
        config.log_level = "debug"
        config.debug = True
        args.reload = True
    
    # Setup logging
    setup_logging(
        log_level=config.log_level,
        log_file=args.log_file or config.log_file
    )
    
    # Handle special commands
    if args.config:
        print_config_summary(config)
        return
    
    if args.check:
        if check_prerequisites():
            print("✅ All prerequisites met")
        else:
            sys.exit(1)
        return
    
    # Check prerequisites before starting
    if not check_prerequisites():
        print("\nPlease fix the issues above before starting the server.")
        print("Run 'python scripts/setup.sh' to set up the environment.")
        sys.exit(1)
    
    # Print startup information
    print("🚀 Starting Adaptrix API Server")
    print("=" * 50)
    print(f"Model: {config.model_id}")
    print(f"Device: {config.device}")
    print(f"Host: {config.host}:{config.port}")
    print(f"Workers: {args.workers or 1}")
    print(f"Reload: {args.reload}")
    print(f"vLLM: {config.use_vllm}")
    print(f"RAG: {config.enable_rag}")
    print(f"MoE: {config.enable_auto_selection}")
    print("=" * 50)
    
    # Start server
    try:
        run_server(
            host=config.host,
            port=config.port,
            reload=args.reload,
            workers=args.workers or 1,
            log_level=config.log_level
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
