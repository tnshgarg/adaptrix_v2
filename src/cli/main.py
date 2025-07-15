#!/usr/bin/env python3
"""
Main entry point for the Adaptrix CLI.

This module defines the main CLI group and imports all command subgroups.
"""

import os
import sys
import logging
from pathlib import Path
import click
from rich.console import Console
from rich.logging import RichHandler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import CLI utilities
from .utils.logging import setup_logging
from .core.config_manager import ConfigManager

# Set up logging
console = Console()
setup_logging()
logger = logging.getLogger("adaptrix.cli")

# Initialize configuration
config_manager = ConfigManager()

@click.group()
@click.version_option(version="1.0.0", prog_name="adaptrix")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.pass_context
def cli(ctx, verbose, quiet, config):
    """
    Adaptrix CLI - Command Line Interface for the Adaptrix Modular AI System.
    
    Run models, manage adapters, integrate RAG, and build custom AI solutions.
    """
    # Set up context object for sharing data between commands
    ctx.ensure_object(dict)
    
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger("adaptrix").setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger("adaptrix").setLevel(logging.ERROR)
    else:
        logging.getLogger("adaptrix").setLevel(logging.INFO)
    
    # Load configuration
    if config:
        config_manager.load_config(config)
    else:
        config_manager.load_default_config()
    
    # Store in context
    ctx.obj["config"] = config_manager
    ctx.obj["console"] = console
    
    logger.debug("Adaptrix CLI initialized")

# Import command groups with error handling
try:
    from .commands.models import models_group
    MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Models commands not available")
    MODELS_AVAILABLE = False
    models_group = None

try:
    from .commands.adapters import adapters_group
    ADAPTERS_AVAILABLE = True
except ImportError:
    logger.warning("Adapters commands not available")
    ADAPTERS_AVAILABLE = False
    adapters_group = None

try:
    from .commands.rag import rag_group
    RAG_AVAILABLE = True
except ImportError:
    logger.warning("RAG commands not available")
    RAG_AVAILABLE = False
    rag_group = None

try:
    from .commands.build import build_group
    BUILD_AVAILABLE = True
except ImportError:
    logger.warning("Build commands not available")
    BUILD_AVAILABLE = False
    build_group = None

try:
    from .commands.inference import run_command, chat_command
    INFERENCE_AVAILABLE = True
except ImportError:
    logger.warning("Inference commands not available")
    INFERENCE_AVAILABLE = False
    run_command = None
    chat_command = None

try:
    from .commands.config import config_group
    CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("Config commands not available")
    CONFIG_AVAILABLE = False
    config_group = None

# Register command groups if available
if MODELS_AVAILABLE and models_group:
    cli.add_command(models_group)
if ADAPTERS_AVAILABLE and adapters_group:
    cli.add_command(adapters_group)
if RAG_AVAILABLE and rag_group:
    cli.add_command(rag_group)
if BUILD_AVAILABLE and build_group:
    cli.add_command(build_group)
if INFERENCE_AVAILABLE:
    if run_command:
        cli.add_command(run_command)
    if chat_command:
        cli.add_command(chat_command)
if CONFIG_AVAILABLE and config_group:
    cli.add_command(config_group)

if __name__ == "__main__":
    cli()
