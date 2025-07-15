"""
Logging utilities for the Adaptrix CLI.

This module provides functions for setting up and configuring logging
for the Adaptrix CLI.
"""

import os
import sys
import logging
import time
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

# Default log directory
DEFAULT_LOG_DIR = Path.home() / ".adaptrix" / "logs"

def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Set up logging for the Adaptrix CLI.
    
    Args:
        log_dir: Directory to store log files (default: ~/.adaptrix/logs)
        log_level: Logging level (default: INFO)
    
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file name with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"adaptrix-cli-{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            RichHandler(rich_tracebacks=True, markup=True)
        ]
    )
    
    # Create and configure adaptrix logger
    logger = logging.getLogger("adaptrix")
    logger.setLevel(log_level)
    
    # Log startup information
    logger.info(f"Adaptrix CLI started, logging to {log_file}")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {sys.platform}")
    
    return logger

def get_logger(name):
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"adaptrix.{name}")

def log_command(command_name, args):
    """
    Log a command execution.
    
    Args:
        command_name: Name of the command
        args: Command arguments
    """
    logger = get_logger("commands")
    logger.info(f"Executing command: {command_name}")
    logger.debug(f"Command arguments: {args}")

def log_operation(operation_name, details=None):
    """
    Log an operation.
    
    Args:
        operation_name: Name of the operation
        details: Operation details
    """
    logger = get_logger("operations")
    logger.info(f"Operation: {operation_name}")
    if details:
        logger.debug(f"Operation details: {details}")

def log_error(error_message, exception=None):
    """
    Log an error.
    
    Args:
        error_message: Error message
        exception: Exception object
    """
    logger = get_logger("errors")
    logger.error(error_message)
    if exception:
        logger.exception(exception)
