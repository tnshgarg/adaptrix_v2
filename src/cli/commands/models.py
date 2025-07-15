"""
Model management commands for the Adaptrix CLI.

This module provides commands for listing, downloading, and running models.
"""

import os
import sys
import click
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.utils.logging import get_logger, log_command
from src.cli.utils.formatting import format_table, format_model_info
from src.cli.utils.validation import validate_model_name, validate_model_size
# Import model manager with error handling
try:
    from src.cli.core.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("ModelManager not available")
    MODEL_MANAGER_AVAILABLE = False

    # Mock implementation
    class MockModelManager:
        def __init__(self, *args, **kwargs):
            pass

        def list_available_models(self):
            return [
                {"name": "qwen/qwen3-1.7b", "parameters": "1.7B", "architecture": "qwen", "downloaded": False},
                {"name": "microsoft/phi-2", "parameters": "2.7B", "architecture": "phi", "downloaded": False}
            ]

        def list_downloaded_models(self):
            return []

        def get_model_info(self, model_name):
            return {"name": model_name, "description": "Mock model info", "downloaded": False}

        def is_model_downloaded(self, model_name):
            return False

        def download_model(self, model_name):
            return False

        def run_model(self, model_name, prompt, **kwargs):
            return f"Mock response for: {prompt}"

    ModelManager = MockModelManager

logger = get_logger("commands.models")
console = Console()

@click.group(name="models")
def models_group():
    """Manage models for Adaptrix."""
    pass

@models_group.command(name="list")
@click.option("--available", "-a", is_flag=True, help="Show all available models, not just downloaded ones")
@click.option("--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table", help="Output format")
@click.pass_context
def list_models(ctx, available, format):
    """List available models."""
    log_command("models list", {"available": available, "format": format})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Get models
        if available:
            models = model_manager.list_available_models()
        else:
            models = model_manager.list_downloaded_models()
        
        # Format output
        if format == "table":
            table = format_table(
                models,
                columns=["name", "parameters", "architecture", "downloaded"],
                title="Available Models" if available else "Downloaded Models"
            )
            console.print(table)
        elif format == "json":
            console.print_json(data=models)
        elif format == "yaml":
            import yaml
            console.print(yaml.dump(models, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@models_group.command(name="download")
@click.argument("model_name")
@click.option("--force", "-f", is_flag=True, help="Force download even if model already exists")
@click.pass_context
def download_model(ctx, model_name, force):
    """Download a model."""
    log_command("models download", {"model_name": model_name, "force": force})
    
    try:
        # Validate model name
        if not validate_model_name(model_name):
            console.print(f"[bold red]Error:[/bold red] Invalid model name: {model_name}")
            sys.exit(1)
        
        # Validate model size
        if not validate_model_size(model_name):
            console.print(f"[bold red]Error:[/bold red] Model {model_name} exceeds the 3B parameter limit.")
            sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Check if model is already downloaded
        if model_manager.is_model_downloaded(model_name) and not force:
            console.print(f"[bold yellow]Model {model_name} is already downloaded.[/bold yellow]")
            console.print("Use --force to download again.")
            return
        
        # Download model
        console.print(f"[bold blue]Downloading model {model_name}...[/bold blue]")
        success = model_manager.download_model(model_name)
        
        if success:
            console.print(f"[bold green]Successfully downloaded model {model_name}[/bold green]")
        else:
            console.print(f"[bold red]Failed to download model {model_name}[/bold red]")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@models_group.command(name="info")
@click.argument("model_name")
@click.pass_context
def model_info(ctx, model_name):
    """Show information about a model."""
    log_command("models info", {"model_name": model_name})
    
    try:
        # Validate model name
        if not validate_model_name(model_name):
            console.print(f"[bold red]Error:[/bold red] Invalid model name: {model_name}")
            sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Get model info
        model_info = model_manager.get_model_info(model_name)
        
        if not model_info:
            console.print(f"[bold red]Error:[/bold red] Model {model_name} not found.")
            sys.exit(1)
        
        # Format and display model info
        table = format_model_info(model_info)
        console.print(table)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@models_group.command(name="run")
@click.argument("model_name")
@click.option("--prompt", "-p", help="Prompt to send to the model")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--max-tokens", "-m", type=int, help="Maximum number of tokens to generate")
@click.option("--temperature", "-t", type=float, help="Temperature for sampling")
@click.pass_context
def run_model(ctx, model_name, prompt, interactive, max_tokens, temperature):
    """Run a model without adapters."""
    log_command("models run", {
        "model_name": model_name,
        "prompt": prompt,
        "interactive": interactive,
        "max_tokens": max_tokens,
        "temperature": temperature
    })
    
    try:
        # Validate model name
        if not validate_model_name(model_name):
            console.print(f"[bold red]Error:[/bold red] Invalid model name: {model_name}")
            sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Check if model is downloaded
        if not model_manager.is_model_downloaded(model_name):
            console.print(f"[bold yellow]Model {model_name} is not downloaded.[/bold yellow]")
            
            # Ask to download
            if click.confirm("Do you want to download it now?"):
                success = model_manager.download_model(model_name)
                if not success:
                    console.print(f"[bold red]Failed to download model {model_name}[/bold red]")
                    sys.exit(1)
            else:
                sys.exit(1)
        
        # Set up generation parameters
        gen_params = {}
        if max_tokens:
            gen_params["max_tokens"] = max_tokens
        if temperature:
            gen_params["temperature"] = temperature
        
        # Run model
        if interactive:
            console.print(f"[bold blue]Running {model_name} in interactive mode. Type 'exit' to quit.[/bold blue]")
            
            while True:
                # Get user input
                user_input = click.prompt("\n[bold green]You[/bold green]")
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                # Generate response
                response = model_manager.run_model(model_name, user_input, **gen_params)
                
                # Print response
                console.print(f"\n[bold purple]Model[/bold purple]: {response}")
        
        elif prompt:
            # Generate response
            response = model_manager.run_model(model_name, prompt, **gen_params)
            
            # Print response
            console.print(f"\n[bold purple]Model[/bold purple]: {response}")
        
        else:
            console.print("[bold yellow]Please provide a prompt with --prompt or use --interactive mode.[/bold yellow]")
        
    except Exception as e:
        logger.error(f"Error running model: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
