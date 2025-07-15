"""
Custom model building commands for the Adaptrix CLI.

This module provides commands for creating and managing custom model configurations.
"""

import os
import sys
import click
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.utils.logging import get_logger, log_command
from src.cli.utils.formatting import format_table
from src.cli.utils.validation import validate_model_name, validate_adapter_name

logger = get_logger("commands.build")
console = Console()

@click.group(name="build")
def build_group():
    """Build and manage custom model configurations."""
    pass

@build_group.command(name="create")
@click.argument("config_name")
@click.option("--model", "-m", required=True, help="Base model to use")
@click.option("--adapters", "-a", multiple=True, help="Adapters to include (can be used multiple times)")
@click.option("--rag-collection", "-r", help="RAG collection to include")
@click.option("--description", help="Description of the custom model")
@click.option("--strategy", type=click.Choice(["sequential", "parallel", "hierarchical", "conditional"]), 
              default="sequential", help="Adapter composition strategy")
@click.pass_context
def create_config(ctx, config_name, model, adapters, rag_collection, description, strategy):
    """Create a new custom model configuration."""
    log_command("build create", {
        "config_name": config_name,
        "model": model,
        "adapters": adapters,
        "rag_collection": rag_collection,
        "description": description,
        "strategy": strategy
    })
    
    try:
        # Validate inputs
        if not validate_model_name(model):
            console.print(f"[bold red]Error:[/bold red] Invalid model name: {model}")
            sys.exit(1)
        
        for adapter in adapters:
            if not validate_adapter_name(adapter):
                console.print(f"[bold red]Error:[/bold red] Invalid adapter name: {adapter}")
                sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Create build directory
        build_dir = Path(config.get("build.output_directory", "~/.adaptrix/builds")).expanduser()
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Create custom model configuration
        custom_config = {
            "name": config_name,
            "description": description or f"Custom model configuration: {config_name}",
            "created_date": str(Path().cwd()),
            "base_model": model,
            "adapters": list(adapters),
            "rag_collection": rag_collection,
            "composition_strategy": strategy,
            "inference_config": {
                "device": config.get("inference.device", "auto"),
                "max_tokens": config.get("inference.max_tokens", 1024),
                "temperature": config.get("inference.temperature", 0.7),
                "top_p": config.get("inference.top_p", 0.9),
                "top_k": config.get("inference.top_k", 50)
            }
        }
        
        # Save configuration
        config_file = build_dir / f"{config_name}.yaml"
        
        if config_file.exists():
            if not click.confirm(f"Configuration '{config_name}' already exists. Overwrite?"):
                console.print("Operation cancelled.")
                return
        
        with open(config_file, 'w') as f:
            yaml.dump(custom_config, f, default_flow_style=False)
        
        console.print(f"[bold green]Successfully created custom model configuration '{config_name}'[/bold green]")
        console.print(f"Configuration saved to: {config_file}")
        
        # Display configuration summary
        console.print(f"\n[bold blue]Configuration Summary:[/bold blue]")
        console.print(f"Base Model: {model}")
        console.print(f"Adapters: {', '.join(adapters) if adapters else 'None'}")
        console.print(f"RAG Collection: {rag_collection or 'None'}")
        console.print(f"Composition Strategy: {strategy}")
        
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@build_group.command(name="list")
@click.option("--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table", help="Output format")
@click.pass_context
def list_configs(ctx, format):
    """List custom model configurations."""
    log_command("build list", {"format": format})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Get build directory
        build_dir = Path(config.get("build.output_directory", "~/.adaptrix/builds")).expanduser()
        
        if not build_dir.exists():
            console.print("[bold yellow]No custom configurations found.[/bold yellow]")
            return
        
        # Load configurations
        configs = []
        for config_file in build_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                config_info = {
                    "name": config_data.get("name", config_file.stem),
                    "description": config_data.get("description", ""),
                    "base_model": config_data.get("base_model", ""),
                    "adapters": len(config_data.get("adapters", [])),
                    "rag_collection": config_data.get("rag_collection", "None"),
                    "strategy": config_data.get("composition_strategy", ""),
                    "created": config_data.get("created_date", "")
                }
                configs.append(config_info)
                
            except Exception as e:
                logger.warning(f"Error loading configuration {config_file}: {e}")
                continue
        
        if not configs:
            console.print("[bold yellow]No valid configurations found.[/bold yellow]")
            return
        
        # Format output
        if format == "table":
            table = format_table(
                configs,
                columns=["name", "base_model", "adapters", "rag_collection", "strategy"],
                title="Custom Model Configurations"
            )
            console.print(table)
        elif format == "json":
            console.print_json(data=configs)
        elif format == "yaml":
            console.print(yaml.dump(configs, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Error listing configurations: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@build_group.command(name="info")
@click.argument("config_name")
@click.pass_context
def config_info(ctx, config_name):
    """Show information about a custom model configuration."""
    log_command("build info", {"config_name": config_name})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Get build directory
        build_dir = Path(config.get("build.output_directory", "~/.adaptrix/builds")).expanduser()
        config_file = build_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            console.print(f"[bold red]Error:[/bold red] Configuration '{config_name}' not found.")
            sys.exit(1)
        
        # Load configuration
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Display configuration
        console.print(f"[bold blue]Configuration: {config_name}[/bold blue]\n")
        
        console.print(f"[bold]Description:[/bold] {config_data.get('description', 'N/A')}")
        console.print(f"[bold]Base Model:[/bold] {config_data.get('base_model', 'N/A')}")
        console.print(f"[bold]Composition Strategy:[/bold] {config_data.get('composition_strategy', 'N/A')}")
        console.print(f"[bold]RAG Collection:[/bold] {config_data.get('rag_collection', 'None')}")
        console.print(f"[bold]Created:[/bold] {config_data.get('created_date', 'N/A')}")
        
        # Display adapters
        adapters = config_data.get("adapters", [])
        if adapters:
            console.print(f"\n[bold]Adapters ({len(adapters)}):[/bold]")
            for i, adapter in enumerate(adapters, 1):
                console.print(f"  {i}. {adapter}")
        else:
            console.print(f"\n[bold]Adapters:[/bold] None")
        
        # Display inference configuration
        inference_config = config_data.get("inference_config", {})
        if inference_config:
            console.print(f"\n[bold]Inference Configuration:[/bold]")
            for key, value in inference_config.items():
                console.print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error getting configuration info: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@build_group.command(name="export")
@click.argument("config_name")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "-f", type=click.Choice(["yaml", "json"]), default="yaml", help="Export format")
@click.pass_context
def export_config(ctx, config_name, output, format):
    """Export a custom model configuration."""
    log_command("build export", {"config_name": config_name, "output": output, "format": format})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Get build directory
        build_dir = Path(config.get("build.output_directory", "~/.adaptrix/builds")).expanduser()
        config_file = build_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            console.print(f"[bold red]Error:[/bold red] Configuration '{config_name}' not found.")
            sys.exit(1)
        
        # Load configuration
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Determine output path
        if not output:
            output = f"{config_name}.{format}"
        
        output_path = Path(output)
        
        # Export configuration
        with open(output_path, 'w') as f:
            if format == "yaml":
                yaml.dump(config_data, f, default_flow_style=False)
            elif format == "json":
                json.dump(config_data, f, indent=2)
        
        console.print(f"[bold green]Successfully exported configuration to {output_path}[/bold green]")
        
    except Exception as e:
        logger.error(f"Error exporting configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@build_group.command(name="delete")
@click.argument("config_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_config(ctx, config_name, yes):
    """Delete a custom model configuration."""
    log_command("build delete", {"config_name": config_name, "yes": yes})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Get build directory
        build_dir = Path(config.get("build.output_directory", "~/.adaptrix/builds")).expanduser()
        config_file = build_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            console.print(f"[bold red]Error:[/bold red] Configuration '{config_name}' not found.")
            sys.exit(1)
        
        # Confirm deletion
        if not yes:
            if not click.confirm(f"Delete configuration '{config_name}'?"):
                console.print("Operation cancelled.")
                return
        
        # Delete configuration
        config_file.unlink()
        
        console.print(f"[bold green]Successfully deleted configuration '{config_name}'[/bold green]")
        
    except Exception as e:
        logger.error(f"Error deleting configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
