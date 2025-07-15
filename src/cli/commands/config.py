"""
Configuration management commands for the Adaptrix CLI.

This module provides commands for managing CLI configuration settings.
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
from src.cli.utils.formatting import format_table, format_yaml
from src.cli.utils.validation import validate_config_value

logger = get_logger("commands.config")
console = Console()

@click.group(name="config")
def config_group():
    """Manage CLI configuration settings."""
    pass

@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
@click.option("--global", "-g", "is_global", is_flag=True, help="Set global configuration")
@click.pass_context
def set_config(ctx, key, value, is_global):
    """Set a configuration value."""
    log_command("config set", {"key": key, "value": value, "global": is_global})
    
    try:
        # Validate configuration value
        if not validate_config_value(key, value):
            console.print(f"[bold red]Error:[/bold red] Invalid value for {key}: {value}")
            sys.exit(1)
        
        # Get configuration manager
        config_manager = ctx.obj["config"]
        
        # Set configuration value
        config_manager.set(key, value)
        
        # Save configuration
        if is_global:
            config_manager.save_config()
        else:
            # Save to project configuration
            project_config_dir = Path.cwd() / ".adaptrix"
            project_config_dir.mkdir(exist_ok=True)
            config_manager.save_config(project_config_dir / "config.yaml")
        
        console.print(f"[bold green]Successfully set {key} = {value}[/bold green]")
        
    except Exception as e:
        logger.error(f"Error setting configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@config_group.command(name="get")
@click.argument("key", required=False)
@click.option("--format", "-f", type=click.Choice(["table", "yaml", "json"]), default="table", help="Output format")
@click.pass_context
def get_config(ctx, key, format):
    """Get configuration value(s)."""
    log_command("config get", {"key": key, "format": format})
    
    try:
        # Get configuration manager
        config_manager = ctx.obj["config"]
        
        if key:
            # Get specific configuration value
            value = config_manager.get(key)
            
            if value is None:
                console.print(f"[bold yellow]Configuration key '{key}' not found.[/bold yellow]")
                return
            
            console.print(f"[bold blue]{key}:[/bold blue] {value}")
        else:
            # Get all configuration values
            all_config = config_manager.get_all()
            
            if format == "table":
                # Flatten configuration for table display
                flattened = []
                
                def flatten_dict(d, prefix=""):
                    for k, v in d.items():
                        full_key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            flatten_dict(v, full_key)
                        else:
                            flattened.append({"key": full_key, "value": str(v)})
                
                flatten_dict(all_config)
                
                table = format_table(
                    flattened,
                    columns=["key", "value"],
                    title="Configuration Settings"
                )
                console.print(table)
            
            elif format == "yaml":
                console.print(format_yaml(all_config))
            
            elif format == "json":
                console.print_json(data=all_config)
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@config_group.command(name="list")
@click.option("--section", "-s", help="Show only specific section")
@click.pass_context
def list_config(ctx, section):
    """List all configuration settings."""
    log_command("config list", {"section": section})
    
    try:
        # Get configuration manager
        config_manager = ctx.obj["config"]
        
        # Get configuration
        if section:
            config_data = config_manager.get(section, {})
            if not config_data:
                console.print(f"[bold yellow]Configuration section '{section}' not found.[/bold yellow]")
                return
        else:
            config_data = config_manager.get_all()
        
        # Display configuration
        console.print(format_yaml(config_data, colorize=True))
        
    except Exception as e:
        logger.error(f"Error listing configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@config_group.command(name="reset")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def reset_config(ctx, yes):
    """Reset configuration to defaults."""
    log_command("config reset", {"yes": yes})
    
    try:
        # Confirm reset
        if not yes:
            if not click.confirm("Reset configuration to defaults? This will remove all custom settings."):
                console.print("Operation cancelled.")
                return
        
        # Get configuration manager
        config_manager = ctx.obj["config"]
        
        # Reset configuration
        config_manager.reset()
        
        # Save default configuration
        config_manager.save_config()
        
        console.print("[bold green]Configuration reset to defaults.[/bold green]")
        
    except Exception as e:
        logger.error(f"Error resetting configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@config_group.command(name="validate")
@click.pass_context
def validate_config(ctx):
    """Validate current configuration."""
    log_command("config validate", {})
    
    try:
        # Get configuration manager
        config_manager = ctx.obj["config"]
        
        # Get all configuration
        config_data = config_manager.get_all()
        
        # Validate configuration
        errors = []
        warnings = []
        
        # Check required directories
        required_dirs = [
            "models.directory",
            "adapters.directory", 
            "rag.directory",
            "logging.directory"
        ]
        
        for dir_key in required_dirs:
            dir_path = config_manager.get(dir_key)
            if dir_path:
                path_obj = Path(dir_path)
                if not path_obj.exists():
                    warnings.append(f"Directory does not exist: {dir_key} = {dir_path}")
                elif not path_obj.is_dir():
                    errors.append(f"Path is not a directory: {dir_key} = {dir_path}")
            else:
                errors.append(f"Required directory not configured: {dir_key}")
        
        # Check device setting
        device = config_manager.get("inference.device")
        if device not in ["auto", "cpu", "cuda", "mps"]:
            errors.append(f"Invalid device setting: {device}")
        
        # Check log level
        log_level = config_manager.get("logging.level")
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            errors.append(f"Invalid log level: {log_level}")
        
        # Check numeric values
        numeric_checks = [
            ("models.max_size_gb", int),
            ("rag.chunk_size", int),
            ("rag.chunk_overlap", int),
            ("inference.max_tokens", int),
            ("inference.temperature", float),
            ("inference.top_p", float),
            ("inference.top_k", int)
        ]
        
        for key, expected_type in numeric_checks:
            value = config_manager.get(key)
            if value is not None:
                try:
                    expected_type(value)
                except (ValueError, TypeError):
                    errors.append(f"Invalid {expected_type.__name__} value for {key}: {value}")
        
        # Display results
        if errors:
            console.print("[bold red]Configuration Errors:[/bold red]")
            for error in errors:
                console.print(f"  • {error}")
        
        if warnings:
            console.print("\n[bold yellow]Configuration Warnings:[/bold yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")
        
        if not errors and not warnings:
            console.print("[bold green]Configuration is valid.[/bold green]")
        elif errors:
            console.print(f"\n[bold red]Found {len(errors)} errors and {len(warnings)} warnings.[/bold red]")
            sys.exit(1)
        else:
            console.print(f"\n[bold yellow]Found {len(warnings)} warnings.[/bold yellow]")
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@config_group.command(name="path")
@click.pass_context
def show_config_path(ctx):
    """Show configuration file paths."""
    log_command("config path", {})
    
    try:
        # Get configuration manager
        config_manager = ctx.obj["config"]
        
        console.print("[bold blue]Configuration File Paths:[/bold blue]")
        console.print(f"Global: {config_manager.global_config_path}")
        console.print(f"Project: {config_manager.project_config_path}")
        
        # Check which files exist
        if config_manager.global_config_path.exists():
            console.print(f"[bold green]✓[/bold green] Global config exists")
        else:
            console.print(f"[bold yellow]✗[/bold yellow] Global config does not exist")
        
        if config_manager.project_config_path.exists():
            console.print(f"[bold green]✓[/bold green] Project config exists")
        else:
            console.print(f"[bold yellow]✗[/bold yellow] Project config does not exist")
        
    except Exception as e:
        logger.error(f"Error showing config paths: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
