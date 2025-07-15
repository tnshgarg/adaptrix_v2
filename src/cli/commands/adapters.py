"""
Adapter management commands for the Adaptrix CLI.

This module provides commands for listing, installing, and managing adapters.
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
from src.cli.utils.formatting import format_table, format_adapter_info
from src.cli.utils.validation import validate_adapter_name
# Import adapter manager with error handling
try:
    from src.cli.core.adapter_manager import CLIAdapterManager
    ADAPTER_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("CLIAdapterManager not available")
    ADAPTER_MANAGER_AVAILABLE = False

    # Mock implementation
    class MockCLIAdapterManager:
        def __init__(self, *args, **kwargs):
            pass

        def list_available_adapters(self):
            return [
                {"name": "code_generator", "domain": "programming", "description": "Code generation adapter", "installed": False},
                {"name": "math_solver", "domain": "mathematics", "description": "Math problem solving", "installed": False}
            ]

        def list_installed_adapters(self):
            return []

        def get_adapter_info(self, adapter_name):
            return {"name": adapter_name, "description": "Mock adapter info", "installed": False}

        def is_adapter_installed(self, adapter_name):
            return False

        def install_adapter(self, adapter_name):
            return False

        def install_from_path(self, adapter_name, source_path):
            return False

        def uninstall_adapter(self, adapter_name):
            return False

        def validate_adapter_structure(self, adapter_path):
            return False, ["Mock validation error"]

    CLIAdapterManager = MockCLIAdapterManager

logger = get_logger("commands.adapters")
console = Console()

@click.group(name="adapters")
def adapters_group():
    """Manage adapters for Adaptrix."""
    pass

@adapters_group.command(name="list")
@click.option("--available", "-a", is_flag=True, help="Show all available adapters, not just installed ones")
@click.option("--domain", "-d", help="Filter by domain (code, math, legal, general)")
@click.option("--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table", help="Output format")
@click.pass_context
def list_adapters(ctx, available, domain, format):
    """List available adapters."""
    log_command("adapters list", {"available": available, "domain": domain, "format": format})

    try:
        # Get configuration
        config = ctx.obj["config"]

        # Skip actual adapter manager initialization for now - use default adapters
        # This is a temporary fix until we resolve the hanging issue
        console.print("[bold blue]Loading adapters...[/bold blue]")

        # Use default adapters
        if available:
            adapters = [
                {"name": "code_generator", "domain": "programming", "description": "Code generation and debugging", "version": "1.0.0", "installed": False, "source": "builtin"},
                {"name": "math_solver", "domain": "mathematics", "description": "Mathematical problem solving", "version": "1.0.0", "installed": False, "source": "builtin"},
                {"name": "legal_analyzer", "domain": "legal", "description": "Legal document analysis", "version": "1.0.0", "installed": False, "source": "builtin"},
                {"name": "general_assistant", "domain": "general", "description": "General purpose assistant", "version": "1.0.0", "installed": False, "source": "builtin"}
            ]
        else:
            # Check if adapters directory exists and has subdirectories
            adapters_dir = Path(config.get("adapters.directory"))
            if adapters_dir.exists():
                adapters = []
                for adapter_dir in adapters_dir.iterdir():
                    if adapter_dir.is_dir():
                        # Try to read metadata
                        metadata_path = adapter_dir / "metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                    adapters.append({
                                        "name": metadata.get("name", adapter_dir.name),
                                        "domain": metadata.get("domain", "unknown"),
                                        "description": metadata.get("description", ""),
                                        "version": metadata.get("version", "unknown"),
                                        "installed": True
                                    })
                            except Exception:
                                # If metadata can't be read, use directory name
                                adapters.append({
                                    "name": adapter_dir.name,
                                    "domain": "unknown",
                                    "description": "Local adapter",
                                    "version": "unknown",
                                    "installed": True
                                })
                        else:
                            # No metadata, use directory name
                            adapters.append({
                                "name": adapter_dir.name,
                                "domain": "unknown",
                                "description": "Local adapter",
                                "version": "unknown",
                                "installed": True
                            })
            else:
                adapters = []
        
        # Filter by domain if specified
        if domain:
            adapters = [a for a in adapters if a.get("domain", "").lower() == domain.lower()]
        
        # Format output
        if format == "table":
            table = format_table(
                adapters,
                columns=["name", "domain", "description", "version", "installed"],
                title="Available Adapters" if available else "Installed Adapters"
            )
            console.print(table)
        elif format == "json":
            console.print_json(data=adapters)
        elif format == "yaml":
            import yaml
            console.print(yaml.dump(adapters, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Error listing adapters: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@adapters_group.command(name="install")
@click.argument("adapter_name")
@click.option("--force", "-f", is_flag=True, help="Force install even if adapter already exists")
@click.option("--from-path", help="Install adapter from local path")
@click.pass_context
def install_adapter(ctx, adapter_name, force, from_path):
    """Install an adapter."""
    log_command("adapters install", {"adapter_name": adapter_name, "force": force, "from_path": from_path})
    
    try:
        # Validate adapter name
        if not validate_adapter_name(adapter_name):
            console.print(f"[bold red]Error:[/bold red] Invalid adapter name: {adapter_name}")
            sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize adapter manager
        adapter_manager = CLIAdapterManager(config)
        
        # Check if adapter is already installed
        if adapter_manager.is_adapter_installed(adapter_name) and not force:
            console.print(f"[bold yellow]Adapter {adapter_name} is already installed.[/bold yellow]")
            console.print("Use --force to reinstall.")
            return
        
        # Install adapter
        console.print(f"[bold blue]Installing adapter {adapter_name}...[/bold blue]")
        
        if from_path:
            success = adapter_manager.install_from_path(adapter_name, from_path)
        else:
            success = adapter_manager.install_adapter(adapter_name)
        
        if success:
            console.print(f"[bold green]Successfully installed adapter {adapter_name}[/bold green]")
        else:
            console.print(f"[bold red]Failed to install adapter {adapter_name}[/bold red]")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error installing adapter: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@adapters_group.command(name="info")
@click.argument("adapter_name")
@click.pass_context
def adapter_info(ctx, adapter_name):
    """Show information about an adapter."""
    log_command("adapters info", {"adapter_name": adapter_name})
    
    try:
        # Validate adapter name
        if not validate_adapter_name(adapter_name):
            console.print(f"[bold red]Error:[/bold red] Invalid adapter name: {adapter_name}")
            sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize adapter manager
        adapter_manager = CLIAdapterManager(config)
        
        # Get adapter info
        adapter_info = adapter_manager.get_adapter_info(adapter_name)
        
        if not adapter_info:
            console.print(f"[bold red]Error:[/bold red] Adapter {adapter_name} not found.")
            sys.exit(1)
        
        # Format and display adapter info
        table = format_adapter_info(adapter_info)
        console.print(table)
        
    except Exception as e:
        logger.error(f"Error getting adapter info: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@adapters_group.command(name="uninstall")
@click.argument("adapter_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def uninstall_adapter(ctx, adapter_name, yes):
    """Uninstall an adapter."""
    log_command("adapters uninstall", {"adapter_name": adapter_name, "yes": yes})
    
    try:
        # Validate adapter name
        if not validate_adapter_name(adapter_name):
            console.print(f"[bold red]Error:[/bold red] Invalid adapter name: {adapter_name}")
            sys.exit(1)
        
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize adapter manager
        adapter_manager = CLIAdapterManager(config)
        
        # Check if adapter is installed
        if not adapter_manager.is_adapter_installed(adapter_name):
            console.print(f"[bold yellow]Adapter {adapter_name} is not installed.[/bold yellow]")
            return
        
        # Confirm uninstallation
        if not yes:
            if not click.confirm(f"Are you sure you want to uninstall adapter {adapter_name}?"):
                console.print("Uninstallation cancelled.")
                return
        
        # Uninstall adapter
        console.print(f"[bold blue]Uninstalling adapter {adapter_name}...[/bold blue]")
        success = adapter_manager.uninstall_adapter(adapter_name)
        
        if success:
            console.print(f"[bold green]Successfully uninstalled adapter {adapter_name}[/bold green]")
        else:
            console.print(f"[bold red]Failed to uninstall adapter {adapter_name}[/bold red]")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error uninstalling adapter: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@adapters_group.command(name="create")
@click.argument("adapter_name")
@click.option("--domain", "-d", help="Adapter domain (code, math, legal, general)")
@click.option("--description", help="Adapter description")
@click.option("--template", help="Template to use for adapter creation")
@click.pass_context
def create_adapter(ctx, adapter_name, domain, description, template):
    """Create a new custom adapter (placeholder)."""
    log_command("adapters create", {
        "adapter_name": adapter_name,
        "domain": domain,
        "description": description,
        "template": template
    })
    
    console.print(f"[bold yellow]Adapter creation is not yet implemented.[/bold yellow]")
    console.print("This feature will allow you to create custom LoRA adapters.")
    console.print("For now, you can install existing adapters or use adapters from local paths.")

@adapters_group.command(name="validate")
@click.argument("adapter_path")
@click.pass_context
def validate_adapter(ctx, adapter_path):
    """Validate an adapter structure."""
    log_command("adapters validate", {"adapter_path": adapter_path})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize adapter manager
        adapter_manager = CLIAdapterManager(config)
        
        # Validate adapter
        is_valid, errors = adapter_manager.validate_adapter_structure(adapter_path)
        
        if is_valid:
            console.print(f"[bold green]Adapter at {adapter_path} is valid.[/bold green]")
        else:
            console.print(f"[bold red]Adapter at {adapter_path} is invalid:[/bold red]")
            for error in errors:
                console.print(f"  • {error}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error validating adapter: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
