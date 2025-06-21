"""
Command-line interface for Adaptrix system.
"""

import click
import json
import logging
import sys
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from ..core.engine import AdaptrixEngine
from ..utils.config import config

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--json-output', '-j', is_flag=True, help='Output in JSON format')
@click.option('--config', '-c', 'config_file', type=str, help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose, json_output, config_file):
    """Adaptrix: Dynamic LoRA Adapter Injection System"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['json_output'] = json_output
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config_file:
        # Load custom configuration
        try:
            config.config_path = config_file
            config._config = config._load_config()
        except Exception as e:
            console.print(f"[red]Error loading config file: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.argument('adapter_name')
@click.option('--layers', '-l', type=str, help='Comma-separated layer indices (e.g., 6,12,18)')
@click.pass_context
def load(ctx, adapter_name, layers):
    """Load an adapter into the model."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Loading adapter {adapter_name}...", total=None)
            
            # Initialize engine
            engine = AdaptrixEngine()
            if not engine.initialize():
                console.print(f"[red]Failed to initialize Adaptrix engine[/red]")
                return
            
            # Parse layer indices
            layer_indices = None
            if layers:
                try:
                    layer_indices = [int(x.strip()) for x in layers.split(',')]
                except ValueError:
                    console.print(f"[red]Invalid layer indices: {layers}[/red]")
                    return
            
            # Load adapter
            success = engine.load_adapter(adapter_name, layer_indices)
            
            if success:
                if ctx.obj['json_output']:
                    result = {
                        'status': 'success',
                        'adapter': adapter_name,
                        'layers': layer_indices or 'default'
                    }
                    console.print(json.dumps(result, indent=2))
                else:
                    console.print(f"[green]✓ Successfully loaded adapter '{adapter_name}'[/green]")
                    if layer_indices:
                        console.print(f"[blue]Injected into layers: {layer_indices}[/blue]")
            else:
                if ctx.obj['json_output']:
                    result = {'status': 'error', 'message': f'Failed to load adapter {adapter_name}'}
                    console.print(json.dumps(result, indent=2))
                else:
                    console.print(f"[red]✗ Failed to load adapter '{adapter_name}'[/red]")
            
            engine.cleanup()
            
    except Exception as e:
        logger.error(f"Error loading adapter: {e}")
        if ctx.obj['json_output']:
            console.print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('adapter_name')
@click.pass_context
def unload(ctx, adapter_name):
    """Unload an adapter from the model."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Unloading adapter {adapter_name}...", total=None)
            
            engine = AdaptrixEngine()
            if not engine.initialize():
                console.print(f"[red]Failed to initialize Adaptrix engine[/red]")
                return
            
            success = engine.unload_adapter(adapter_name)
            
            if success:
                if ctx.obj['json_output']:
                    result = {'status': 'success', 'adapter': adapter_name}
                    console.print(json.dumps(result, indent=2))
                else:
                    console.print(f"[green]✓ Successfully unloaded adapter '{adapter_name}'[/green]")
            else:
                if ctx.obj['json_output']:
                    result = {'status': 'error', 'message': f'Failed to unload adapter {adapter_name}'}
                    console.print(json.dumps(result, indent=2))
                else:
                    console.print(f"[red]✗ Failed to unload adapter '{adapter_name}'[/red]")
            
            engine.cleanup()
            
    except Exception as e:
        logger.error(f"Error unloading adapter: {e}")
        if ctx.obj['json_output']:
            console.print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.pass_context
def list_adapters(ctx):
    """List all available adapters."""
    try:
        engine = AdaptrixEngine()
        adapters = engine.list_adapters()
        
        if ctx.obj['json_output']:
            result = {'adapters': adapters, 'count': len(adapters)}
            console.print(json.dumps(result, indent=2))
        else:
            if adapters:
                table = Table(title="Available Adapters")
                table.add_column("Name", style="cyan")
                table.add_column("Description", style="green")
                table.add_column("Layers", style="yellow")
                
                for adapter_name in adapters:
                    info = engine.get_adapter_info(adapter_name)
                    if info:
                        metadata = info.get('metadata', {})
                        description = metadata.get('description', 'No description')
                        layers = str(metadata.get('target_layers', []))
                        table.add_row(adapter_name, description, layers)
                    else:
                        table.add_row(adapter_name, "Error loading info", "Unknown")
                
                console.print(table)
            else:
                console.print("[yellow]No adapters found[/yellow]")
        
    except Exception as e:
        logger.error(f"Error listing adapters: {e}")
        if ctx.obj['json_output']:
            console.print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.pass_context
def active(ctx):
    """Show currently active adapters."""
    try:
        engine = AdaptrixEngine()
        if not engine.initialize():
            console.print(f"[red]Failed to initialize Adaptrix engine[/red]")
            return
        
        loaded_adapters = engine.get_loaded_adapters()
        
        if ctx.obj['json_output']:
            result = {'loaded_adapters': loaded_adapters, 'count': len(loaded_adapters)}
            console.print(json.dumps(result, indent=2))
        else:
            if loaded_adapters:
                table = Table(title="Active Adapters")
                table.add_column("Name", style="cyan")
                table.add_column("Status", style="green")
                
                for adapter_name in loaded_adapters:
                    table.add_row(adapter_name, "✓ Loaded")
                
                console.print(table)
            else:
                console.print("[yellow]No adapters currently loaded[/yellow]")
        
        engine.cleanup()
        
    except Exception as e:
        logger.error(f"Error getting active adapters: {e}")
        if ctx.obj['json_output']:
            console.print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('text')
@click.option('--adapter', '-a', type=str, help='Specific adapter to use')
@click.option('--max-length', '-m', type=int, default=100, help='Maximum generation length')
@click.option('--temperature', '-t', type=float, default=0.7, help='Generation temperature')
@click.pass_context
def query(ctx, text, adapter, max_length, temperature):
    """Run inference with the current model configuration."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating response...", total=None)
            
            engine = AdaptrixEngine()
            if not engine.initialize():
                console.print(f"[red]Failed to initialize Adaptrix engine[/red]")
                return
            
            response = engine.query(
                text, 
                adapter_name=adapter,
                max_length=max_length,
                temperature=temperature
            )
            
            if ctx.obj['json_output']:
                result = {
                    'prompt': text,
                    'response': response,
                    'adapter': adapter,
                    'parameters': {
                        'max_length': max_length,
                        'temperature': temperature
                    }
                }
                console.print(json.dumps(result, indent=2))
            else:
                console.print(Panel(
                    f"[bold blue]Prompt:[/bold blue] {text}\n\n"
                    f"[bold green]Response:[/bold green] {response}",
                    title=f"Query Result{f' (using {adapter})' if adapter else ''}",
                    border_style="blue"
                ))
            
            engine.cleanup()
            
    except Exception as e:
        logger.error(f"Error during query: {e}")
        if ctx.obj['json_output']:
            console.print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and information."""
    try:
        engine = AdaptrixEngine()
        if not engine.initialize():
            console.print(f"[red]Failed to initialize Adaptrix engine[/red]")
            return
        
        status_info = engine.get_system_status()
        
        if ctx.obj['json_output']:
            console.print(json.dumps(status_info, indent=2, default=str))
        else:
            # Display formatted status
            console.print(Panel(
                f"[bold green]Adaptrix System Status[/bold green]\n\n"
                f"[blue]Model:[/blue] {status_info.get('model_name', 'Unknown')}\n"
                f"[blue]Device:[/blue] {status_info.get('device', 'Unknown')}\n"
                f"[blue]Initialized:[/blue] {'✓' if status_info.get('initialized') else '✗'}\n"
                f"[blue]Model Loaded:[/blue] {'✓' if status_info.get('model_loaded') else '✗'}\n"
                f"[blue]Active Adapters:[/blue] {len(status_info.get('loaded_adapters', []))}\n"
                f"[blue]Available Adapters:[/blue] {len(status_info.get('available_adapters', []))}",
                title="System Status",
                border_style="green"
            ))
            
            # Memory usage
            memory_info = status_info.get('memory_usage', {})
            if memory_info:
                console.print(Panel(
                    f"[blue]System Memory:[/blue] {memory_info.get('system_memory', {}).get('system_used', 0):.1f}GB used\n"
                    f"[blue]LoRA Memory:[/blue] {memory_info.get('injector_memory', {}).get('memory_gb', 0):.3f}GB\n"
                    f"[blue]Cache Memory:[/blue] {memory_info.get('cache_memory_gb', 0):.3f}GB",
                    title="Memory Usage",
                    border_style="yellow"
                ))
        
        engine.cleanup()
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        if ctx.obj['json_output']:
            console.print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    cli()
