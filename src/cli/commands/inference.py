"""
Inference commands for the Adaptrix CLI.

This module provides commands for running inference with models and adapters.
"""

import os
import sys
import yaml
import click
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.markdown import Markdown

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.utils.logging import get_logger, log_command
# Import engine manager with error handling
try:
    from src.cli.core.engine_manager import EngineManager
    ENGINE_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("EngineManager not available")
    ENGINE_MANAGER_AVAILABLE = False

    # Mock implementation
    class MockEngineManager:
        def __init__(self, *args, **kwargs):
            pass

        def create_engine(self, model_id, adapters=None, rag_collection=None, composition_strategy="sequential"):
            return MockEngine()

        def get_engine(self, cache_key):
            return MockEngine()

    class MockEngine:
        def generate(self, prompt, **kwargs):
            return f"Mock response for: {prompt}"

    EngineManager = MockEngineManager

logger = get_logger("commands.inference")
console = Console()

@click.command(name="run")
@click.argument("config_name")
@click.option("--prompt", "-p", help="Prompt to send to the model")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--max-tokens", "-m", type=int, help="Maximum number of tokens to generate")
@click.option("--temperature", "-t", type=float, help="Temperature for sampling")
@click.pass_context
def run_command(ctx, config_name, prompt, interactive, max_tokens, temperature):
    """Run inference with a custom model configuration."""
    log_command("run", {
        "config_name": config_name,
        "prompt": prompt,
        "interactive": interactive,
        "max_tokens": max_tokens,
        "temperature": temperature
    })
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize engine manager
        engine_manager = EngineManager(config)
        
        # Load custom model configuration
        build_dir = Path(config.get("build.output_directory", "~/.adaptrix/builds")).expanduser()
        config_file = build_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            console.print(f"[bold red]Error:[/bold red] Configuration '{config_name}' not found.")
            sys.exit(1)
        
        # Load configuration
        with open(config_file, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Initialize engine
        console.print(f"[bold blue]Initializing engine for '{config_name}'...[/bold blue]")
        
        engine = engine_manager.create_engine(
            model_id=model_config["base_model"],
            adapters=model_config.get("adapters", []),
            rag_collection=model_config.get("rag_collection"),
            composition_strategy=model_config.get("composition_strategy", "sequential")
        )
        
        if not engine:
            console.print(f"[bold red]Error:[/bold red] Failed to initialize engine.")
            sys.exit(1)
        
        console.print(f"[bold green]Engine initialized successfully.[/bold green]")
        
        # Set up generation parameters
        gen_params = {}
        if max_tokens:
            gen_params["max_tokens"] = max_tokens
        elif "max_tokens" in model_config.get("inference_config", {}):
            gen_params["max_tokens"] = model_config["inference_config"]["max_tokens"]
        
        if temperature:
            gen_params["temperature"] = temperature
        elif "temperature" in model_config.get("inference_config", {}):
            gen_params["temperature"] = model_config["inference_config"]["temperature"]
        
        # Add other parameters from config
        for param in ["top_p", "top_k", "repetition_penalty"]:
            if param in model_config.get("inference_config", {}):
                gen_params[param] = model_config["inference_config"][param]
        
        # Run inference
        if interactive:
            console.print(f"[bold blue]Running '{config_name}' in interactive mode. Type 'exit' to quit.[/bold blue]")
            console.print(f"[bold blue]Model: {model_config['base_model']}[/bold blue]")
            
            if model_config.get("adapters"):
                console.print(f"[bold blue]Adapters: {', '.join(model_config['adapters'])}[/bold blue]")
            
            if model_config.get("rag_collection"):
                console.print(f"[bold blue]RAG Collection: {model_config['rag_collection']}[/bold blue]")
            
            console.print()
            
            while True:
                # Get user input
                user_input = click.prompt("\n[bold green]You[/bold green]")
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                # Generate response
                response = engine.generate(user_input, **gen_params)
                
                # Print response
                console.print("\n[bold purple]Model[/bold purple]:")
                console.print(Markdown(response))
        
        elif prompt:
            # Generate response
            response = engine.generate(prompt, **gen_params)
            
            # Print response
            console.print("\n[bold purple]Model[/bold purple]:")
            console.print(Markdown(response))
        
        else:
            console.print("[bold yellow]Please provide a prompt with --prompt or use --interactive mode.[/bold yellow]")
        
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@click.command(name="chat")
@click.option("--model", "-m", help="Base model to use")
@click.option("--adapters", "-a", multiple=True, help="Adapters to include (can be used multiple times)")
@click.option("--rag-collection", "-r", help="RAG collection to include")
@click.option("--strategy", type=click.Choice(["sequential", "parallel", "hierarchical", "conditional"]), 
              default="sequential", help="Adapter composition strategy")
@click.pass_context
def chat_command(ctx, model, adapters, rag_collection, strategy):
    """Start an interactive chat session with a model and adapters."""
    log_command("chat", {
        "model": model,
        "adapters": adapters,
        "rag_collection": rag_collection,
        "strategy": strategy
    })
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Use default model if not specified
        if not model:
            model = config.get("models.default_model")
            console.print(f"[bold blue]Using default model: {model}[/bold blue]")
        
        # Initialize engine manager
        engine_manager = EngineManager(config)
        
        # Initialize engine
        console.print(f"[bold blue]Initializing engine...[/bold blue]")
        
        engine = engine_manager.create_engine(
            model_id=model,
            adapters=adapters,
            rag_collection=rag_collection,
            composition_strategy=strategy
        )
        
        if not engine:
            console.print(f"[bold red]Error:[/bold red] Failed to initialize engine.")
            sys.exit(1)
        
        console.print(f"[bold green]Engine initialized successfully.[/bold green]")
        console.print(f"[bold blue]Model: {model}[/bold blue]")
        
        if adapters:
            console.print(f"[bold blue]Adapters: {', '.join(adapters)}[/bold blue]")
        
        if rag_collection:
            console.print(f"[bold blue]RAG Collection: {rag_collection}[/bold blue]")
        
        console.print("\n[bold blue]Starting chat session. Type 'exit' to quit.[/bold blue]\n")
        
        # Set up generation parameters
        gen_params = {
            "max_tokens": config.get("inference.max_tokens", 1024),
            "temperature": config.get("inference.temperature", 0.7),
            "top_p": config.get("inference.top_p", 0.9),
            "top_k": config.get("inference.top_k", 50)
        }
        
        # Chat loop
        chat_history = []
        
        while True:
            # Get user input
            user_input = click.prompt("\n[bold green]You[/bold green]")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Add to history
            chat_history.append({"role": "user", "content": user_input})
            
            # Generate response
            response = engine.generate(user_input, chat_history=chat_history, **gen_params)
            
            # Add to history
            chat_history.append({"role": "assistant", "content": response})
            
            # Print response
            console.print("\n[bold purple]Model[/bold purple]:")
            console.print(Markdown(response))
        
    except Exception as e:
        logger.error(f"Error in chat session: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
