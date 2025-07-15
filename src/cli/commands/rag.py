"""
RAG management commands for the Adaptrix CLI.

This module provides commands for managing RAG document collections and vector stores.
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
from src.cli.utils.formatting import format_table
from src.cli.utils.validation import validate_path
# Import RAG manager with error handling
try:
    from src.cli.core.rag_manager import RAGManager
    RAG_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("RAGManager not available")
    RAG_MANAGER_AVAILABLE = False

    # Mock implementation
    class MockRAGManager:
        def __init__(self, *args, **kwargs):
            pass

        def list_collections(self):
            return []

        def list_documents(self, collection_name):
            return []

        def add_documents(self, document_path, collection_name, recursive):
            return 0

        def create_vector_store(self, collection_name, **kwargs):
            return False

        def get_collection_info(self, collection_name):
            return None

        def remove_document(self, collection_name, document_name):
            return False

        def remove_collection(self, collection_name):
            return False

        def search_documents(self, query, collection_name, top_k):
            return []

    RAGManager = MockRAGManager

logger = get_logger("commands.rag")
console = Console()

@click.group(name="rag")
def rag_group():
    """Manage RAG document collections and vector stores."""
    pass

@rag_group.command(name="add")
@click.argument("documents", nargs=-1, required=True)
@click.option("--collection", "-c", default="default", help="Collection name")
@click.option("--recursive", "-r", is_flag=True, help="Process directories recursively")
@click.pass_context
def add_documents(ctx, documents, collection, recursive):
    """Add documents to a RAG collection."""
    log_command("rag add", {"documents": documents, "collection": collection, "recursive": recursive})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        # Validate document paths
        valid_paths = []
        for doc_path in documents:
            if validate_path(doc_path, must_exist=True):
                valid_paths.append(doc_path)
            else:
                console.print(f"[bold yellow]Warning:[/bold yellow] Path does not exist: {doc_path}")
        
        if not valid_paths:
            console.print("[bold red]Error:[/bold red] No valid document paths provided.")
            sys.exit(1)
        
        # Add documents
        console.print(f"[bold blue]Adding documents to collection '{collection}'...[/bold blue]")
        
        total_added = 0
        for doc_path in valid_paths:
            added = rag_manager.add_documents(doc_path, collection, recursive)
            total_added += added
            console.print(f"Added {added} documents from {doc_path}")
        
        console.print(f"[bold green]Successfully added {total_added} documents to collection '{collection}'[/bold green]")
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@rag_group.command(name="list")
@click.option("--collection", "-c", help="Show documents in specific collection")
@click.option("--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table", help="Output format")
@click.pass_context
def list_collections(ctx, collection, format):
    """List RAG collections and documents."""
    log_command("rag list", {"collection": collection, "format": format})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        if collection:
            # List documents in specific collection
            documents = rag_manager.list_documents(collection)
            
            if format == "table":
                table = format_table(
                    documents,
                    columns=["filename", "chunks", "size", "added_date"],
                    title=f"Documents in Collection '{collection}'"
                )
                console.print(table)
            elif format == "json":
                console.print_json(data=documents)
            elif format == "yaml":
                import yaml
                console.print(yaml.dump(documents, default_flow_style=False))
        else:
            # List all collections
            collections = rag_manager.list_collections()
            
            if format == "table":
                table = format_table(
                    collections,
                    columns=["name", "documents", "chunks", "size"],
                    title="RAG Collections"
                )
                console.print(table)
            elif format == "json":
                console.print_json(data=collections)
            elif format == "yaml":
                import yaml
                console.print(yaml.dump(collections, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@rag_group.command(name="create-store")
@click.argument("collection_name")
@click.option("--embedding-model", help="Embedding model to use")
@click.option("--chunk-size", type=int, help="Chunk size for document processing")
@click.pass_context
def create_store(ctx, collection_name, embedding_model, chunk_size):
    """Create a new vector store."""
    log_command("rag create-store", {
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size
    })
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        # Create vector store
        console.print(f"[bold blue]Creating vector store '{collection_name}'...[/bold blue]")
        
        success = rag_manager.create_vector_store(
            collection_name,
            embedding_model=embedding_model,
            chunk_size=chunk_size
        )
        
        if success:
            console.print(f"[bold green]Successfully created vector store '{collection_name}'[/bold green]")
        else:
            console.print(f"[bold red]Failed to create vector store '{collection_name}'[/bold red]")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@rag_group.command(name="info")
@click.argument("collection_name")
@click.pass_context
def collection_info(ctx, collection_name):
    """Show information about a RAG collection."""
    log_command("rag info", {"collection_name": collection_name})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        # Get collection info
        info = rag_manager.get_collection_info(collection_name)
        
        if not info:
            console.print(f"[bold red]Error:[/bold red] Collection '{collection_name}' not found.")
            sys.exit(1)
        
        # Display info
        console.print(f"[bold blue]Collection: {collection_name}[/bold blue]")
        console.print(f"Documents: {info.get('documents', 0)}")
        console.print(f"Chunks: {info.get('chunks', 0)}")
        console.print(f"Size: {info.get('size', 'Unknown')}")
        console.print(f"Embedding Model: {info.get('embedding_model', 'Unknown')}")
        console.print(f"Created: {info.get('created_date', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@rag_group.command(name="remove")
@click.argument("collection_name")
@click.option("--document", help="Remove specific document from collection")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def remove_documents(ctx, collection_name, document, yes):
    """Remove documents from a RAG collection."""
    log_command("rag remove", {"collection_name": collection_name, "document": document, "yes": yes})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        if document:
            # Remove specific document
            if not yes:
                if not click.confirm(f"Remove document '{document}' from collection '{collection_name}'?"):
                    console.print("Operation cancelled.")
                    return
            
            success = rag_manager.remove_document(collection_name, document)
            
            if success:
                console.print(f"[bold green]Successfully removed document '{document}' from collection '{collection_name}'[/bold green]")
            else:
                console.print(f"[bold red]Failed to remove document '{document}'[/bold red]")
                sys.exit(1)
        else:
            # Remove entire collection
            if not yes:
                if not click.confirm(f"Remove entire collection '{collection_name}'?"):
                    console.print("Operation cancelled.")
                    return
            
            success = rag_manager.remove_collection(collection_name)
            
            if success:
                console.print(f"[bold green]Successfully removed collection '{collection_name}'[/bold green]")
            else:
                console.print(f"[bold red]Failed to remove collection '{collection_name}'[/bold red]")
                sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error removing documents: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

@rag_group.command(name="search")
@click.argument("query")
@click.option("--collection", "-c", default="default", help="Collection to search")
@click.option("--top-k", "-k", type=int, default=5, help="Number of results to return")
@click.pass_context
def search_documents(ctx, query, collection, top_k):
    """Search documents in a RAG collection."""
    log_command("rag search", {"query": query, "collection": collection, "top_k": top_k})
    
    try:
        # Get configuration
        config = ctx.obj["config"]
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        # Search documents
        results = rag_manager.search_documents(query, collection, top_k)
        
        if not results:
            console.print(f"[bold yellow]No results found for query: {query}[/bold yellow]")
            return
        
        # Display results
        console.print(f"[bold blue]Search Results for: {query}[/bold blue]\n")
        
        for i, result in enumerate(results, 1):
            console.print(f"[bold]{i}. Score: {result.get('score', 0):.3f}[/bold]")
            console.print(f"Document: {result.get('document', 'Unknown')}")
            console.print(f"Content: {result.get('content', '')[:200]}...")
            console.print()
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
