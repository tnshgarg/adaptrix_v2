"""
Progress display utilities for the Adaptrix CLI.

This module provides classes and functions for displaying progress
bars and other progress indicators in the CLI.
"""

import os
import time
import requests
from typing import Optional, Callable, Any
from pathlib import Path
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

class ProgressBar:
    """
    Rich progress bar for CLI operations.
    """
    
    def __init__(self, description: str = "Progress", total: int = 100):
        """
        Initialize progress bar.
        
        Args:
            description: Description of the progress bar
            total: Total number of steps
        """
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
            expand=True
        )
        self.task_id = None
        self.description = description
        self.total = total
    
    def __enter__(self):
        """Start progress bar context."""
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End progress bar context."""
        self.progress.stop()
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """
        Update progress bar.
        
        Args:
            advance: Number of steps to advance
            description: New description (optional)
        """
        if description:
            self.progress.update(self.task_id, description=description, advance=advance)
        else:
            self.progress.update(self.task_id, advance=advance)
    
    def set_total(self, total: int):
        """
        Set total number of steps.
        
        Args:
            total: New total
        """
        self.progress.update(self.task_id, total=total)
    
    def set_description(self, description: str):
        """
        Set progress bar description.
        
        Args:
            description: New description
        """
        self.progress.update(self.task_id, description=description)

def download_with_progress(url: str, output_path: str, description: Optional[str] = None) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the file
        description: Progress bar description
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up progress bar
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            expand=True
        )
        
        # Start download
        with progress:
            # Get file size
            response = requests.head(url, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Create task
            task_id = progress.add_task(
                description or f"Downloading {os.path.basename(output_path)}",
                total=total_size
            )
            
            # Download file
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))
        
        return True
    
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Download failed: {e}")
        return False

def operation_with_spinner(operation: Callable, description: str, *args, **kwargs) -> Any:
    """
    Run an operation with a spinner.
    
    Args:
        operation: Function to run
        description: Operation description
        *args: Arguments to pass to the operation
        **kwargs: Keyword arguments to pass to the operation
    
    Returns:
        Result of the operation
    """
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(pulse_style="blue"),
        expand=True
    ) as progress:
        task_id = progress.add_task(description, total=None)
        result = operation(*args, **kwargs)
        progress.update(task_id, completed=True)
        return result
