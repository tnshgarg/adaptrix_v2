"""
Formatting utilities for the Adaptrix CLI.

This module provides functions for formatting output in various formats
such as tables, JSON, and YAML.
"""

import json
import yaml
from rich.table import Table
from rich.console import Console
from rich.syntax import Syntax

def format_table(data, columns=None, title=None):
    """
    Format data as a rich table.
    
    Args:
        data: List of dictionaries to format
        columns: List of column names to include (default: all keys in first item)
        title: Table title
    
    Returns:
        Rich Table object
    """
    if not data:
        return Table(title=title or "No data")
    
    # Determine columns if not provided
    if not columns:
        columns = list(data[0].keys())
    
    # Create table
    table = Table(title=title)
    
    # Add columns
    for column in columns:
        table.add_column(column.replace("_", " ").title())
    
    # Add rows
    for item in data:
        row = []
        for column in columns:
            value = item.get(column, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            row.append(str(value))
        table.add_row(*row)
    
    return table

def format_json(data, colorize=True):
    """
    Format data as JSON.
    
    Args:
        data: Data to format
        colorize: Whether to colorize the output
    
    Returns:
        Formatted JSON string or Rich Syntax object
    """
    json_str = json.dumps(data, indent=2)
    
    if colorize:
        return Syntax(json_str, "json", theme="monokai", line_numbers=True)
    
    return json_str

def format_yaml(data, colorize=True):
    """
    Format data as YAML.
    
    Args:
        data: Data to format
        colorize: Whether to colorize the output
    
    Returns:
        Formatted YAML string or Rich Syntax object
    """
    yaml_str = yaml.dump(data, default_flow_style=False)
    
    if colorize:
        return Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    
    return yaml_str

def format_model_info(model_info):
    """
    Format model information.
    
    Args:
        model_info: Model information dictionary
    
    Returns:
        Rich Table object
    """
    table = Table(title=f"Model: {model_info.get('name', 'Unknown')}")
    
    table.add_column("Property")
    table.add_column("Value")
    
    for key, value in model_info.items():
        if key == "name":
            continue
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        
        table.add_row(key.replace("_", " ").title(), str(value))
    
    return table

def format_adapter_info(adapter_info):
    """
    Format adapter information.
    
    Args:
        adapter_info: Adapter information dictionary
    
    Returns:
        Rich Table object
    """
    table = Table(title=f"Adapter: {adapter_info.get('name', 'Unknown')}")
    
    table.add_column("Property")
    table.add_column("Value")
    
    for key, value in adapter_info.items():
        if key == "name":
            continue
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        
        table.add_row(key.replace("_", " ").title(), str(value))
    
    return table
