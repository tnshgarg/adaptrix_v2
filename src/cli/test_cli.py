#!/usr/bin/env python3
"""
Test script for the Adaptrix CLI.

This script tests basic CLI functionality to ensure everything is working correctly.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(command):
    """Run a CLI command and return the result."""
    try:
        # Run the command
        result = subprocess.run(
            ["python", "-m", "src.cli.main"] + command.split(),
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def test_basic_commands():
    """Test basic CLI commands."""
    print("Testing basic CLI commands...")
    
    # Test help command
    print("  Testing help command...")
    returncode, stdout, stderr = run_cli_command("--help")
    if returncode == 0:
        print("    ✓ Help command works")
    else:
        print(f"    ✗ Help command failed: {stderr}")
    
    # Test version command
    print("  Testing version command...")
    returncode, stdout, stderr = run_cli_command("--version")
    if returncode == 0:
        print("    ✓ Version command works")
    else:
        print(f"    ✗ Version command failed: {stderr}")
    
    # Test config commands
    print("  Testing config commands...")
    returncode, stdout, stderr = run_cli_command("config list")
    if returncode == 0:
        print("    ✓ Config list works")
    else:
        print(f"    ✗ Config list failed: {stderr}")
    
    # Test models list command
    print("  Testing models list command...")
    returncode, stdout, stderr = run_cli_command("models list --available")
    if returncode == 0:
        print("    ✓ Models list works")
    else:
        print(f"    ✗ Models list failed: {stderr}")
    
    # Test adapters list command
    print("  Testing adapters list command...")
    returncode, stdout, stderr = run_cli_command("adapters list --available")
    if returncode == 0:
        print("    ✓ Adapters list works")
    else:
        print(f"    ✗ Adapters list failed: {stderr}")
    
    # Test rag list command
    print("  Testing rag list command...")
    returncode, stdout, stderr = run_cli_command("rag list")
    if returncode == 0:
        print("    ✓ RAG list works")
    else:
        print(f"    ✗ RAG list failed: {stderr}")
    
    # Test build list command
    print("  Testing build list command...")
    returncode, stdout, stderr = run_cli_command("build list")
    if returncode == 0:
        print("    ✓ Build list works")
    else:
        print(f"    ✗ Build list failed: {stderr}")

def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration management...")
    
    # Test setting a configuration value
    print("  Testing config set...")
    returncode, stdout, stderr = run_cli_command("config set inference.temperature 0.8")
    if returncode == 0:
        print("    ✓ Config set works")
    else:
        print(f"    ✗ Config set failed: {stderr}")
    
    # Test getting a configuration value
    print("  Testing config get...")
    returncode, stdout, stderr = run_cli_command("config get inference.temperature")
    if returncode == 0 and "0.8" in stdout:
        print("    ✓ Config get works")
    else:
        print(f"    ✗ Config get failed: {stderr}")
    
    # Test config validation
    print("  Testing config validate...")
    returncode, stdout, stderr = run_cli_command("config validate")
    if returncode == 0:
        print("    ✓ Config validate works")
    else:
        print(f"    ✗ Config validate failed: {stderr}")

def test_model_info():
    """Test model information commands."""
    print("\nTesting model information...")
    
    # Test model info for a known model
    print("  Testing model info...")
    returncode, stdout, stderr = run_cli_command("models info qwen/qwen3-1.7b")
    if returncode == 0:
        print("    ✓ Model info works")
    else:
        print(f"    ✗ Model info failed: {stderr}")

def test_adapter_info():
    """Test adapter information commands."""
    print("\nTesting adapter information...")
    
    # Test adapter info for a builtin adapter
    print("  Testing adapter info...")
    returncode, stdout, stderr = run_cli_command("adapters info code_generator")
    if returncode == 0:
        print("    ✓ Adapter info works")
    else:
        print(f"    ✗ Adapter info failed: {stderr}")

def test_build_commands():
    """Test build commands."""
    print("\nTesting build commands...")
    
    # Test creating a build configuration
    print("  Testing build create...")
    returncode, stdout, stderr = run_cli_command(
        "build create test_config --model qwen/qwen3-1.7b --adapters code_generator --description 'Test configuration'"
    )
    if returncode == 0:
        print("    ✓ Build create works")
        
        # Test listing build configurations
        print("  Testing build list...")
        returncode, stdout, stderr = run_cli_command("build list")
        if returncode == 0 and "test_config" in stdout:
            print("    ✓ Build list works")
        else:
            print(f"    ✗ Build list failed: {stderr}")
        
        # Test build info
        print("  Testing build info...")
        returncode, stdout, stderr = run_cli_command("build info test_config")
        if returncode == 0:
            print("    ✓ Build info works")
        else:
            print(f"    ✗ Build info failed: {stderr}")
        
        # Test build delete
        print("  Testing build delete...")
        returncode, stdout, stderr = run_cli_command("build delete test_config --yes")
        if returncode == 0:
            print("    ✓ Build delete works")
        else:
            print(f"    ✗ Build delete failed: {stderr}")
    else:
        print(f"    ✗ Build create failed: {stderr}")

def main():
    """Run all tests."""
    print("Adaptrix CLI Test Suite")
    print("=" * 50)
    
    # Test basic commands
    test_basic_commands()
    
    # Test configuration
    test_configuration()
    
    # Test model info
    test_model_info()
    
    # Test adapter info
    test_adapter_info()
    
    # Test build commands
    test_build_commands()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nNote: Some tests may fail if dependencies are not installed.")
    print("This is expected in a development environment.")

if __name__ == "__main__":
    main()
