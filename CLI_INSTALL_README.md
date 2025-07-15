# Adaptrix CLI Installation

This repository contains the installation script for the Adaptrix CLI, a powerful command-line interface for the Adaptrix modular AI system.

## One-Line Installation

```bash
curl -sSL https://raw.githubusercontent.com/adaptrix/adaptrix/main/install_adaptrix_cli.sh | bash
```

## What Does It Do?

The installation script:

1. Clones the Adaptrix repository to `~/.adaptrix`
2. Installs required dependencies
3. Creates a global `adaptrix` command
4. Sets up configuration directories

## Requirements

- Python 3.8+
- Git
- pip

## Supported Platforms

- macOS
- Linux
- Windows (via WSL)

## CLI Features

The Adaptrix CLI provides access to:

- **Model Management**: Download and run open-source models (<3B parameters)
- **Adapter Management**: Install and compose LoRA adapters
- **RAG Integration**: Add documents and create vector stores
- **Custom Model Building**: Create custom model configurations
- **Inference**: Run models with adapters and RAG

## Quick Start

After installation, try these commands:

```bash
# Get help
adaptrix --help

# List available models
adaptrix models list --available

# Download a model
adaptrix models download qwen/qwen3-1.7b

# List available adapters
adaptrix adapters list --available

# Install an adapter
adaptrix adapters install code_generator

# Create a custom configuration
adaptrix build create my_assistant --model qwen/qwen3-1.7b --adapters code_generator

# Start chatting
adaptrix chat --model qwen/qwen3-1.7b
```

## Documentation

For more information, visit [https://docs.adaptrix.ai/cli](https://docs.adaptrix.ai/cli)

## License

MIT License
