# Adaptrix CLI

A powerful command-line interface for the Adaptrix modular AI system, allowing you to run open-source models, manage adapters, integrate RAG, and build custom AI solutions.

## Features

- **Run Open Source Models**: Use small (<3B parameter) open-source models like Qwen3-1.7B, Phi-2, and TinyLlama
- **Attach Adapters**: Use existing adapters or create custom ones to enhance model capabilities
- **Integrate RAG**: Add document collections for retrieval-augmented generation
- **Build Custom Models**: Compose custom models with adapters and RAG layers
- **Run Inference**: Interact with your models through the CLI

## Installation

```bash
# Install from PyPI
pip install adaptrix-cli

# Or install from source
git clone https://github.com/adaptrix/adaptrix
cd adaptrix
pip install -e src/cli
```

## Quick Start

```bash
# List available models
adaptrix models list --available

# Download a model
adaptrix models download qwen/qwen3-1.7b

# List available adapters
adaptrix adapters list --available

# Install an adapter
adaptrix adapters install code_generator

# Add documents to RAG
adaptrix rag add --collection code_docs path/to/documents

# Create a custom model configuration
adaptrix build create my_code_assistant --model qwen/qwen3-1.7b --adapters code_generator --rag-collection code_docs

# Run inference with your custom model
adaptrix run my_code_assistant --interactive

# Or use the chat command for a quick session
adaptrix chat --model qwen/qwen3-1.7b --adapters code_generator
```

## Command Structure

```
adaptrix
├── models
│   ├── list                # List available models
│   ├── download            # Download a model
│   ├── info                # Show model information
│   └── run                 # Run a model without adapters
├── adapters
│   ├── list                # List available adapters
│   ├── install             # Install an adapter
│   ├── info                # Show adapter information
│   ├── create              # Create a custom adapter (placeholder)
│   └── uninstall           # Uninstall an adapter
├── rag
│   ├── add                 # Add documents to RAG store
│   ├── list                # List document collections
│   ├── create-store        # Create a new vector store
│   ├── info                # Show RAG store information
│   └── remove              # Remove documents from RAG store
├── build
│   ├── create              # Create a custom model configuration
│   ├── list                # List custom model configurations
│   ├── export              # Export a custom model
│   └── delete              # Delete a custom model configuration
├── run                     # Run inference with a model + adapters + RAG
├── chat                    # Interactive chat with a model + adapters + RAG
└── config
    ├── set                 # Set configuration options
    ├── get                 # Get configuration options
    ├── list                # List all configuration options
    └── reset               # Reset configuration to defaults
```

## Configuration

Adaptrix CLI uses a hierarchical configuration system:

1. **Default Configuration**: Built-in defaults
2. **Global Configuration**: User-specific settings in `~/.adaptrix/config.yaml`
3. **Project Configuration**: Project-specific settings in `.adaptrix/config.yaml`
4. **Command-line Arguments**: Override settings for a specific command

View your current configuration:

```bash
adaptrix config list
```

Set configuration values:

```bash
adaptrix config set models.directory /path/to/models
adaptrix config set inference.device cuda
```

## Adapter Composition

Adaptrix supports multiple adapter composition strategies:

- **Sequential**: Chain adapters in sequence (default)
- **Parallel**: Apply adapters in parallel and combine outputs
- **Hierarchical**: Apply adapters in a hierarchical structure
- **Conditional**: Use a router to select the best adapter (MoE)

Specify the strategy when building a custom model:

```bash
adaptrix build create my_custom_model --model qwen/qwen3-1.7b --adapters code_generator math_solver --strategy parallel
```

## RAG Integration

Add documents to a RAG collection:

```bash
# Add a single document
adaptrix rag add --collection docs path/to/document.pdf

# Add a directory of documents
adaptrix rag add --collection docs --recursive path/to/documents
```

Use RAG in inference:

```bash
adaptrix run my_model --rag-collection docs
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM (more for larger models)
- 2GB+ disk space for models and adapters

## License

MIT License
