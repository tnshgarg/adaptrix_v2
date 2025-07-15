#!/bin/bash

# Adaptrix Setup Script
# This script sets up the complete Adaptrix environment

set -e  # Exit on any error

echo "🚀 Setting up Adaptrix - The World's First Modular AI System"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.8"
VENV_NAME="adaptrix_env"
MODEL_ID="Qwen/Qwen3-1.7B"
DEVICE="auto"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    
    if [[ $(echo "$PYTHON_VER >= $PYTHON_VERSION" | bc -l) -eq 1 ]]; then
        print_success "Python $PYTHON_VER found"
    else
        print_error "Python $PYTHON_VERSION or higher required. Found: $PYTHON_VER"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_NAME"
    fi
    
    $PYTHON_CMD -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install core dependencies
    pip install -r requirements.txt
    
    # Install optional dependencies based on user choice
    echo ""
    read -p "Install vLLM for optimized inference? (y/n): " install_vllm
    if [[ $install_vllm == "y" || $install_vllm == "Y" ]]; then
        print_status "Installing vLLM..."
        pip install vllm
        print_success "vLLM installed"
    fi
    
    echo ""
    read -p "Install development dependencies? (y/n): " install_dev
    if [[ $install_dev == "y" || $install_dev == "Y" ]]; then
        print_status "Installing development dependencies..."
        pip install pytest pytest-asyncio black flake8 mypy
        print_success "Development dependencies installed"
    fi
    
    print_success "All dependencies installed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p models/classifier
    mkdir -p models/rag_vector_store
    mkdir -p adapters
    mkdir -p logs
    mkdir -p cache
    mkdir -p data/documents
    mkdir -p data/training
    
    print_success "Directory structure created"
}

# Download and setup models
setup_models() {
    print_status "Setting up models..."
    
    echo ""
    read -p "Download base model ($MODEL_ID)? This may take several GB. (y/n): " download_model
    if [[ $download_model == "y" || $download_model == "Y" ]]; then
        print_status "Downloading base model..."
        $PYTHON_CMD -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_ID', trust_remote_code=True)

print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_ID',
    torch_dtype=torch.float16,
    device_map='$DEVICE',
    trust_remote_code=True
)

print('Model downloaded successfully!')
"
        print_success "Base model downloaded"
    else
        print_warning "Base model download skipped. It will be downloaded on first use."
    fi
}

# Train classifier
train_classifier() {
    print_status "Training MoE classifier..."
    
    if [ -f "scripts/train_classifier.py" ]; then
        $PYTHON_CMD scripts/train_classifier.py
        print_success "MoE classifier trained"
    else
        print_warning "Classifier training script not found. Run manually later."
    fi
}

# Setup RAG system
setup_rag() {
    print_status "Setting up RAG system..."
    
    echo ""
    read -p "Setup RAG with sample documents? (y/n): " setup_rag_choice
    if [[ $setup_rag_choice == "y" || $setup_rag_choice == "Y" ]]; then
        if [ -f "scripts/setup_rag.py" ]; then
            $PYTHON_CMD scripts/setup_rag.py
            print_success "RAG system setup complete"
        else
            print_warning "RAG setup script not found. Run manually later."
        fi
    else
        print_warning "RAG setup skipped"
    fi
}

# Create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    cat > .env << EOF
# Adaptrix Configuration
ADAPTRIX_MODEL_ID=$MODEL_ID
ADAPTRIX_DEVICE=$DEVICE
ADAPTRIX_ADAPTERS_DIR=adapters
ADAPTRIX_CLASSIFIER_PATH=models/classifier
ADAPTRIX_RAG_VECTOR_STORE_PATH=models/rag_vector_store

# API Configuration
ADAPTRIX_HOST=0.0.0.0
ADAPTRIX_PORT=8000
ADAPTRIX_ENABLE_AUTO_SELECTION=true
ADAPTRIX_ENABLE_RAG=true
ADAPTRIX_USE_OPTIMIZED_ENGINE=true
ADAPTRIX_ENABLE_CACHING=true

# Performance Settings
ADAPTRIX_MAX_BATCH_SIZE=32
ADAPTRIX_MAX_CONCURRENT_REQUESTS=100

# Security (Change in production)
ADAPTRIX_ENABLE_AUTH=false
ADAPTRIX_API_KEY=your-api-key-here

# Logging
ADAPTRIX_LOG_LEVEL=info
ADAPTRIX_LOG_REQUESTS=true
EOF
    
    print_success "Configuration file created (.env)"
}

# Run tests
run_tests() {
    print_status "Running basic tests..."
    
    echo ""
    read -p "Run system tests? This may take a while. (y/n): " run_tests_choice
    if [[ $run_tests_choice == "y" || $run_tests_choice == "Y" ]]; then
        if [ -f "tests/test_complete_system.py" ]; then
            $PYTHON_CMD -m pytest tests/test_complete_system.py::TestAdaptrixSystem::test_basic_generation -v
            print_success "Basic tests passed"
        else
            print_warning "Test files not found"
        fi
    else
        print_warning "Tests skipped"
    fi
}

# Main setup function
main() {
    echo ""
    print_status "Starting Adaptrix setup..."
    
    # Check prerequisites
    check_python
    
    # Setup environment
    create_venv
    install_dependencies
    create_directories
    
    # Setup models and data
    setup_models
    train_classifier
    setup_rag
    
    # Configuration
    create_config
    
    # Testing
    run_tests
    
    echo ""
    echo "============================================================"
    print_success "🎉 Adaptrix setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source $VENV_NAME/bin/activate"
    echo "2. Start the API server: python -m src.api.main"
    echo "3. Visit http://localhost:8000/docs for API documentation"
    echo "4. Run tests: python -m pytest tests/"
    echo ""
    echo "For more information, see README.md"
    echo "============================================================"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Adaptrix Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --quick        Quick setup (minimal prompts)"
        echo "  --dev          Development setup (includes dev dependencies)"
        echo ""
        exit 0
        ;;
    --quick)
        # Quick setup with defaults
        export QUICK_SETUP=true
        ;;
    --dev)
        # Development setup
        export DEV_SETUP=true
        ;;
esac

# Run main setup
main
