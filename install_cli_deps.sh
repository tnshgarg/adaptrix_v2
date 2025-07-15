#!/bin/bash
"""
Installation script for Adaptrix CLI dependencies.

This script installs the minimal dependencies needed for the Adaptrix CLI.
"""

set -e

echo "🚀 Installing Adaptrix CLI dependencies..."

# Get the current directory
ADAPTRIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install minimal dependencies
echo "📦 Installing minimal dependencies..."
pip install click rich pyyaml requests

echo "✅ Minimal dependencies installed successfully!"
echo ""
echo "To install all dependencies, run:"
echo "  pip install -r src/cli/requirements.txt"
echo ""
echo "🎉 Installation complete!"
echo ""
echo "Try running:"
echo "  adaptrix --help"
echo "  adaptrix models list --available"
echo "  adaptrix config list"
echo ""
echo "For more information, see: src/cli/README.md"
