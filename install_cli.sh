#!/bin/bash
"""
Installation script for Adaptrix CLI.

This script installs the Adaptrix CLI and makes it available as a global command.
"""

set -e

echo "🚀 Installing Adaptrix CLI..."

# Get the current directory
ADAPTRIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a symlink in /usr/local/bin (requires sudo)
if [ -w "/usr/local/bin" ]; then
    echo "📦 Creating symlink in /usr/local/bin..."
    ln -sf "$ADAPTRIX_DIR/adaptrix" /usr/local/bin/adaptrix
    echo "✅ Adaptrix CLI installed successfully!"
    echo "You can now run 'adaptrix' from anywhere."
else
    echo "⚠️  Cannot write to /usr/local/bin. Trying alternative installation..."
    
    # Try to create symlink in user's local bin
    mkdir -p "$HOME/.local/bin"
    ln -sf "$ADAPTRIX_DIR/adaptrix" "$HOME/.local/bin/adaptrix"
    
    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "📝 Adding ~/.local/bin to PATH..."
        
        # Add to appropriate shell config file
        if [ -f "$HOME/.bashrc" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            echo "Added to ~/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
            echo "Added to ~/.zshrc"
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
            echo "Added to ~/.profile"
        fi
        
        echo "⚠️  Please restart your terminal or run 'source ~/.bashrc' (or appropriate shell config)"
    fi
    
    echo "✅ Adaptrix CLI installed successfully in ~/.local/bin!"
    echo "You can now run 'adaptrix' from anywhere."
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Try running:"
echo "  adaptrix --help"
echo "  adaptrix models list --available"
echo "  adaptrix config list"
echo ""
echo "For more information, see: src/cli/README.md"
