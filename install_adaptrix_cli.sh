#!/bin/bash
# Adaptrix CLI Installer
# This script downloads and installs the Adaptrix CLI

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Emojis
ROCKET="🚀"
PACKAGE="📦"
CHECK="✅"
WARNING="⚠️"
SPARKLES="✨"
TOOLS="🛠️"

echo -e "${BLUE}${ROCKET} Adaptrix CLI Installer${NC}"
echo -e "${BLUE}==============================${NC}"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed. Please install git first.${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install pip first.${NC}"
    exit 1
fi

# Determine pip command
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

# Create installation directory
INSTALL_DIR="$HOME/.adaptrix"
echo -e "${BLUE}${PACKAGE} Creating installation directory at ${INSTALL_DIR}...${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone or update the repository
if [ -d "$INSTALL_DIR/adaptrix" ]; then
    echo -e "${BLUE}${PACKAGE} Updating existing Adaptrix repository...${NC}"
    cd adaptrix
    git pull
else
    echo -e "${BLUE}${PACKAGE} Cloning Adaptrix repository...${NC}"
    git clone https://github.com/adaptrix/adaptrix.git
    cd adaptrix
fi

# Install dependencies
echo -e "${BLUE}${PACKAGE} Installing dependencies...${NC}"

# Install core dependencies first
echo -e "${BLUE}Installing core dependencies...${NC}"
$PIP_CMD install torch transformers peft accelerate

# Install CLI-specific dependencies
echo -e "${BLUE}Installing CLI dependencies...${NC}"
$PIP_CMD install click rich pyyaml requests huggingface-hub

# Install optional dependencies for full functionality
echo -e "${BLUE}Installing optional dependencies...${NC}"
$PIP_CMD install faiss-cpu sentence-transformers pypdf2 python-docx markdown || echo -e "${YELLOW}Some optional dependencies failed to install${NC}"

# Install from requirements if available
if [ -f "src/cli/requirements.txt" ]; then
    echo -e "${BLUE}Installing from requirements.txt...${NC}"
    $PIP_CMD install -r src/cli/requirements.txt || echo -e "${YELLOW}Some requirements failed to install${NC}"
fi

# Make CLI executable
echo -e "${BLUE}${TOOLS} Making CLI executable...${NC}"
chmod +x adaptrix

# Create symlink
echo -e "${BLUE}${PACKAGE} Creating CLI symlink...${NC}"

# Try to create symlink in /usr/local/bin (requires sudo)
if [ -w "/usr/local/bin" ]; then
    ln -sf "$PWD/adaptrix" /usr/local/bin/adaptrix
    echo -e "${GREEN}${CHECK} Adaptrix CLI installed successfully in /usr/local/bin!${NC}"
    echo -e "You can now run 'adaptrix' from anywhere."
else
    echo -e "${YELLOW}${WARNING} Cannot write to /usr/local/bin. Trying alternative installation...${NC}"
    
    # Try to create symlink in user's local bin
    mkdir -p "$HOME/.local/bin"
    ln -sf "$PWD/adaptrix" "$HOME/.local/bin/adaptrix"
    
    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo -e "${BLUE}${PACKAGE} Adding ~/.local/bin to PATH...${NC}"
        
        # Add to appropriate shell config file
        if [ -f "$HOME/.bashrc" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            echo -e "Added to ~/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
            echo -e "Added to ~/.zshrc"
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
            echo -e "Added to ~/.profile"
        fi
        
        echo -e "${YELLOW}${WARNING} Please restart your terminal or run 'source ~/.bashrc' (or appropriate shell config)${NC}"
    fi
    
    echo -e "${GREEN}${CHECK} Adaptrix CLI installed successfully in ~/.local/bin!${NC}"
    echo -e "You can now run 'adaptrix' from anywhere."
fi

# Create configuration directory
echo -e "${BLUE}${PACKAGE} Creating configuration directory...${NC}"
mkdir -p "$HOME/.adaptrix/config"
mkdir -p "$HOME/.adaptrix/models"
mkdir -p "$HOME/.adaptrix/adapters"
mkdir -p "$HOME/.adaptrix/rag"
mkdir -p "$HOME/.adaptrix/logs"
mkdir -p "$HOME/.adaptrix/cache"

# Copy default configuration if it exists
if [ -f "$PWD/src/cli/config/default_config.yaml" ]; then
    echo -e "${BLUE}${PACKAGE} Setting up default configuration...${NC}"
    cp "$PWD/src/cli/config/default_config.yaml" "$HOME/.adaptrix/config/config.yaml"
fi

# Download sample adapters
echo -e "${BLUE}${PACKAGE} Setting up sample adapters...${NC}"
mkdir -p "$HOME/.adaptrix/adapters/code_generator"
mkdir -p "$HOME/.adaptrix/adapters/math_solver"
mkdir -p "$HOME/.adaptrix/adapters/general_assistant"

# Create adapter metadata
cat > "$HOME/.adaptrix/adapters/code_generator/metadata.json" << EOL
{
    "name": "code_generator",
    "description": "Code generation and debugging",
    "domain": "programming",
    "version": "1.0.0",
    "target_layers": [6, 12, 18],
    "target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj"],
    "lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
    "source": "builtin",
    "install_date": "$(date)"
}
EOL

cat > "$HOME/.adaptrix/adapters/math_solver/metadata.json" << EOL
{
    "name": "math_solver",
    "description": "Mathematical problem solving",
    "domain": "mathematics",
    "version": "1.0.0",
    "target_layers": [6, 12, 18],
    "target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj"],
    "lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
    "source": "builtin",
    "install_date": "$(date)"
}
EOL

cat > "$HOME/.adaptrix/adapters/general_assistant/metadata.json" << EOL
{
    "name": "general_assistant",
    "description": "General purpose assistant",
    "domain": "general",
    "version": "1.0.0",
    "target_layers": [6, 12, 18],
    "target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj"],
    "lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
    "source": "builtin",
    "install_date": "$(date)"
}
EOL

# Print success message
echo ""
echo -e "${GREEN}${SPARKLES} Installation complete! ${SPARKLES}${NC}"
echo ""
echo -e "Try running:"
echo -e "  ${BLUE}adaptrix --help${NC}"
echo -e "  ${BLUE}adaptrix models list --available${NC}"
echo -e "  ${BLUE}adaptrix config list${NC}"
echo ""
echo -e "For more information, see: ${BLUE}https://docs.adaptrix.ai/cli${NC}"
echo ""
echo -e "${YELLOW}${ROCKET} Welcome to the Adaptrix ecosystem!${NC}"
