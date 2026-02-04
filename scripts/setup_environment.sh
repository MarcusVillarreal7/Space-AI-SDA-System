#!/bin/bash
# Setup script for Space AI development environment

set -e  # Exit on error

echo "=========================================="
echo "Space AI Environment Setup"
echo "=========================================="

# Check Python version
echo -e "\n[1/6] Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.10"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo "❌ Python $PYTHON_VERSION found, but $REQUIRED_VERSION+ required"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# Create virtual environment
echo -e "\n[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n[3/6] Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo -e "\n[4/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"

# Install dependencies
echo -e "\n[5/6] Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Install development dependencies
echo -e "\n[6/6] Installing development dependencies..."
pip install -r requirements-dev.txt
echo "✅ Development dependencies installed"

# Create default configurations
echo -e "\nCreating default configurations..."
python -c "from src.utils.config_loader import Config; Config().create_default_configs()"
echo "✅ Default configurations created"

# Run verification
echo -e "\n=========================================="
echo "Running verification checks..."
echo "=========================================="
python scripts/verify_setup.py

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo -e "\nTo activate the environment in the future:"
echo "  source venv/bin/activate"
echo -e "\nTo deactivate:"
echo "  deactivate"
