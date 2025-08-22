#!/bin/bash

# Ensemble Ploidy Classifier Installation Script

echo "Ensemble Ploidy Classifier - Installation Script"
echo "================================================"

# Check if Python 3.8+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version is compatible"
else
    echo "✗ Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing ensemble-ploidy-classifier in development mode..."
pip install -e .

# Install additional dependencies
echo "Installing additional dependencies..."
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib seaborn
pip install optuna tqdm pyyaml loguru

# Create necessary directories
echo "Creating directories..."
mkdir -p models logs data results plots

# Run tests
read -p "Run tests? (y/n): " run_tests
if [[ $run_tests =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    python -m pytest tests/ -v
fi

echo ""
echo "✓ Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run example: python examples/basic_usage.py"
echo "3. Check documentation: README.md"
echo ""
echo "Happy coding!" 