#!/usr/bin/env python3
"""
Setup script for the Ensemble Ploidy Classifier environment.

This script helps set up the development environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and handle errors.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies() -> bool:
    """Install required dependencies."""
    # Upgrade pip
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core dependencies
    dependencies = [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "optuna>=3.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "loguru>=0.6.0",
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True


def install_dev_dependencies() -> bool:
    """Install development dependencies."""
    dev_dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "pre-commit>=2.20.0",
    ]
    
    for dep in dev_dependencies:
        if not run_command(f"pip install {dep}", f"Installing dev dependency {dep}"):
            return False
    
    return True


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks."""
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        return False
    
    return True


def create_directories() -> bool:
    """Create necessary directories."""
    directories = [
        "models",
        "logs",
        "data",
        "results",
        "plots",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True


def run_tests() -> bool:
    """Run the test suite."""
    return run_command("python -m pytest tests/ -v", "Running tests")


def main():
    """Main setup function."""
    print("Ensemble Ploidy Classifier - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)
    
    # Ask if user wants dev dependencies
    response = input("\nInstall development dependencies? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        if not install_dev_dependencies():
            print("\n✗ Failed to install development dependencies")
            sys.exit(1)
        
        # Setup pre-commit
        response = input("Setup pre-commit hooks? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if not setup_pre_commit():
                print("\n✗ Failed to setup pre-commit hooks")
    
    # Create directories
    if not create_directories():
        print("\n✗ Failed to create directories")
        sys.exit(1)
    
    # Run tests
    response = input("\nRun tests? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        if not run_tests():
            print("\n✗ Some tests failed")
        else:
            print("\n✓ All tests passed")
    
    print("\n" + "=" * 50)
    print("✓ Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Run the basic usage example: python examples/basic_usage.py")
    print("3. Check the documentation in README.md")
    print("4. Start developing!")


if __name__ == "__main__":
    main() 