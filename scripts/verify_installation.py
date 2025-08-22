#!/usr/bin/env python3
"""
Verification script to check if ensemble_ploidy_classifier is properly installed.
Run this after installation to ensure everything is working correctly.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_package_import():
    """Check if the main package can be imported."""
    try:
        import ensemble_ploidy_classifier
        print("✅ Package import - Success")
        print(f"   Package location: {ensemble_ploidy_classifier.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Package import - Failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = [
        "torch",
        "torchvision", 
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "optuna",
        "tqdm",
        "yaml",
        "loguru"
    ]
    
    failed = []
    for dep in dependencies:
        try:
            if dep == "sklearn":
                import sklearn
            elif dep == "yaml":
                import yaml
            else:
                __import__(dep)
            print(f"✅ {dep} - Available")
        except ImportError:
            print(f"❌ {dep} - Missing")
            failed.append(dep)
    
    return len(failed) == 0, failed

def check_models_submodules():
    """Check if model submodules can be imported."""
    submodules = [
        "ensemble_ploidy_classifier.models.dynamic_lenet",
        "ensemble_ploidy_classifier.models.classification_mlp", 
        "ensemble_ploidy_classifier.models.trainable_aggregator",
        "ensemble_ploidy_classifier.config.model_config",
        "ensemble_ploidy_classifier.training.trainer",
    ]
    
    failed = []
    for module in submodules:
        try:
            __import__(module)
            print(f"✅ {module.split('.')[-1]} - Available")
        except ImportError as e:
            print(f"❌ {module.split('.')[-1]} - Failed: {e}")
            failed.append(module)
    
    return len(failed) == 0, failed

def check_examples():
    """Check if examples directory exists and contains files."""
    examples_dir = Path("examples")
    if examples_dir.exists():
        examples = list(examples_dir.glob("*.py"))
        if examples:
            print(f"✅ Examples - Found {len(examples)} example files")
            for example in examples:
                print(f"   - {example.name}")
            return True
        else:
            print("⚠️  Examples directory exists but is empty")
            return False
    else:
        print("⚠️  Examples directory not found")
        return False

def run_basic_functionality_test():
    """Run a basic functionality test."""
    try:
        print("\n🧪 Running basic functionality test...")
        
        # Test model configuration
        from ensemble_ploidy_classifier.config import ModelConfig
        config = ModelConfig(
            input_channels=1,
            input_size=1000,
            num_layers=2,
            num_filters=[32, 64],
            kernel_sizes=[3, 3],
            dropout_rate=0.3,
            activation_fn='relu',
            pool_type='max',
            embedding_dim=128
        )
        print("✅ ModelConfig creation - Success")
        
        # Test model creation
        from ensemble_ploidy_classifier.models import DynamicLeNet
        model = DynamicLeNet(
            input_channels=config.input_channels,
            input_size=config.input_size,
            num_layers=config.num_layers,
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            dropout_rate=config.dropout_rate,
            activation_fn=config.activation_fn,
            pool_type=config.pool_type,
            embedding_dim=config.embedding_dim
        )
        print("✅ DynamicLeNet creation - Success")
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(1, 1, 50, 20)  # Small test input
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass - Success (output shape: {output.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test - Failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🔍 Verifying Ensemble Ploidy Classifier Installation")
    print("=" * 55)
    
    checks = []
    
    # Python version check
    checks.append(check_python_version())
    
    # Package import check
    checks.append(check_package_import())
    
    # Dependencies check
    deps_ok, failed_deps = check_dependencies()
    checks.append(deps_ok)
    
    print("\n📦 Checking model submodules...")
    modules_ok, failed_modules = check_models_submodules()
    checks.append(modules_ok)
    
    print("\n📁 Checking examples...")
    examples_ok = check_examples()
    
    # Basic functionality test
    if all(checks):
        functionality_ok = run_basic_functionality_test()
        checks.append(functionality_ok)
    
    print("\n" + "=" * 55)
    
    if all(checks):
        print("🎉 All checks passed! Installation is working correctly.")
        print("\n📖 Next steps:")
        print("   1. Try running: python examples/basic_usage.py")
        print("   2. Check the README.md for usage examples")
        print("   3. Visit the documentation for more details")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        if not deps_ok:
            print(f"\n💡 To install missing dependencies:")
            print(f"   pip install {' '.join(failed_deps)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 