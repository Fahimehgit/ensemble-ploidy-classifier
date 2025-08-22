#!/usr/bin/env python3
"""
Simple Usage Example for Ensemble Ploidy Classifier

This example shows the most basic way to use the package,
with fallbacks for potential import issues.
"""

def simple_model_example():
    """Demonstrate basic model usage with error handling."""
    print("🚀 Simple Ensemble Ploidy Classifier Example")
    print("=" * 45)
    
    try:
        # Method 1: Direct imports (recommended)
        from ensemble_ploidy_classifier.models.dynamic_lenet import DynamicLeNet
        from ensemble_ploidy_classifier.config.model_config import ModelConfig
        print("✅ Direct imports successful")
        
    except ImportError:
        print("⚠️  Direct imports failed, trying alternative method...")
        try:
            # Method 2: Import from installed package
            import sys
            import os
            
            # Add the source directory to path if needed
            src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
            if os.path.exists(src_path):
                sys.path.insert(0, src_path)
            
            from ensemble_ploidy_classifier.models.dynamic_lenet import DynamicLeNet
            from ensemble_ploidy_classifier.config.model_config import ModelConfig
            print("✅ Alternative imports successful")
            
        except ImportError as e:
            print(f"❌ Failed to import required modules: {e}")
            print("\n💡 Try installing the package with:")
            print("   pip install git+https://github.com/Fahimehgit/ensemble-ploidy-classifier.git")
            return False
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch")
        return False
    
    # Create a simple configuration
    print("\n🔧 Creating model configuration...")
    config = ModelConfig(
        input_channels=1,
        input_size=1000,  # Smaller for testing
        num_layers=2,
        num_filters=[32, 64],
        kernel_sizes=[3, 3],
        dropout_rate=0.3,
        activation_fn='relu',
        pool_type='max',
        embedding_dim=128
    )
    print(f"✅ Config created: {config.num_layers} layers, {config.embedding_dim}D embeddings")
    
    # Create the model
    print("\n🧠 Creating DynamicLeNet model...")
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
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy data
    print("\n🧪 Testing with dummy data...")
    dummy_input = torch.randn(2, 1, 50, 20)  # batch_size=2, channels=1, height=50, width=20
    print(f"   Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Interpret results
    print("\n📊 Results interpretation:")
    for i, prediction in enumerate(output):
        prob = prediction.item()
        prediction_class = "Polyploid" if prob > 0.5 else "Diploid"
        confidence = max(prob, 1-prob)
        print(f"   Sample {i+1}: {prediction_class} (confidence: {confidence:.3f})")
    
    print("\n🎉 Example completed successfully!")
    print("\n📚 Next steps:")
    print("   - Try with your own data")
    print("   - Explore ensemble_classifier.py for advanced usage")
    print("   - Check examples/basic_usage.py for more features")
    
    return True

if __name__ == "__main__":
    success = simple_model_example()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure PyTorch is installed: pip install torch")
        print("   2. Reinstall the package: pip install --force-reinstall git+https://github.com/Fahimehgit/ensemble-ploidy-classifier.git")
        print("   3. Try running from the project directory after: pip install -e .")
        exit(1) 