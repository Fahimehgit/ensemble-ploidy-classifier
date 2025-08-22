#!/usr/bin/env python3
"""
Quick test script to verify package imports work correctly.
"""

def test_imports():
    """Test all major imports step by step."""
    print("🔍 Testing Ensemble Ploidy Classifier imports...")
    
    # Test 1: Main package import
    try:
        import ensemble_ploidy_classifier
        print("✅ Main package import - Success")
        print(f"   Version: {ensemble_ploidy_classifier.__version__}")
        print(f"   Available: {len(ensemble_ploidy_classifier.__all__)} components")
    except Exception as e:
        print(f"❌ Main package import - Failed: {e}")
        return False
    
    # Test 2: Individual model imports
    try:
        from ensemble_ploidy_classifier.models.dynamic_lenet import DynamicLeNet
        print("✅ DynamicLeNet import - Success")
    except Exception as e:
        print(f"❌ DynamicLeNet import - Failed: {e}")
    
    try:
        from ensemble_ploidy_classifier.models.classification_mlp import ClassificationMLP
        print("✅ ClassificationMLP import - Success")
    except Exception as e:
        print(f"❌ ClassificationMLP import - Failed: {e}")
    
    try:
        from ensemble_ploidy_classifier.models.trainable_aggregator import TrainableAggregator
        print("✅ TrainableAggregator import - Success")
    except Exception as e:
        print(f"❌ TrainableAggregator import - Failed: {e}")
    
    # Test 3: Config imports
    try:
        from ensemble_ploidy_classifier.config.model_config import ModelConfig
        print("✅ ModelConfig import - Success")
    except Exception as e:
        print(f"❌ ModelConfig import - Failed: {e}")
    
    # Test 4: Try creating a simple model
    try:
        import torch
        from ensemble_ploidy_classifier.models.dynamic_lenet import DynamicLeNet
        
        model = DynamicLeNet(
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
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 50, 20)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model creation and forward pass - Success (output shape: {output.shape})")
        
    except Exception as e:
        print(f"❌ Model creation test - Failed: {e}")
    
    print("\n🎉 Import testing completed!")

if __name__ == "__main__":
    test_imports() 