#!/usr/bin/env python3
"""
Quick Start Example for Ensemble Ploidy Classifier

This example shows how to quickly load and use pre-trained models
for ploidy classification on new data.
"""

import numpy as np
import torch
from pathlib import Path

# Import the package
from ensemble_ploidy_classifier.utils.model_loader import (
    PretrainedModelLoader, 
    load_best_ploidy_model,
    load_top_ploidy_models
)

def example_1_load_best_model():
    """Example 1: Load and use the best performing model."""
    print("🚀 Example 1: Loading best model...")
    
    # Load the best model
    model, config = load_best_ploidy_model()
    model.eval()
    
    print(f"✅ Loaded best model: {config}")
    
    # Example prediction on dummy data
    # Replace this with your actual data
    dummy_input = torch.randn(1, 1, 351, 169)  # (batch, channels, height, width)
    
    with torch.no_grad():
        prediction = model(dummy_input)
        probability = prediction.item()
        
    print(f"📊 Prediction probability: {probability:.4f}")
    print(f"🎯 Predicted class: {'Polyploid' if probability > 0.5 else 'Diploid'}")

def example_2_ensemble_prediction():
    """Example 2: Use multiple models for ensemble prediction."""
    print("\n🚀 Example 2: Ensemble prediction with top models...")
    
    # Load top 3 models
    models_and_configs = load_top_ploidy_models(n=3)
    models = [model for model, config in models_and_configs]
    
    print(f"✅ Loaded {len(models)} models for ensemble")
    
    # Set all models to evaluation mode
    for model in models:
        model.eval()
    
    # Example prediction
    dummy_input = torch.randn(1, 1, 351, 169)
    
    predictions = []
    with torch.no_grad():
        for i, model in enumerate(models):
            pred = model(dummy_input).item()
            predictions.append(pred)
            print(f"  Model {i+1} prediction: {pred:.4f}")
    
    # Ensemble prediction (simple average)
    ensemble_pred = np.mean(predictions)
    ensemble_class = "Polyploid" if ensemble_pred > 0.5 else "Diploid"
    
    print(f"📊 Ensemble prediction: {ensemble_pred:.4f}")
    print(f"🎯 Ensemble class: {ensemble_class}")

def example_3_custom_data_loading():
    """Example 3: Load your own data and make predictions."""
    print("\n🚀 Example 3: Custom data prediction...")
    
    # Load the model loader
    loader = PretrainedModelLoader()
    
    # List available models
    available_models = loader.list_available_models()
    print(f"📋 Available models: {available_models[:10]}...")  # Show first 10
    
    # Load a specific model
    if available_models:
        probe_id = available_models[0]
        model, config = loader.load_probe_model(probe_id)
        model.eval()
        
        print(f"✅ Loaded probe {probe_id}")
        
        # Example: Load your JSON data (replace with actual data loading)
        """
        # Your data loading code would go here:
        with open('your_data.json', 'r') as f:
            data = json.load(f)
        
        # Process your data into the expected format:
        # - Shape: (batch_size, 1, 351, sequence_length)
        # - Reference sequence + auxiliary sequences
        """
        
        # Dummy data for demonstration
        dummy_data = torch.randn(5, 1, 351, 169)  # 5 samples
        
        with torch.no_grad():
            predictions = model(dummy_data)
            
        print("📊 Predictions for 5 samples:")
        for i, pred in enumerate(predictions):
            prob = pred.item()
            class_name = "Polyploid" if prob > 0.5 else "Diploid"
            print(f"  Sample {i+1}: {prob:.4f} ({class_name})")

def main():
    """Run all examples."""
    print("🧬 Ensemble Ploidy Classifier - Quick Start Examples")
    print("=" * 60)
    
    try:
        example_1_load_best_model()
        example_2_ensemble_prediction()
        example_3_custom_data_loading()
        
        print("\n✅ All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("  1. Replace dummy data with your actual sequence data")
        print("  2. Adjust input dimensions if needed")
        print("  3. Fine-tune models on your specific species/dataset")
        print("  4. Check the documentation for advanced usage")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        print("💡 Make sure you have trained models in organized_project/models/")

if __name__ == "__main__":
    main()
