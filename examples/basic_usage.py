#!/usr/bin/env python3
"""
Basic usage example for the Ensemble Ploidy Classifier.

This script demonstrates how to use the ensemble classifier for training
and prediction.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import the ensemble classifier
from ensemble_ploidy_classifier import EnsemblePloidyClassifier
from ensemble_ploidy_classifier.config import ModelConfig, TrainingConfig
from ensemble_ploidy_classifier.data import CustomDataset


def create_dummy_data(num_samples: int = 1000, input_size: int = 59136) -> tuple:
    """
    Create dummy data for demonstration.
    
    Args:
        num_samples: Number of samples to create
        input_size: Size of input sequences
        
    Returns:
        Tuple of (sequences, labels)
    """
    # Create dummy sequences (simulating genomic data)
    sequences = np.random.randn(num_samples, 1, int(np.sqrt(input_size)), int(np.sqrt(input_size)))
    
    # Create dummy labels (binary classification)
    labels = np.random.randint(0, 2, num_samples)
    
    return sequences, labels


def main():
    """Main function demonstrating basic usage."""
    print("Ensemble Ploidy Classifier - Basic Usage Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create dummy data
    print("Creating dummy data...")
    train_sequences, train_labels = create_dummy_data(800)
    val_sequences, val_labels = create_dummy_data(200)
    test_sequences, test_labels = create_dummy_data(100)
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    
    # Create configurations
    print("\nSetting up configurations...")
    model_config = ModelConfig(
        input_channels=1,
        input_size=59136,
        num_layers=3,
        num_filters=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        dropout_rate=0.5,
        activation_fn="relu",
        pool_type="max",
        embedding_dim=700,
        hidden_dim1=512,
        hidden_dim2=256,
        dropout_rate1=0.3,
        dropout_rate2=0.3,
    )
    
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_epochs=10,  # Reduced for demo
        patience=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Initialize classifier
    print("Initializing ensemble classifier...")
    classifier = EnsemblePloidyClassifier(model_config, training_config)
    
    # Train the ensemble
    print("\nTraining ensemble...")
    training_history = classifier.train(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        num_probes=3,
        save_dir="./models",
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluation_results = classifier.evaluate(
        test_sequences=test_sequences,
        test_labels=test_labels,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 30)
    print("Ensemble Performance:")
    for metric, value in evaluation_results["ensemble"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nIndividual Probe Performance:")
    for probe_name, metrics in evaluation_results["probes"].items():
        print(f"  {probe_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Get aggregation weights
    print("\nLearned Aggregation Weights:")
    weights = classifier.get_aggregation_weights()
    for i, weight in enumerate(weights):
        print(f"  Probe {i}: {weight:.3f}")
    
    # Make predictions on new data
    print("\nMaking predictions on new data...")
    new_sequences, _ = create_dummy_data(10)
    probabilities, predictions = classifier.predict(new_sequences)
    
    print("Predictions:")
    for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
        print(f"  Sample {i}: Probability={prob:.3f}, Prediction={pred}")
    
    # Save models
    print("\nSaving models...")
    classifier.save_models("./models")
    
    # Load models (demonstration)
    print("Loading models...")
    new_classifier = EnsemblePloidyClassifier(model_config, training_config)
    new_classifier.load_models("./models")
    
    # Verify loaded models work
    test_probs, test_preds = new_classifier.predict(test_sequences[:5])
    print("Loaded model predictions (first 5 samples):")
    for i, (prob, pred) in enumerate(zip(test_probs, test_preds)):
        print(f"  Sample {i}: Probability={prob:.3f}, Prediction={pred}")
    
    print("\nBasic usage example completed successfully!")


if __name__ == "__main__":
    main() 