"""
Working example for the Ensemble Ploidy Classifier.

This script demonstrates a complete working example with proper configuration.
"""

import numpy as np
import torch
from ensemble_ploidy_classifier import EnsemblePloidyClassifier
from ensemble_ploidy_classifier.config import ModelConfig, TrainingConfig


def create_dummy_data(num_samples: int = 1000, input_size: int = 256) -> tuple:
    """
    Create dummy data for testing the ensemble classifier.
    
    Args:
        num_samples: Number of samples to create
        input_size: Size of input sequences (should be a perfect square for 2D data)
        
    Returns:
        Tuple of (train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels)
    """
    print("Creating dummy data...")
    
    # Calculate dimensions for 2D data (assuming square input)
    side_length = int(np.sqrt(input_size))
    if side_length * side_length != input_size:
        # If not a perfect square, use the closest perfect square
        side_length = int(np.sqrt(input_size))
        input_size = side_length * side_length
        print(f"Adjusted input_size to {input_size} for square dimensions")
    
    # Create dummy sequences (num_samples, 1, height, width)
    sequences = np.random.randn(num_samples, 1, side_length, side_length).astype(np.float32)
    
    # Create dummy labels (binary classification)
    labels = np.random.randint(0, 2, num_samples).astype(np.float32)
    
    # Split into train/validation/test
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_sequences = sequences[:train_size]
    train_labels = labels[:train_size]
    
    val_sequences = sequences[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    
    test_sequences = sequences[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    print(f"Input shape: {train_sequences.shape[1:]}")
    
    return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels


def main():
    """Main function demonstrating working example."""
    print("Ensemble Ploidy Classifier - Working Example")
    print("=" * 50)
    
    # Create dummy data with manageable size
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = create_dummy_data(
        num_samples=1000, input_size=256  # 16x16 = 256
    )
    
    print("\nSetting up configurations...")
    
    # Set up model configuration with small, manageable dimensions
    model_config = ModelConfig(
        input_channels=1,
        input_size=256,  # 16x16
        num_layers=2,
        num_filters=[16, 32],  # Small filter sizes
        kernel_sizes=[3, 3],
        dropout_rate=0.3,
        activation_fn='relu',
        pool_type='max',
        embedding_dim=64,  # Small embedding dimension
        num_classes=1,
        hidden_dim1=32,  # Small hidden dimensions
        hidden_dim2=16,
        dropout_rate1=0.2,
        dropout_rate2=0.2,
        device='cpu'
    )
    
    # Set up training configuration for quick testing
    training_config = TrainingConfig(
        batch_size=8,  # Small batch size
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_epochs=5,  # Very few epochs for quick testing
        patience=3,
        early_stopping=True,
        device='cpu'
    )
    
    print("Model Configuration:")
    print(f"  Input size: {model_config.input_size}")
    print(f"  Embedding dim: {model_config.embedding_dim}")
    print(f"  Hidden dims: {model_config.hidden_dim1}, {model_config.hidden_dim2}")
    
    print("\nTraining Configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Device: {training_config.device}")
    
    print("\nInitializing ensemble classifier...")
    
    # Initialize the ensemble classifier
    classifier = EnsemblePloidyClassifier(
        model_config=model_config,
        training_config=training_config
    )
    
    print("\nTraining ensemble...")
    
    # Train the ensemble
    try:
        training_history = classifier.train(
            train_sequences=train_sequences,
            train_labels=train_labels,
            val_sequences=val_sequences,
            val_labels=val_labels,
            num_probes=2,  # Reduced number of probes
            save_dir='./models'
        )
        
        print("\nTraining completed!")
        print(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
        print(f"Final training AUC: {training_history['train_auc'][-1]:.4f}")
        print(f"Final validation AUC: {training_history['val_auc'][-1]:.4f}")
        
        print("\nEvaluating on test set...")
        
        # Evaluate on test set
        test_metrics = classifier.evaluate(
            test_sequences=test_sequences,
            test_labels=test_labels
        )
        
        print("Test Set Performance:")
        for metric, value in test_metrics['ensemble'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nMaking predictions...")
        
        # Make predictions
        predictions, probabilities = classifier.predict(test_sequences)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Sample predictions: {predictions[:5]}")
        print(f"Sample probabilities: {probabilities[:5]}")
        
        print("\nGetting aggregation weights...")
        
        # Get aggregation weights
        weights = classifier.get_aggregation_weights()
        print(f"Aggregation weights: {weights}")
        
        print("\nGetting model summary...")
        
        # Get model summary
        summary = classifier.get_model_summary()
        print("Model Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Working example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 