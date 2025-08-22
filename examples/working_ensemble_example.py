"""
Working Ensemble Example

This example demonstrates how to use the WorkingEnsembleClassifier with the actual
trained models from the GreedyModel.ipynb notebook. This is the main way users
should interact with your ensemble ploidy classifier.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ensemble_ploidy_classifier import WorkingEnsembleClassifier, MLPClassifier, ProbeModelLoader
    print("✓ Successfully imported WorkingEnsembleClassifier")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Make sure you have installed the package or are running from the correct directory")
    sys.exit(1)


def main():
    """
    Main example demonstrating the working ensemble classifier.
    """
    print("=" * 60)
    print("Working Ensemble Classifier Example")
    print("=" * 60)
    
    # Configuration - Update these paths to match your setup
    EMBEDDINGS_DIR = "organized_project/embeddings/ReducedEmbeddings1"
    MODEL_WEIGHTS_DIR = "/blue/juannanzhou/fahimeh.rahimi/Gordon /ploidy /ModelWeights"
    
    print(f"Embeddings directory: {EMBEDDINGS_DIR}")
    print(f"Model weights directory: {MODEL_WEIGHTS_DIR}")
    
    # Check if directories exist
    if not os.path.exists(EMBEDDINGS_DIR):
        print(f"✗ Embeddings directory not found: {EMBEDDINGS_DIR}")
        print("Please update EMBEDDINGS_DIR to point to your ReducedEmbeddings1 directory")
        return
    
    if not os.path.exists(MODEL_WEIGHTS_DIR):
        print(f"✗ Model weights directory not found: {MODEL_WEIGHTS_DIR}")
        print("Please update MODEL_WEIGHTS_DIR to point to your ModelWeights directory")
        return
    
    print("✓ All directories found!")
    
    try:
        # Initialize the working ensemble classifier
        print("\n1. Initializing Working Ensemble Classifier...")
        ensemble = WorkingEnsembleClassifier(
            embeddings_dir=EMBEDDINGS_DIR,
            model_weights_dir=MODEL_WEIGHTS_DIR
        )
        
        # Display ensemble information
        info = ensemble.get_ensemble_info()
        print(f"   Available probes: {info['available_probes']}")
        print(f"   Device: {info['device']}")
        print(f"   Is trained: {info['is_trained']}")
        
        # Run greedy probe selection
        print("\n2. Running Greedy Probe Selection...")
        selected_probes = ensemble.greedy_probe_selection(
            max_probes=10,  # Limit to 10 probes for faster execution
            patience=3,      # Stop after 3 probes without improvement
            cv_folds=3       # Use 3-fold CV for faster execution
        )
        
        print(f"   Selected probes: {selected_probes}")
        
        # Train the final ensemble
        print("\n3. Training Final Ensemble...")
        ensemble.train_final_ensemble()
        
        # Evaluate the ensemble
        print("\n4. Evaluating Ensemble Performance...")
        evaluation_results = ensemble.evaluate()
        
        print("   Evaluation Results:")
        print(f"     AUC: {evaluation_results['auc']:.4f}")
        print(f"     Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"     Number of probes: {evaluation_results['num_probes']}")
        print(f"     Selected probes: {evaluation_results['selected_probes']}")
        
        # Make predictions on test data
        print("\n5. Making Predictions...")
        try:
            predictions = ensemble.predict()
            probabilities = ensemble.predict_proba()
            
            print(f"   Predictions shape: {predictions.shape}")
            print(f"   Probabilities shape: {probabilities.shape}")
            print(f"   Sample predictions: {predictions[:10]}")
            print(f"   Sample probabilities: {probabilities[:10, 1]}")  # Positive class probabilities
        except Exception as e:
            print(f"   Prediction failed: {e}")
        
        # Save the trained ensemble
        print("\n6. Saving Ensemble...")
        save_path = "trained_ensemble.pkl"
        ensemble.save_ensemble(save_path)
        print(f"   Ensemble saved to: {save_path}")
        
        # Demonstrate loading the ensemble
        print("\n7. Loading Saved Ensemble...")
        new_ensemble = WorkingEnsembleClassifier(
            embeddings_dir=EMBEDDINGS_DIR,
            model_weights_dir=MODEL_WEIGHTS_DIR
        )
        new_ensemble.load_ensemble(save_path)
        
        # Verify it's the same
        new_info = new_ensemble.get_ensemble_info()
        print(f"   Loaded ensemble probes: {new_info['selected_probes']}")
        print(f"   Is trained: {new_info['is_trained']}")
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_individual_probe_usage():
    """
    Demonstrate how to use individual probe models.
    """
    print("\n" + "=" * 60)
    print("Individual Probe Model Usage Example")
    print("=" * 60)
    
    MODEL_WEIGHTS_DIR = "/blue/juannanzhou/fahimeh.rahimi/Gordon /ploidy /ModelWeights"
    
    try:
        # Initialize probe model loader
        print("1. Initializing Probe Model Loader...")
        probe_loader = ProbeModelLoader(MODEL_WEIGHTS_DIR)
        
        # Get available probes
        available_probes = probe_loader.get_available_probes()
        print(f"   Available probes: {len(available_probes)}")
        print(f"   First 10 probes: {available_probes[:10]}")
        
        # Load a specific probe model
        print("\n2. Loading Individual Probe Model...")
        probe_idx = available_probes[0]  # Use first available probe
        model = probe_loader.load_probe_model(probe_idx)
        
        print(f"   Loaded probe {probe_idx}")
        print(f"   Model architecture: {model}")
        
        # Create dummy data for demonstration
        print("\n3. Testing Model with Dummy Data...")
        dummy_input = torch.randn(5, 700)  # 5 samples, 700 features
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = model.predict_proba(dummy_input)
            predictions = model.predict(dummy_input)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output logits shape: {output.shape}")
        print(f"   Probabilities shape: {probabilities.shape}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions}")
        
        print("\n✓ Individual probe example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during individual probe example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the main ensemble example
    main()
    
    # Run the individual probe example
    demonstrate_individual_probe_usage()
    
    print("\n" + "=" * 60)
    print("Example Summary")
    print("=" * 60)
    print("This example demonstrates:")
    print("1. ✓ Loading your actual trained models from GreedyModel.ipynb")
    print("2. ✓ Running the greedy algorithm for probe selection")
    print("3. ✓ Training the final ensemble classifier")
    print("4. ✓ Making predictions with the trained ensemble")
    print("5. ✓ Saving and loading trained ensembles")
    print("6. ✓ Using individual probe models")
    print("\nThis is the complete working implementation that users will use!") 