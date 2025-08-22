"""
Working Ensemble Classifier for Ploidy Classification

This module implements the actual working ensemble that uses the trained models
from the GreedyModel.ipynb notebook. It loads the trained MLP weights and
implements the greedy algorithm for probe selection.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from .models.mlp_classifier import MLPClassifier, ProbeModelLoader


class WorkingEnsembleClassifier:
    """
    Working Ensemble Classifier that uses the actual trained models from GreedyModel.ipynb.
    
    This classifier:
    1. Loads the trained MLP models for each probe
    2. Implements the greedy algorithm for probe selection
    3. Combines predictions using trainable weights
    4. Provides the final ensemble predictions
    """
    
    def __init__(self, 
                 embeddings_dir: str,
                 model_weights_dir: str,
                 device: Optional[torch.device] = None):
        """
        Initialize the Working Ensemble Classifier.
        
        Args:
            embeddings_dir: Directory containing probe embeddings (ReducedEmbeddings1)
            model_weights_dir: Directory containing trained model weights (ModelWeights)
            device: Device to run models on (default: auto-detect)
        """
        self.embeddings_dir = embeddings_dir
        self.model_weights_dir = model_weights_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.probe_loader = ProbeModelLoader(model_weights_dir, self.device)
        self.scaler = StandardScaler()
        self.final_classifier = None
        
        # Ensemble state
        self.selected_probes = []
        self.probe_weights = None
        self.ensemble_performance = None
        
        # Load available probes
        self.available_probes = self.probe_loader.get_available_probes()
        print(f"Found {len(self.available_probes)} available probe models")
        
        # Check if embeddings directory exists
        if not os.path.exists(embeddings_dir):
            raise ValueError(f"Embeddings directory not found: {embeddings_dir}")
    
    def greedy_probe_selection(self, 
                              max_probes: int = 20,
                              patience: int = 5,
                              cv_folds: int = 5) -> List[int]:
        """
        Perform greedy probe selection to find the best combination.
        
        Args:
            max_probes: Maximum number of probes to select
            patience: Number of probes to try without improvement before stopping
            cv_folds: Number of cross-validation folds
            
        Returns:
            List of selected probe indices
        """
        print(f"Starting greedy probe selection with max {max_probes} probes...")
        
        selected_probes = []
        best_score = 0
        no_improvement_count = 0
        
        # Load training data for evaluation
        train_embeddings, train_labels = self._load_training_data()
        
        for i, probe_idx in enumerate(self.available_probes):
            if len(selected_probes) >= max_probes:
                print(f"Reached maximum number of probes ({max_probes})")
                break
                
            if no_improvement_count >= patience:
                print(f"Stopping due to no improvement for {patience} consecutive probes")
                break
            
            print(f"Evaluating probe {probe_idx} ({i+1}/{len(self.available_probes)})...")
            
            # Try adding this probe
            current_probes = selected_probes + [probe_idx]
            current_score = self._evaluate_probe_combination(
                current_probes, train_embeddings, train_labels, cv_folds
            )
            
            print(f"  Current probes: {current_probes}")
            print(f"  Score: {current_score:.4f} (Best: {best_score:.4f})")
            
            if current_score > best_score:
                best_score = current_score
                selected_probes = current_probes
                no_improvement_count = 0
                print(f"  ✓ Added probe {probe_idx} (new best score: {best_score:.4f})")
            else:
                no_improvement_count += 1
                print(f"  ✗ Probe {probe_idx} did not improve performance")
        
        self.selected_probes = selected_probes
        print(f"\nFinal selected probes: {selected_probes}")
        print(f"Best ensemble score: {best_score:.4f}")
        
        return selected_probes
    
    def _evaluate_probe_combination(self, 
                                   probe_indices: List[int],
                                   train_embeddings: Dict[int, np.ndarray],
                                   train_labels: np.ndarray,
                                   cv_folds: int) -> float:
        """
        Evaluate a combination of probes using cross-validation.
        
        Args:
            probe_indices: List of probe indices to evaluate
            train_embeddings: Dictionary of training embeddings for each probe
            train_labels: Training labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Average AUC across cross-validation folds
        """
        # Load models for the selected probes
        probe_models = {}
        for probe_idx in probe_indices:
            if probe_idx in train_embeddings:
                probe_models[probe_idx] = self.probe_loader.load_probe_model(probe_idx)
        
        # Perform k-fold cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in kfold.split(train_labels):
            # Get predictions for this fold
            fold_predictions = []
            
            for probe_idx in probe_indices:
                if probe_idx in probe_models:
                    # Get embeddings for this fold
                    probe_embeddings = train_embeddings[probe_idx][train_idx]
                    
                    # Convert to tensor and get predictions
                    probe_embeddings_tensor = torch.tensor(
                        probe_embeddings, dtype=torch.float32, device=self.device
                    )
                    
                    with torch.no_grad():
                        probe_probs = probe_models[probe_idx].predict_proba(probe_embeddings_tensor)
                        probe_probs = probe_probs.cpu().numpy()
                    
                    fold_predictions.append(probe_probs[:, 1])  # Probability of positive class
            
            if fold_predictions:
                # Average predictions across probes
                ensemble_predictions = np.mean(fold_predictions, axis=0)
                
                # Calculate AUC for this fold
                try:
                    fold_auc = roc_auc_score(train_labels[val_idx], ensemble_predictions)
                    fold_scores.append(fold_auc)
                except ValueError:
                    fold_scores.append(0.5)  # Default AUC for edge cases
        
        return np.mean(fold_scores) if fold_scores else 0.5
    
    def _load_training_data(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Load training data for all available probes.
        
        Returns:
            Tuple of (embeddings_dict, labels)
        """
        embeddings_dict = {}
        labels = None
        
        print("Loading training data for all probes...")
        
        for probe_idx in self.available_probes:
            probe_dir = os.path.join(self.embeddings_dir, f'probe_{probe_idx}')
            train_file = os.path.join(probe_dir, 'reduced_embeddings_train.npz')
            
            if os.path.exists(train_file):
                try:
                    data = np.load(train_file)
                    if 'embeddings' in data and 'labels' in data:
                        embeddings_dict[probe_idx] = data['embeddings']
                        if labels is None:
                            labels = data['labels']
                        print(f"  Loaded probe {probe_idx}: {data['embeddings'].shape}")
                except Exception as e:
                    print(f"  Error loading probe {probe_idx}: {e}")
            else:
                print(f"  Training file not found for probe {probe_idx}")
        
        print(f"Loaded training data for {len(embeddings_dict)} probes")
        return embeddings_dict, labels
    
    def train_final_ensemble(self, 
                            train_data: Optional[Dict] = None,
                            validation_data: Optional[Dict] = None):
        """
        Train the final ensemble classifier.
        
        Args:
            train_data: Optional training data (if not using probe embeddings)
            validation_data: Optional validation data
        """
        if not self.selected_probes:
            print("No probes selected. Running greedy probe selection first...")
            self.greedy_probe_selection()
        
        print(f"Training final ensemble with {len(self.selected_probes)} selected probes...")
        
        # Load training data
        train_embeddings, train_labels = self._load_training_data()
        
        # Get predictions from all selected probes
        all_predictions = []
        
        for probe_idx in self.selected_probes:
            if probe_idx in train_embeddings:
                # Load model and get predictions
                model = self.probe_loader.load_probe_model(probe_idx)
                embeddings = train_embeddings[probe_idx]
                
                # Convert to tensor and get predictions
                embeddings_tensor = torch.tensor(
                    embeddings, dtype=torch.float32, device=self.device
                )
                
                with torch.no_grad():
                    probe_probs = model.predict_proba(embeddings_tensor)
                    probe_probs = probe_probs.cpu().numpy()
                
                all_predictions.append(probe_probs[:, 1])  # Probability of positive class
        
        if all_predictions:
            # Stack predictions from all probes
            ensemble_features = np.column_stack(all_predictions)
            
            # Scale features
            ensemble_features_scaled = self.scaler.fit_transform(ensemble_features)
            
            # Train final classifier (logistic regression)
            self.final_classifier = LogisticRegression(
                C=1.0, random_state=42, max_iter=1000
            )
            self.final_classifier.fit(ensemble_features_scaled, train_labels)
            
            print("Final ensemble training completed!")
        else:
            raise ValueError("No valid predictions obtained from selected probes")
    
    def predict(self, test_data: Optional[Dict] = None) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            test_data: Optional test data (if not using probe embeddings)
            
        Returns:
            Predictions
        """
        if self.final_classifier is None:
            raise ValueError("Final ensemble not trained. Please call train_final_ensemble() first.")
        
        if not self.selected_probes:
            raise ValueError("No probes selected. Please run greedy_probe_selection() first.")
        
        # Load test data
        test_embeddings, _ = self._load_test_data()
        
        # Get predictions from all selected probes
        all_predictions = []
        
        for probe_idx in self.selected_probes:
            if probe_idx in test_embeddings:
                # Load model and get predictions
                model = self.probe_loader.load_probe_model(probe_idx)
                embeddings = test_embeddings[probe_idx]
                
                # Convert to tensor and get predictions
                embeddings_tensor = torch.tensor(
                    embeddings, dtype=torch.float32, device=self.device
                )
                
                with torch.no_grad():
                    probe_probs = model.predict_proba(embeddings_tensor)
                    probe_probs = probe_probs.cpu().numpy()
                
                all_predictions.append(probe_probs[:, 1])  # Probability of positive class
        
        if all_predictions:
            # Stack predictions from all probes
            ensemble_features = np.column_stack(all_predictions)
            
            # Scale features
            ensemble_features_scaled = self.scaler.transform(ensemble_features)
            
            # Make final predictions
            predictions = self.final_classifier.predict(ensemble_features_scaled)
            return predictions
        else:
            raise ValueError("No valid predictions obtained from selected probes")
    
    def predict_proba(self, test_data: Optional[Dict] = None) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            test_data: Optional test data (if not using probe embeddings)
            
        Returns:
            Prediction probabilities
        """
        if self.final_classifier is None:
            raise ValueError("Final ensemble not trained. Please call train_final_ensemble() first.")
        
        if not self.selected_probes:
            raise ValueError("No probes selected. Please run greedy_probe_selection() first.")
        
        # Load test data
        test_embeddings, _ = self._load_test_data()
        
        # Get predictions from all selected probes
        all_predictions = []
        
        for probe_idx in self.selected_probes:
            if probe_idx in test_embeddings:
                # Load model and get predictions
                model = self.probe_loader.load_probe_model(probe_idx)
                embeddings = test_embeddings[probe_idx]
                
                # Convert to tensor and get predictions
                embeddings_tensor = torch.tensor(
                    embeddings, dtype=torch.float32, device=self.device
                )
                
                with torch.no_grad():
                    probe_probs = model.predict_proba(embeddings_tensor)
                    probe_probs = probe_probs.cpu().numpy()
                
                all_predictions.append(probe_probs[:, 1])  # Probability of positive class
        
        if all_predictions:
            # Stack predictions from all probes
            ensemble_features = np.column_stack(all_predictions)
            
            # Scale features
            ensemble_features_scaled = self.scaler.transform(ensemble_features)
            
            # Get final probabilities
            probabilities = self.final_classifier.predict_proba(ensemble_features_scaled)
            return probabilities
        else:
            raise ValueError("No valid predictions obtained from selected probes")
    
    def _load_test_data(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Load test data for all selected probes.
        
        Returns:
            Tuple of (embeddings_dict, labels)
        """
        embeddings_dict = {}
        labels = None
        
        for probe_idx in self.selected_probes:
            probe_dir = os.path.join(self.embeddings_dir, f'probe_{probe_idx}')
            test_file = os.path.join(probe_dir, 'reduced_embeddings_test.npz')
            
            if os.path.exists(test_file):
                try:
                    data = np.load(test_file)
                    if 'embeddings' in data and 'labels' in data:
                        embeddings_dict[probe_idx] = data['embeddings']
                        if labels is None:
                            labels = data['labels']
                except Exception as e:
                    print(f"Error loading test data for probe {probe_idx}: {e}")
        
        return embeddings_dict, labels
    
    def evaluate(self, test_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Evaluate the ensemble classifier.
        
        Args:
            test_data: Optional test data (if not using probe embeddings)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.final_classifier is None:
            raise ValueError("Final ensemble not trained. Please call train_final_ensemble() first.")
        
        # Load test data
        test_embeddings, test_labels = self._load_test_data()
        
        # Get predictions
        predictions = self.predict()
        probabilities = self.predict_proba()
        
        # Calculate metrics
        try:
            auc = roc_auc_score(test_labels, probabilities[:, 1])
        except ValueError:
            auc = 0.5
        
        accuracy = accuracy_score(test_labels, predictions)
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'num_probes': len(self.selected_probes),
            'selected_probes': self.selected_probes
        }
    
    def save_ensemble(self, filepath: str):
        """Save the trained ensemble."""
        ensemble_data = {
            'selected_probes': self.selected_probes,
            'scaler': self.scaler,
            'final_classifier': self.final_classifier,
            'embeddings_dir': self.embeddings_dir,
            'model_weights_dir': self.model_weights_dir
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        print(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a trained ensemble."""
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.selected_probes = ensemble_data['selected_probes']
        self.scaler = ensemble_data['scaler']
        self.final_classifier = ensemble_data['final_classifier']
        
        print(f"Ensemble loaded from {filepath}")
        print(f"Selected probes: {self.selected_probes}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the current ensemble."""
        return {
            'selected_probes': self.selected_probes,
            'num_probes': len(self.selected_probes),
            'available_probes': len(self.available_probes),
            'embeddings_dir': self.embeddings_dir,
            'model_weights_dir': self.model_weights_dir,
            'device': str(self.device),
            'is_trained': self.final_classifier is not None
        } 