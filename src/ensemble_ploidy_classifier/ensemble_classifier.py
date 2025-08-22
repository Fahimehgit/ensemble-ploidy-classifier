"""
Main EnsemblePloidyClassifier class for the Ensemble Ploidy Classifier.

This module provides a high-level interface for the complete ensemble system.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import os
import json

from .models import DynamicLeNet, ClassificationMLP, TrainableAggregator
from .data import CustomDataset, EmbeddingDataset
from .config import ModelConfig, TrainingConfig
from .training import EnsembleTrainer
from .utils.metrics import calculate_metrics
from .utils.logging import setup_logging, log_training_start, log_evaluation_results


class EnsemblePloidyClassifier:
    """
    High-level interface for the Ensemble Ploidy Classification system.
    
    This class provides a complete interface for training and using the ensemble
    system, including probe training, embedding generation, and ensemble training.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initialize the EnsemblePloidyClassifier.
        
        Args:
            model_config: Configuration for model architecture
            training_config: Configuration for training parameters
            log_file: Optional path for logging
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup logging
        setup_logging(log_file)
        
        # Initialize trainer
        self.trainer = EnsembleTrainer(model_config, training_config)
        
        # Store trained models
        self.probe_models: List[DynamicLeNet] = []
        self.classifier: Optional[ClassificationMLP] = None
        self.aggregator: Optional[TrainableAggregator] = None
        
        # Training history
        self.training_history: Dict[str, List[float]] = {}
    
    def train(
        self,
        train_sequences: np.ndarray,
        train_labels: np.ndarray,
        val_sequences: np.ndarray,
        val_labels: np.ndarray,
        num_probes: int = 3,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the complete ensemble system.
        
        Args:
            train_sequences: Training sequence data
            train_labels: Training labels
            val_sequences: Validation sequence data
            val_labels: Validation labels
            num_probes: Number of probe models to train
            save_dir: Directory to save trained models
            
        Returns:
            Dictionary containing training history
        """
        log_training_start(
            self.model_config.to_dict(),
            self.training_config.to_dict(),
            num_probes,
        )
        
        # Create probe models
        self.probe_models = self.trainer.create_probe_models(num_probes)
        
        # Create data loaders
        train_dataset = CustomDataset(train_sequences, train_labels)
        val_dataset = CustomDataset(val_sequences, val_labels)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            pin_memory=self.training_config.pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            pin_memory=self.training_config.pin_memory,
        )
        
        # Train individual probes
        probe_histories = []
        for i in range(num_probes):
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"probe_{i}.pth")
            
            history = self.trainer.train_probe(i, train_loader, val_loader, save_path)
            probe_histories.append(history)
        
        # Generate embeddings
        train_embeddings = []
        val_embeddings = []
        
        for i in range(num_probes):
            train_emb = self.trainer.get_probe_embeddings(i, train_loader)
            val_emb = self.trainer.get_probe_embeddings(i, val_loader)
            
            train_embeddings.append(train_emb)
            val_embeddings.append(val_emb)
        
        # Train ensemble
        ensemble_history = self.trainer.train_ensemble(
            train_embeddings, train_labels, val_embeddings, val_labels
        )
        
        # Store ensemble components
        self.classifier = self.trainer.classifier
        self.aggregator = self.trainer.aggregator
        
        # Combine histories
        self.training_history = {
            "probe_histories": probe_histories,
            "ensemble_history": ensemble_history,
        }
        
        # Save models if directory provided
        if save_dir:
            self.save_models(save_dir)
        
        return self.training_history
    
    def predict(
        self,
        sequences: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained ensemble.
        
        Args:
            sequences: Input sequence data
            threshold: Threshold for binary classification
            
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        if not self.probe_models or self.classifier is None or self.aggregator is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Create dataset and loader
        dataset = CustomDataset(sequences, np.zeros(len(sequences)))  # Dummy labels
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            pin_memory=self.training_config.pin_memory,
        )
        
        # Generate embeddings from all probes
        probe_embeddings = []
        for i in range(len(self.probe_models)):
            embeddings = self.trainer.get_probe_embeddings(i, loader)
            probe_embeddings.append(embeddings)
        
        # Aggregate embeddings
        aggregated_embeddings = np.array(probe_embeddings).transpose(1, 0, 2)
        
        # Make predictions
        self.classifier.eval()
        self.aggregator.eval()
        
        probabilities = []
        with torch.no_grad():
            for batch_embeddings in loader:
                batch_embeddings = batch_embeddings[0].to(self.trainer.device)
                
                # Aggregate embeddings
                probe_emb_list = [
                    batch_embeddings[:, i, :] for i in range(len(self.probe_models))
                ]
                aggregated = self.aggregator(probe_emb_list)
                
                # Classify
                outputs = self.classifier(aggregated).squeeze()
                probabilities.extend(outputs.cpu().numpy())
        
        probabilities = np.array(probabilities)
        binary_predictions = (probabilities >= threshold).astype(int)
        
        return probabilities, binary_predictions
    
    def evaluate(
        self,
        test_sequences: np.ndarray,
        test_labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the ensemble on test data.
        
        Args:
            test_sequences: Test sequence data
            test_labels: Test labels
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        probabilities, binary_predictions = self.predict(test_sequences, threshold)
        
        # Calculate metrics
        ensemble_metrics = calculate_metrics(test_labels, probabilities, threshold)
        
        # Calculate individual probe metrics
        probe_metrics = {}
        probe_predictions = []
        
        for i, model in enumerate(self.probe_models):
            dataset = CustomDataset(test_sequences, test_labels)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                pin_memory=self.training_config.pin_memory,
            )
            
            model.eval()
            probe_probs = []
            
            with torch.no_grad():
                for batch_sequences, _ in loader:
                    batch_sequences = batch_sequences.to(self.trainer.device)
                    outputs = model(batch_sequences).squeeze()
                    probe_probs.extend(outputs.cpu().numpy())
            
            probe_probs = np.array(probe_probs)
            probe_metrics[f"probe_{i}"] = calculate_metrics(test_labels, probe_probs, threshold)
            probe_predictions.append(probe_probs)
        
        # Log results
        log_evaluation_results(ensemble_metrics, probe_metrics, ensemble_metrics)
        
        return {
            "ensemble": ensemble_metrics,
            "probes": probe_metrics,
            "probe_predictions": probe_predictions,
            "ensemble_predictions": probabilities,
        }
    
    def save_models(self, save_dir: str) -> None:
        """
        Save all trained models.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configurations
        with open(os.path.join(save_dir, "model_config.json"), "w") as f:
            json.dump(self.model_config.to_dict(), f, indent=2)
        
        with open(os.path.join(save_dir, "training_config.json"), "w") as f:
            json.dump(self.training_config.to_dict(), f, indent=2)
        
        # Save training history
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save models using trainer
        self.trainer.save_models(save_dir)
    
    def load_models(self, load_dir: str) -> None:
        """
        Load trained models.
        
        Args:
            load_dir: Directory containing saved models
        """
        # Load configurations
        with open(os.path.join(load_dir, "model_config.json"), "r") as f:
            model_config_dict = json.load(f)
            self.model_config = ModelConfig.from_dict(model_config_dict)
        
        with open(os.path.join(load_dir, "training_config.json"), "r") as f:
            training_config_dict = json.load(f)
            self.training_config = TrainingConfig.from_dict(training_config_dict)
        
        # Load training history
        history_path = os.path.join(load_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.training_history = json.load(f)
        
        # Recreate trainer and load models
        self.trainer = EnsembleTrainer(self.model_config, self.training_config)
        self.trainer.load_models(load_dir)
        
        # Store references
        self.probe_models = self.trainer.probe_models
        self.classifier = self.trainer.classifier
        self.aggregator = self.trainer.aggregator
    
    def get_aggregation_weights(self) -> np.ndarray:
        """
        Get the learned aggregation weights.
        
        Returns:
            Array of aggregation weights
        """
        if self.aggregator is None:
            raise ValueError("Aggregator not trained. Call train() first.")
        
        return self.aggregator.get_probe_importance().numpy()
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get a summary of the trained models.
        
        Returns:
            Dictionary containing model summary
        """
        summary = {
            "num_probes": len(self.probe_models),
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
        }
        
        if self.aggregator is not None:
            summary["aggregation_weights"] = self.get_aggregation_weights().tolist()
        
        if self.training_history:
            summary["training_history"] = self.training_history
        
        return summary 