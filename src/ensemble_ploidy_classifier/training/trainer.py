"""
EnsembleTrainer for the Ensemble Ploidy Classifier.

This module implements the training logic for the ensemble system,
including probe training, embedding generation, and ensemble training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import json

from ..models import DynamicLeNet, ClassificationMLP, TrainableAggregator
from ..data import CustomDataset, EmbeddingDataset
from ..config import ModelConfig, TrainingConfig
from ..utils.metrics import calculate_metrics


class EnsembleTrainer:
    """
    Trainer for the ensemble ploidy classification system.
    
    This class handles the complete training pipeline including:
    1. Individual probe training
    2. Embedding generation
    3. Ensemble training with learnable aggregation
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the EnsembleTrainer.
        
        Args:
            model_config: Configuration for model architecture
            training_config: Configuration for training parameters
            device: Device to use for training (default: from config)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or training_config.device
        
        # Initialize models
        self.probe_models: List[DynamicLeNet] = []
        self.classifier: Optional[ClassificationMLP] = None
        self.aggregator: Optional[TrainableAggregator] = None
        
        # Training history
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
    
    def create_probe_models(self, num_probes: int) -> List[DynamicLeNet]:
        """
        Create multiple probe models.
        
        Args:
            num_probes: Number of probe models to create
            
        Returns:
            List of DynamicLeNet models
        """
        probe_models = []
        
        for i in range(num_probes):
            model = DynamicLeNet(
                input_channels=self.model_config.input_channels,
                input_size=self.model_config.input_size,
                num_layers=self.model_config.num_layers,
                num_filters=self.model_config.num_filters,
                kernel_sizes=self.model_config.kernel_sizes,
                dropout_rate=self.model_config.dropout_rate,
                activation_fn=self.model_config.activation_fn,
                pool_type=self.model_config.pool_type,
                fc_layer_sizes=[self.model_config.embedding_dim],  # Single layer for embedding
                num_classes=1,  # Binary classification output
            ).to(self.device)
            
            probe_models.append(model)
        
        self.probe_models = probe_models
        return probe_models
    
    def train_probe(
        self,
        probe_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train a single probe model.
        
        Args:
            probe_idx: Index of the probe to train
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save the trained model
            
        Returns:
            Dictionary containing training history
        """
        if probe_idx >= len(self.probe_models):
            raise ValueError(f"Probe index {probe_idx} out of range")
        
        model = self.probe_models[probe_idx]
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        criterion = nn.BCELoss()
        
        # Training history for this probe
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
        }
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(self.training_config.num_epochs):
            # Training phase
            model.train()
            train_losses = []
            train_predictions = []
            train_labels = []
            
            for batch_sequences, batch_labels in tqdm(
                train_loader, desc=f"Training probe {probe_idx}, epoch {epoch+1}"
            ):
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device).squeeze()
                
                optimizer.zero_grad()
                outputs = model(batch_sequences).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_labels.extend(batch_labels.detach().cpu().numpy())
            
            # Validation phase
            model.eval()
            val_losses = []
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch_sequences, batch_labels in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device).squeeze()
                    
                    outputs = model(batch_sequences).squeeze()
                    loss = criterion(outputs, batch_labels)
                    
                    val_losses.append(loss.item())
                    val_predictions.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_auc = calculate_metrics(train_labels, train_predictions)["auc_roc"]
            val_auc = calculate_metrics(val_labels, val_predictions)["auc_roc"]
            
            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.training_config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % self.training_config.log_frequency == 0:
                print(
                    f"Epoch {epoch+1}/{self.training_config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}"
                )
        
        return history
    
    def generate_embeddings(
        self,
        probe_idx: int,
        data_loader: DataLoader,
        model: Optional[DynamicLeNet] = None,
    ) -> np.ndarray:
        """
        Generate embeddings using a trained probe model.
        
        Args:
            probe_idx: Index of the probe model
            data_loader: Data loader for generating embeddings
            model: Optional model to use (if None, uses self.probe_models[probe_idx])
            
        Returns:
            Embeddings array of shape (num_samples, embedding_dim)
        """
        if model is None:
            if probe_idx >= len(self.probe_models):
                raise ValueError(f"Probe index {probe_idx} out of range")
            model = self.probe_models[probe_idx]
        
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch_sequences, _ in tqdm(
                data_loader, desc=f"Generating embeddings for probe {probe_idx}"
            ):
                batch_sequences = batch_sequences.to(self.device)
                batch_embeddings = model(batch_sequences)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_probe_embeddings(
        self,
        probe_idx: int,
        data_loader: DataLoader,
        model: Optional[DynamicLeNet] = None,
    ) -> np.ndarray:
        """
        Get embeddings from a trained probe model by accessing the penultimate layer.
        
        Args:
            probe_idx: Index of the probe model
            data_loader: Data loader for generating embeddings
            model: Optional model to use (if None, uses self.probe_models[probe_idx])
            
        Returns:
            Embeddings array of shape (num_samples, embedding_dim)
        """
        if model is None:
            if probe_idx >= len(self.probe_models):
                raise ValueError(f"Probe index {probe_idx} out of range")
            model = self.probe_models[probe_idx]
        
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch_sequences, _ in tqdm(
                data_loader, desc=f"Getting embeddings for probe {probe_idx}"
            ):
                batch_sequences = batch_sequences.to(self.device)
                
                # Get embeddings from the penultimate layer (before the final classification layer)
                x = model.conv_layers(batch_sequences)
                x = model.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = model.dropout(x)
                
                # Get the output from the last FC layer before the final classification layer
                for i, layer in enumerate(model.fc_layers[:-1]):  # Exclude the final layer
                    x = layer(x)
                
                embeddings.append(x.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def train_ensemble(
        self,
        train_embeddings: List[np.ndarray],
        train_labels: np.ndarray,
        val_embeddings: List[np.ndarray],
        val_labels: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Train the ensemble with learnable aggregation.
        
        Args:
            train_embeddings: List of training embeddings from each probe
            train_labels: Training labels
            val_embeddings: List of validation embeddings from each probe
            val_labels: Validation labels
            
        Returns:
            Dictionary containing training history
        """
        num_probes = len(train_embeddings)
        embedding_dim = train_embeddings[0].shape[1]
        
        # Create ensemble components
        self.classifier = ClassificationMLP(
            input_dim=embedding_dim,
            hidden_dim1=self.model_config.hidden_dim1,
            hidden_dim2=self.model_config.hidden_dim2,
            dropout_rate1=self.model_config.dropout_rate1,
            dropout_rate2=self.model_config.dropout_rate2,
        ).to(self.device)
        
        self.aggregator = TrainableAggregator(
            num_probes=num_probes,
            input_dim=embedding_dim,
            device=self.device,
        ).to(self.device)
        
        # Create data loaders
        train_dataset = EmbeddingDataset(
            np.array(train_embeddings).transpose(1, 0, 2),  # (num_samples, num_probes, embedding_dim)
            train_labels,
        )
        val_dataset = EmbeddingDataset(
            np.array(val_embeddings).transpose(1, 0, 2),
            val_labels,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            pin_memory=self.training_config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            pin_memory=self.training_config.pin_memory,
        )
        
        # Training setup
        optimizer = optim.Adam(
            list(self.classifier.parameters()) + list(self.aggregator.parameters()),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        criterion = nn.BCELoss()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(self.training_config.num_epochs):
            # Training phase
            self.classifier.train()
            self.aggregator.train()
            train_losses = []
            train_predictions = []
            train_labels = []
            
            for batch_embeddings, batch_labels in tqdm(
                train_loader, desc=f"Training ensemble, epoch {epoch+1}"
            ):
                batch_embeddings = batch_embeddings.to(self.device)  # (batch_size, num_probes, embedding_dim)
                batch_labels = batch_labels.to(self.device).squeeze()
                
                optimizer.zero_grad()
                
                # Aggregate embeddings
                probe_embeddings = [
                    batch_embeddings[:, i, :] for i in range(num_probes)
                ]
                aggregated_embeddings = self.aggregator(probe_embeddings)
                
                # Classify
                outputs = self.classifier(aggregated_embeddings).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_labels.extend(batch_labels.detach().cpu().numpy())
            
            # Validation phase
            self.classifier.eval()
            self.aggregator.eval()
            val_losses = []
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch_embeddings, batch_labels in val_loader:
                    batch_embeddings = batch_embeddings.to(self.device)
                    batch_labels = batch_labels.to(self.device).squeeze()
                    
                    probe_embeddings = [
                        batch_embeddings[:, i, :] for i in range(num_probes)
                    ]
                    aggregated_embeddings = self.aggregator(probe_embeddings)
                    outputs = self.classifier(aggregated_embeddings).squeeze()
                    loss = criterion(outputs, batch_labels)
                    
                    val_losses.append(loss.item())
                    val_predictions.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_metrics = calculate_metrics(train_labels, train_predictions)
            val_metrics = calculate_metrics(val_labels, val_predictions)
            
            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_metrics["auc_roc"])
            history["val_auc"].append(val_metrics["auc_roc"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            
            # Early stopping
            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.training_config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % self.training_config.log_frequency == 0:
                print(
                    f"Epoch {epoch+1}/{self.training_config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train AUC: {train_metrics['auc_roc']:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}"
                )
        
        self.training_history = history
        return history
    
    def save_models(self, save_dir: str) -> None:
        """
        Save all trained models.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save probe models
        for i, model in enumerate(self.probe_models):
            torch.save(model.state_dict(), os.path.join(save_dir, f"probe_{i}.pth"))
        
        # Save ensemble components
        if self.classifier is not None:
            torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier.pth"))
        
        if self.aggregator is not None:
            torch.save(self.aggregator.state_dict(), os.path.join(save_dir, "aggregator.pth"))
        
        # Save training history
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_models(self, load_dir: str) -> None:
        """
        Load trained models.
        
        Args:
            load_dir: Directory containing saved models
        """
        # Load probe models
        for i, model in enumerate(self.probe_models):
            model_path = os.path.join(load_dir, f"probe_{i}.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load ensemble components
        classifier_path = os.path.join(load_dir, "classifier.pth")
        if os.path.exists(classifier_path) and self.classifier is not None:
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        
        aggregator_path = os.path.join(load_dir, "aggregator.pth")
        if os.path.exists(aggregator_path) and self.aggregator is not None:
            self.aggregator.load_state_dict(torch.load(aggregator_path, map_location=self.device))
        
        # Load training history
        history_path = os.path.join(load_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.training_history = json.load(f) 