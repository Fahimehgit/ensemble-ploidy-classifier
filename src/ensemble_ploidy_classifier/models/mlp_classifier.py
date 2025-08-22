"""
MLP Classifier for Individual Probes

This module implements the MLP architecture that matches the trained models
from the GreedyModel.ipynb notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MLPClassifier(nn.Module):
    """
    MLP Classifier that matches the architecture used in the trained models.
    
    This is the exact architecture that was trained and saved in the ModelWeights directory.
    """
    
    def __init__(self, input_dim: int = 700, hidden_dims: list = [512, 256, 128], 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize the MLP Classifier.
        
        Args:
            input_dim: Dimension of input embeddings (default: 700 for reduced embeddings)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes (default: 2 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.mlp(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Prediction probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class labels
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate embeddings from the last hidden layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Embeddings from the last hidden layer
        """
        # Get all layers except the output layer
        feature_layers = self.mlp[:-1]
        embeddings = feature_layers(x)
        return embeddings


class ProbeModelLoader:
    """
    Loader for individual probe models trained in the GreedyModel.ipynb notebook.
    """
    
    def __init__(self, model_weights_dir: str, device: Optional[torch.device] = None):
        """
        Initialize the probe model loader.
        
        Args:
            model_weights_dir: Directory containing the trained model weights
            device: Device to load models on (default: auto-detect)
        """
        self.model_weights_dir = model_weights_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}
        
        # Check if the directory exists
        if not os.path.exists(model_weights_dir):
            raise ValueError(f"Model weights directory not found: {model_weights_dir}")
    
    def load_probe_model(self, probe_idx: int) -> MLPClassifier:
        """
        Load a specific probe model.
        
        Args:
            probe_idx: Index of the probe to load
            
        Returns:
            Loaded MLPClassifier model
        """
        if probe_idx in self.loaded_models:
            return self.loaded_models[probe_idx]
        
        # Construct the weight file path
        weight_file = os.path.join(self.model_weights_dir, f'mlp_weights_probe_{probe_idx}.pt')
        
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Weight file not found: {weight_file}")
        
        # Create model with the correct architecture
        model = MLPClassifier(input_dim=700)  # Based on your reduced embeddings dimension
        
        # Load the trained weights
        checkpoint = torch.load(weight_file, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # Cache the loaded model
        self.loaded_models[probe_idx] = model
        
        return model
    
    def load_multiple_probes(self, probe_indices: list) -> dict:
        """
        Load multiple probe models.
        
        Args:
            probe_indices: List of probe indices to load
            
        Returns:
            Dictionary mapping probe indices to loaded models
        """
        models = {}
        for probe_idx in probe_indices:
            models[probe_idx] = self.load_probe_model(probe_idx)
        return models
    
    def get_available_probes(self) -> list:
        """
        Get list of available probe indices.
        
        Returns:
            List of available probe indices
        """
        available_probes = []
        for filename in os.listdir(self.model_weights_dir):
            if filename.startswith('mlp_weights_probe_') and filename.endswith('.pt'):
                # Extract probe index from filename
                probe_idx = int(filename.replace('mlp_weights_probe_', '').replace('.pt', ''))
                available_probes.append(probe_idx)
        
        return sorted(available_probes)
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Import os for path operations
import os 