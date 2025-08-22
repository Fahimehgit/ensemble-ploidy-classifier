"""
Model configuration for the Ensemble Ploidy Classifier.

This module defines the configuration classes for model architecture parameters.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import torch


@dataclass
class ModelConfig:
    """
    Configuration class for model architecture parameters.
    
    This class contains all the hyperparameters needed to configure
    the DynamicLeNet, ClassificationMLP, and other model components.
    """
    
    # Input parameters
    input_channels: int = 1
    input_size: int = 59136
    
    # Convolutional layer parameters
    num_layers: int = 3
    num_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    
    # Regularization parameters
    dropout_rate: float = 0.5
    
    # Activation and pooling
    activation_fn: Literal["relu", "leaky_relu", "elu"] = "relu"
    pool_type: Literal["max", "avg"] = "max"
    
    # Embedding and classification parameters
    embedding_dim: int = 700
    num_classes: int = 1
    
    # MLP parameters
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    dropout_rate1: float = 0.3
    dropout_rate2: float = 0.3
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate that all parameters are within acceptable ranges."""
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        if len(self.num_filters) != self.num_layers:
            raise ValueError(f"num_filters length ({len(self.num_filters)}) must match num_layers ({self.num_layers})")
        
        if len(self.kernel_sizes) != self.num_layers:
            raise ValueError(f"kernel_sizes length ({len(self.kernel_sizes)}) must match num_layers ({self.num_layers})")
        
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
        
        if not (0.0 <= self.dropout_rate1 <= 1.0):
            raise ValueError("dropout_rate1 must be between 0.0 and 1.0")
        
        if not (0.0 <= self.dropout_rate2 <= 1.0):
            raise ValueError("dropout_rate2 must be between 0.0 and 1.0")
        
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        if self.hidden_dim1 <= 0 or self.hidden_dim2 <= 0:
            raise ValueError("hidden dimensions must be positive")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "num_layers": self.num_layers,
            "num_filters": self.num_filters,
            "kernel_sizes": self.kernel_sizes,
            "dropout_rate": self.dropout_rate,
            "activation_fn": self.activation_fn,
            "pool_type": self.pool_type,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "dropout_rate1": self.dropout_rate1,
            "dropout_rate2": self.dropout_rate2,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ModelConfig(input_channels={self.input_channels}, input_size={self.input_size}, " \
               f"num_layers={self.num_layers}, embedding_dim={self.embedding_dim}, device={self.device})" 