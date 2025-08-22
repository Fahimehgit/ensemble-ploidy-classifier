"""
Training configuration for the Ensemble Ploidy Classifier.

This module defines the configuration classes for training parameters.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    
    This class contains all the hyperparameters needed for training
    the ensemble models including optimizers, schedulers, and training loops.
    """
    
    # Data parameters
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimizer parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer_type: str = "adam"  # "adam", "sgd", "adamw"
    
    # Training loop parameters
    num_epochs: int = 150
    patience: int = 20
    early_stopping: bool = True
    
    # Loss function
    loss_function: str = "bce"  # "bce", "focal", "weighted_bce"
    pos_weight: Optional[float] = None
    
    # Scheduler parameters
    scheduler_type: str = "reduce_lr_on_plateau"  # "cosine", "step", "none"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Validation parameters
    validation_frequency: int = 1
    save_best_model: bool = True
    save_last_model: bool = True
    
    # Logging parameters
    log_frequency: int = 10
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensemble specific parameters
    num_probes: int = 3
    probe_indices: Optional[List[int]] = None
    
    # Hyperparameter optimization
    use_optuna: bool = False
    optuna_n_trials: int = 20
    optuna_timeout: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate that all parameters are within acceptable ranges."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        
        if self.scheduler_factor <= 0 or self.scheduler_factor >= 1:
            raise ValueError("scheduler_factor must be between 0 and 1")
        
        if self.scheduler_min_lr <= 0:
            raise ValueError("scheduler_min_lr must be positive")
        
        if self.num_probes <= 0:
            raise ValueError("num_probes must be positive")
        
        if self.optuna_n_trials <= 0:
            raise ValueError("optuna_n_trials must be positive")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer_type": self.optimizer_type,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "early_stopping": self.early_stopping,
            "loss_function": self.loss_function,
            "pos_weight": self.pos_weight,
            "scheduler_type": self.scheduler_type,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_min_lr": self.scheduler_min_lr,
            "validation_frequency": self.validation_frequency,
            "save_best_model": self.save_best_model,
            "save_last_model": self.save_last_model,
            "log_frequency": self.log_frequency,
            "tensorboard_logging": self.tensorboard_logging,
            "wandb_logging": self.wandb_logging,
            "device": self.device,
            "num_probes": self.num_probes,
            "probe_indices": self.probe_indices,
            "use_optuna": self.use_optuna,
            "optuna_n_trials": self.optuna_n_trials,
            "optuna_timeout": self.optuna_timeout,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"TrainingConfig(batch_size={self.batch_size}, lr={self.learning_rate}, " \
               f"epochs={self.num_epochs}, device={self.device})" 