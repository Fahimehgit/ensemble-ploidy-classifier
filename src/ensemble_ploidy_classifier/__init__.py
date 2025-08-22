"""
Ensemble Ploidy Classifier

A professional deep learning system for ploidy classification using ensemble methods.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

# Core components
from .ensemble_classifier import EnsemblePloidyClassifier
from .models import DynamicLeNet, ClassificationMLP, TrainableAggregator
from .config import ModelConfig, TrainingConfig
from .utils.model_loader import PretrainedModelLoader, load_best_ploidy_model

# Convenience imports
from .training import EnsembleTrainer
from .data import CustomDataset, EmbeddingDataset

__all__ = [
    "EnsemblePloidyClassifier",
    "DynamicLeNet", 
    "ClassificationMLP",
    "TrainableAggregator",
    "ModelConfig",
    "TrainingConfig", 
    "PretrainedModelLoader",
    "load_best_ploidy_model",
    "EnsembleTrainer",
    "CustomDataset",
    "EmbeddingDataset"
]
