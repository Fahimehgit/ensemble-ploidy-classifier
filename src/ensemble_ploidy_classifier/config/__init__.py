"""
Configuration module for the Ensemble Ploidy Classifier.

This module contains configuration classes for model architecture,
training parameters, and system settings.
"""

from .model_config import ModelConfig
from .training_config import TrainingConfig

__all__ = ["ModelConfig", "TrainingConfig"] 