"""
Ensemble Ploidy Classifier

A professional deep learning system for ploidy classification using ensemble methods.
"""

__version__ = "0.1.0"
__author__ = "Fahimeh Rahimi"
__email__ = "fahimeh.rahimi@ufl.edu"

# Core imports with error handling
try:
from .ensemble_classifier import EnsemblePloidyClassifier
except ImportError as e:
    print(f"Warning: Could not import EnsemblePloidyClassifier: {e}")
    EnsemblePloidyClassifier = None

try:
    from .models.dynamic_lenet import DynamicLeNet
    from .models.classification_mlp import ClassificationMLP
    from .models.trainable_aggregator import TrainableAggregator
    from .models.dimensionality_reduction import DimensionalityReductionMLP
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    DynamicLeNet = ClassificationMLP = TrainableAggregator = DimensionalityReductionMLP = None

try:
    from .config.model_config import ModelConfig
    from .config.training_config import TrainingConfig
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    ModelConfig = TrainingConfig = None

try:
from .utils.model_loader import PretrainedModelLoader, load_best_ploidy_model
except ImportError as e:
    print(f"Warning: Could not import model loader: {e}")
    PretrainedModelLoader = load_best_ploidy_model = None

try:
    from .training.trainer import EnsembleTrainer
except ImportError as e:
    print(f"Warning: Could not import trainer: {e}")
    EnsembleTrainer = None

try:
    from .data.dataset import CustomDataset, EmbeddingDataset
except ImportError as e:
    print(f"Warning: Could not import data classes: {e}")
    CustomDataset = EmbeddingDataset = None

# Only include successfully imported components in __all__
__all__ = []
if EnsemblePloidyClassifier is not None:
    __all__.append("EnsemblePloidyClassifier")
if DynamicLeNet is not None:
    __all__.extend(["DynamicLeNet", "ClassificationMLP", "TrainableAggregator", "DimensionalityReductionMLP"])
if ModelConfig is not None:
    __all__.extend(["ModelConfig", "TrainingConfig"])
if PretrainedModelLoader is not None:
    __all__.extend(["PretrainedModelLoader", "load_best_ploidy_model"])
if EnsembleTrainer is not None:
    __all__.append("EnsembleTrainer")
if CustomDataset is not None:
    __all__.extend(["CustomDataset", "EmbeddingDataset"])
