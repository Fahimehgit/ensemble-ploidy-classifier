# Ensemble Ploidy Classifier

A modular deep learning system for ploidy classification using ensemble methods with multiple probes and trainable aggregation mechanisms.

##  Overview

This package implements an ensemble classification system that combines multiple deep learning models (probes) with a learnable aggregation mechanism. The system is designed for binary classification tasks, particularly in biological contexts where different aspects of the data need to be captured by different feature extractors.

**Key Features:**
- **867 Pre-trained Probe Models** for immediate use
- **Fixed Indices Methodology** ensuring zero data leakage
- **Modular Architecture** for easy extension and customization
- **Rigorous Evaluation** on consistent 66-sample test set
- **Species Adaptable** with clear guidelines for new datasets

## Architecture

### Core Components

1. **DynamicLeNet**: A flexible CNN architecture that serves as the feature extractor
2. **ClassificationMLP**: A multi-layer perceptron for final classification
3. **TrainableAggregator**: A learnable ensemble mechanism that optimally combines embeddings from multiple probes
4. **DimensionalityReductionMLP**: Optional dimensionality reduction for embeddings

### Model Workflow

```
Input Data → Multiple DynamicLeNet Models → Embeddings → TrainableAggregator → ClassificationMLP → Prediction
```

##  Installation

### From Source

```bash
git clone https://github.com/fahimehgit/ensemble-ploidy-classifier.git
cd ensemble-ploidy-classifier
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

##  Quick Start

### Basic Usage

```python
from ensemble_ploidy_classifier import EnsemblePloidyClassifier
from ensemble_ploidy_classifier.config import ModelConfig

# Initialize configuration
config = ModelConfig(
    input_channels=1,
    input_size=59136,
    num_layers=3,
    num_filters=[64, 128, 256],
    kernel_sizes=[3, 3, 3],
    dropout_rate=0.5,
    activation_fn='relu',
    pool_type='max',
    embedding_dim=700
)

# Create and train the ensemble classifier
classifier = EnsemblePloidyClassifier(config)
classifier.train(train_data, validation_data)
predictions = classifier.predict(test_data)
```

### Advanced Usage with Custom Probes

```python
from ensemble_ploidy_classifier.models import DynamicLeNet, ClassificationMLP, TrainableAggregator
from ensemble_ploidy_classifier.training import EnsembleTrainer

# Create individual probe models
probe_models = []
for i in range(num_probes):
    model = DynamicLeNet(
        input_channels=1,
        input_size=59136,
        num_layers=3,
        num_filters=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        dropout_rate=0.5,
        activation_fn='relu',
        pool_type='max',
        embedding_dim=700
    )
    probe_models.append(model)

# Create ensemble components
classifier = ClassificationMLP(input_dim=700, hidden_dim1=512, hidden_dim2=256)
aggregator = TrainableAggregator(num_probes=len(probe_models), input_dim=700)

# Train the ensemble
trainer = EnsembleTrainer(probe_models, classifier, aggregator)
trainer.train(train_loaders, val_loader, epochs=150, learning_rate=1e-3)
```

##  Project Structure

```
ensemble_ploidy_classifier/
├── src/ensemble_ploidy_classifier/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dynamic_lenet.py
│   │   ├── classification_mlp.py
│   │   ├── trainable_aggregator.py
│   │   └── dimensionality_reduction.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── logging.py
│   └── config/
│       ├── __init__.py
│       ├── model_config.py
│       └── training_config.py
├── tests/
├── docs/
├── examples/
├── scripts/
├── pyproject.toml
└── README.md
```

## Configuration

```python
from ensemble_ploidy_classifier.config import ModelConfig, TrainingConfig

# Model configuration
model_config = ModelConfig(
    input_channels=1,
    input_size=59136,
    num_layers=3,
    num_filters=[64, 128, 256],
    kernel_sizes=[3, 3, 3],
    dropout_rate=0.5,
    activation_fn='relu',
    pool_type='max',
    embedding_dim=700
)

# Training configuration
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_epochs=150,
    patience=20,
    device='cuda'
)
```

## Key Features

### 1. Dynamic Architecture
- Configurable number of convolutional layers
- Flexible activation functions (ReLU, LeakyReLU, ELU)
- Adjustable pooling strategies (Max, Average)
- Customizable fully connected layers

### 2. Ensemble Learning
- Multiple probe models capture different data aspects
- Learnable aggregation weights via softmax
- End-to-end training of the entire ensemble

### 3. Professional Features
- Comprehensive logging and monitoring
- Hyperparameter optimization with Optuna
- Model checkpointing and early stopping
- Extensive testing and validation
- Type hints and documentation

### 4. Performance Optimization
- GPU acceleration support
- Memory-efficient training
- Batch processing capabilities
- Parallel data loading

## Data Integrity & Methodology

### Fixed Indices for Rigorous Evaluation

This package implements a **fixed indices approach** to ensure consistent, leak-free data splits across all experiments:

#### The 4-Split System
1. **Training Set** (~516 samples): Used for training each probe model
2. **Validation Set** (~64 samples): Used for hyperparameter tuning and early stopping  
3. **Test Set** (33 fixed samples): Consistent test evaluation across all models
4. **True Validation Set** (33 fixed samples): Additional validation samples

#### Why This Matters
- **Zero Data Leakage**: Same samples always in same splits across all 867 probe models + Prevents accidental data contamination
- **Fair Comparison**: All models evaluated on identical test samples
- **Reproducible Results**: Fixed indices ensure consistent evaluation

#### Using Fixed Indices
```python
# The package automatically respects fixed indices
from ensemble_ploidy_classifier.utils.model_loader import PretrainedModelLoader

loader = PretrainedModelLoader()
rankings = loader.get_model_rankings()  # Based on fixed 66-sample test set
```

For complete details, see [`FIXED_INDICES_DOCUMENTATION.md`](FIXED_INDICES_DOCUMENTATION.md).

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=ensemble_ploidy_classifier
```

##  Model Performance

The ensemble system typically achieves:
- **AUC-ROC**: 0.85-0.95
- **Accuracy**: 80-90%
- **Robustness**: Improved generalization through ensemble diversity
