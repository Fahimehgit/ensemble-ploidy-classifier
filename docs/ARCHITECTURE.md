# Ensemble Ploidy Classifier - Architecture Documentation

## Overview

The Ensemble Ploidy Classifier is a sophisticated deep learning system designed for binary classification tasks, particularly in biological contexts. It implements an ensemble approach using multiple probe models with a learnable aggregation mechanism.

## System Architecture

### High-Level Design

```
Input Data → Multiple Probe Models → Embeddings → Trainable Aggregator → Classification MLP → Prediction
```

### Core Components

#### 1. DynamicLeNet (Feature Extractor)

**Purpose**: Flexible CNN architecture that serves as the feature extractor for each probe.

**Key Features**:
- Configurable number of convolutional layers
- Flexible activation functions (ReLU, LeakyReLU, ELU)
- Adjustable pooling strategies (Max, Average)
- Adaptive pooling for variable input sizes
- Dropout regularization

**Architecture**:
```
Input → Conv2D → Activation → Pooling → ... → Adaptive Pool → Flatten → FC Layers → Output
```

**Parameters**:
- `input_channels`: Number of input channels (default: 1)
- `input_size`: Size of input (default: 59136)
- `num_layers`: Number of convolutional layers (default: 3)
- `num_filters`: List of filter counts for each layer (default: [64, 128, 256])
- `kernel_sizes`: List of kernel sizes for each layer (default: [3, 3, 3])
- `dropout_rate`: Dropout rate for regularization (default: 0.5)
- `activation_fn`: Activation function type (default: "relu")
- `pool_type`: Pooling type (default: "max")
- `fc_layer_sizes`: List of fully connected layer sizes
- `num_classes`: Number of output classes (default: 1)

#### 2. ClassificationMLP (Classifier)

**Purpose**: Multi-layer perceptron for final classification of aggregated embeddings.

**Architecture**:
```
Input → FC1 → ReLU → Dropout → FC2 → ReLU → Dropout → FC3 → Sigmoid → Output
```

**Parameters**:
- `input_dim`: Dimension of input embeddings (default: 700)
- `hidden_dim1`: Size of first hidden layer (default: 512)
- `hidden_dim2`: Size of second hidden layer (default: 256)
- `dropout_rate1`: Dropout rate for first layer (default: 0.3)
- `dropout_rate2`: Dropout rate for second layer (default: 0.3)
- `num_classes`: Number of output classes (default: 1)

#### 3. TrainableAggregator (Ensemble Mechanism)

**Purpose**: Learnable ensemble mechanism that optimally combines embeddings from multiple probes.

**Key Features**:
- Learnable weights for each probe
- Softmax normalization to ensure weights sum to 1
- Interpretable as probability weights
- End-to-end training with the ensemble

**Architecture**:
```
Probe Embeddings → Weighted Sum → Aggregated Embedding
```

**Parameters**:
- `num_probes`: Number of probes to aggregate
- `input_dim`: Dimension of input embeddings from each probe
- `device`: Device to place the model on

#### 4. DimensionalityReductionMLP (Optional)

**Purpose**: Optional dimensionality reduction for embeddings.

**Use Cases**:
- Visualization of high-dimensional embeddings
- Efficiency improvements
- Preprocessing for other models

## Training Pipeline

### Phase 1: Individual Probe Training

1. **Model Creation**: Create multiple DynamicLeNet models with identical architecture
2. **Training**: Train each probe independently on the same dataset
3. **Embedding Generation**: Extract embeddings from each trained probe

### Phase 2: Ensemble Training

1. **Embedding Aggregation**: Use TrainableAggregator to combine embeddings
2. **Classification**: Train ClassificationMLP on aggregated embeddings
3. **End-to-End Training**: Jointly optimize aggregator and classifier

### Training Configuration

#### Model Configuration
```python
model_config = ModelConfig(
    input_channels=1,
    input_size=59136,
    num_layers=3,
    num_filters=[64, 128, 256],
    kernel_sizes=[3, 3, 3],
    dropout_rate=0.5,
    activation_fn="relu",
    pool_type="max",
    embedding_dim=700,
    hidden_dim1=512,
    hidden_dim2=256,
    dropout_rate1=0.3,
    dropout_rate2=0.3,
)
```

#### Training Configuration
```python
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_epochs=150,
    patience=20,
    device="cuda",
    early_stopping=True,
    validation_frequency=1,
)
```

## Data Flow

### Input Data
- **Format**: NumPy arrays of shape (num_samples, channels, height, width)
- **Type**: Float32 tensors
- **Preprocessing**: Normalized to [0, 1] range

### Embedding Generation
- **Probe Output**: Embeddings of shape (num_samples, embedding_dim)
- **Aggregation**: Weighted combination of probe embeddings
- **Final Output**: Binary classification probabilities

### Output
- **Format**: NumPy arrays of shape (num_samples,)
- **Type**: Float32 probabilities in [0, 1] range
- **Threshold**: Configurable threshold for binary classification

## Model Persistence

### Saved Components
1. **Probe Models**: Individual DynamicLeNet models
2. **Classifier**: ClassificationMLP model
3. **Aggregator**: TrainableAggregator model
4. **Configurations**: Model and training configurations
5. **Training History**: Loss and metric curves

### File Structure
```
models/
├── probe_0.pth
├── probe_1.pth
├── probe_2.pth
├── classifier.pth
├── aggregator.pth
├── model_config.json
├── training_config.json
└── training_history.json
```

## Performance Optimization

### Memory Management
- **Batch Processing**: Configurable batch sizes
- **Gradient Accumulation**: For large models
- **Mixed Precision**: Optional FP16 training

### Computational Efficiency
- **GPU Acceleration**: CUDA support
- **Parallel Processing**: Multi-worker data loading
- **Model Optimization**: TorchScript compilation support

### Scalability
- **Modular Design**: Easy to add/remove probes
- **Configurable Architecture**: Flexible model sizes
- **Distributed Training**: Support for multiple GPUs

## Evaluation Metrics

### Primary Metrics
- **AUC-ROC**: Area under the ROC curve
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate
- **F1-Score**: Harmonic mean of precision and recall

### Secondary Metrics
- **Specificity**: True negative rate
- **Balanced Accuracy**: Average of sensitivity and specificity
- **Confidence Intervals**: Bootstrap-based uncertainty estimation

## Hyperparameter Optimization

### Optuna Integration
- **Objective Function**: Maximize validation AUC-ROC
- **Search Space**: Configurable parameter ranges
- **Pruning**: Early stopping for poor trials
- **Parallelization**: Multi-process optimization

### Optimized Parameters
- **Learning Rate**: Log-uniform distribution
- **Batch Size**: Categorical choices
- **Dropout Rates**: Uniform distribution
- **Hidden Dimensions**: Integer ranges
- **Number of Layers**: Integer ranges

## Monitoring and Logging

### Training Monitoring
- **Real-time Metrics**: Live loss and accuracy curves
- **Validation Tracking**: Regular validation evaluation
- **Early Stopping**: Automatic training termination
- **Model Checkpointing**: Best model preservation

### Logging System
- **Structured Logging**: JSON-formatted logs
- **Log Rotation**: Automatic log file management
- **Multiple Outputs**: Console and file logging
- **Log Levels**: Configurable verbosity

## Error Handling

### Input Validation
- **Data Type Checking**: Automatic type conversion
- **Shape Validation**: Dimension compatibility
- **Range Validation**: Value bounds checking
- **Missing Data**: Graceful handling of NaN values

### Model Validation
- **Architecture Validation**: Parameter consistency
- **Device Compatibility**: Automatic device placement
- **Memory Checking**: Available memory validation
- **Gradient Checking**: Numerical stability

## Security Considerations

### Data Privacy
- **Local Processing**: No external data transmission
- **Secure Storage**: Encrypted model files
- **Access Control**: File permission management
- **Audit Logging**: Training activity tracking

### Model Security
- **Input Sanitization**: Malicious input prevention
- **Model Verification**: Integrity checking
- **Version Control**: Model versioning
- **Backup Strategy**: Redundant model storage

## Future Enhancements

### Planned Features
1. **Multi-class Classification**: Extension beyond binary
2. **Attention Mechanisms**: Self-attention for embeddings
3. **Graph Neural Networks**: Relational modeling
4. **Federated Learning**: Distributed training
5. **AutoML Integration**: Automated architecture search

### Research Directions
1. **Interpretability**: Model explanation methods
2. **Uncertainty Quantification**: Bayesian approaches
3. **Adversarial Robustness**: Attack resistance
4. **Transfer Learning**: Pre-trained model adaptation
5. **Continual Learning**: Incremental model updates

## Conclusion

The Ensemble Ploidy Classifier represents a state-of-the-art approach to ensemble learning in deep neural networks. Its modular design, comprehensive configuration system, and robust training pipeline make it suitable for a wide range of binary classification tasks, particularly in biological and medical applications.

The system's key strengths include:
- **Flexibility**: Configurable architecture for different use cases
- **Robustness**: Ensemble approach reduces overfitting
- **Interpretability**: Learnable aggregation weights provide insights
- **Scalability**: Modular design supports easy extension
- **Reliability**: Comprehensive testing and validation framework

