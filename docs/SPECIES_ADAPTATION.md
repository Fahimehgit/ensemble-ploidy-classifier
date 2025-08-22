# Adding New Species to the Ensemble Ploidy Classifier

## 📋 Overview

This guide shows you how to adapt the ensemble ploidy classifier to work with new species or datasets.

## 🔧 Step-by-Step Process

### 1. Prepare Your Data

Your data should be in JSON format with the following structure:

```json
{
    "species_1": {
        "reference_tokens": [...],  # Reference sequence (list of numbers)
        "auxiliary_tokens": [[...], [...], ...],  # List of auxiliary sequences
        "label": 0  # 0 for diploid, 1 for polyploid
    },
    "species_2": {
        ...
    }
}
```

### 2. Update Configuration

```python
from ensemble_ploidy_classifier.config import ModelConfig, TrainingConfig

# Adjust these parameters based on your data
model_config = ModelConfig(
    input_channels=1,
    input_size=64*64,  # Adjust based on your sequence length
    num_layers=3,
    num_filters=[64, 128, 256],
    kernel_sizes=[3, 3, 3],
    dropout_rate=0.3,  # Tune for your dataset
    activation_fn='relu',
    pool_type='max',
    embedding_dim=500,  # Adjust based on complexity
    num_classes=1
)

training_config = TrainingConfig(
    batch_size=16,  # Adjust based on memory
    learning_rate=0.001,
    num_epochs=100,
    patience=15,
    early_stopping=True
)
```

### 3. Train New Probes

```python
from ensemble_ploidy_classifier import EnsemblePloidyClassifier

# Initialize classifier
classifier = EnsemblePloidyClassifier(model_config, training_config)

# Load your data
# your_train_data, your_val_data = load_your_data()

# Train the ensemble
classifier.train(your_train_data, your_val_data)

# Save the trained models
classifier.save_models("models/your_species/")
```

### 4. Fine-tune Existing Models

```python
from ensemble_ploidy_classifier.utils.model_loader import PretrainedModelLoader

# Load pre-trained models
loader = PretrainedModelLoader()
models = loader.load_top_models(n=5)

# Fine-tune on your data
for i, (model, config) in enumerate(models):
    # Set up fine-tuning (lower learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Fine-tune for fewer epochs
    fine_tune_epochs = 10
    
    # Your fine-tuning loop here
    # ...
```

### 5. Evaluate Performance

```python
# Test the adapted models
test_results = classifier.evaluate(your_test_data)

print(f"Accuracy: {test_results['accuracy']:.4f}")
print(f"AUC-ROC: {test_results['auc_roc']:.4f}")
```

## 💡 Tips for Success

1. **Start Small**: Begin with a subset of your data to test the pipeline
2. **Monitor Overfitting**: Use validation curves to tune hyperparameters
3. **Data Quality**: Ensure your sequences are properly formatted and quality-controlled
4. **Ensemble Size**: Start with 3-5 probes, then expand based on performance
5. **Cross-Validation**: Use k-fold validation for robust performance estimates

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or input dimensions
2. **Poor Performance**: Try different hyperparameters or more data
3. **Model Loading Errors**: Check file paths and model compatibility

### Contact

For support, please open an issue on GitHub or contact the maintainers.
