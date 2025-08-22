# Ensemble Ploidy Classifier

A professional deep learning system for ploidy classification using ensemble methods, featuring **384 trained probe models** and a **greedy algorithm** for optimal probe selection.

## 🚀 Quick Start

The **WorkingEnsembleClassifier** is the main class you should use - it loads your actual trained models and implements the greedy algorithm from your research:

```python
from ensemble_ploidy_classifier import WorkingEnsembleClassifier

# Initialize with your trained models and embeddings
ensemble = WorkingEnsembleClassifier(
    embeddings_dir="organized_project/embeddings/ReducedEmbeddings1",
    model_weights_dir="/path/to/your/ModelWeights"
)

# Run greedy probe selection (automatically finds best probe combination)
selected_probes = ensemble.greedy_probe_selection(max_probes=20)

# Train the final ensemble
ensemble.train_final_ensemble()

# Make predictions
predictions = ensemble.predict()
probabilities = ensemble.predict_proba()

# Evaluate performance
results = ensemble.evaluate()
print(f"AUC: {results['auc']:.4f}")
print(f"Accuracy: {results['accuracy']:.4f}")
```

## ✨ Key Features

- **384 Trained Probe Models**: Pre-trained MLP classifiers for each probe
- **Greedy Algorithm**: Automatically selects the best combination of probes
- **Reduced Embeddings**: Uses your 700-dimensional reduced embeddings
- **Production Ready**: Loads actual trained weights from your research
- **Easy to Use**: Simple API for training, prediction, and evaluation

## 📁 Project Structure

```
ensemble_ploidy_classifier/
├── src/ensemble_ploidy_classifier/
│   ├── working_ensemble.py          # 🎯 MAIN CLASS - WorkingEnsembleClassifier
│   ├── models/
│   │   └── mlp_classifier.py        # Individual probe MLP models
│   └── __init__.py
├── organized_project/
│   └── embeddings/
│       └── ReducedEmbeddings1/      # Your 384 probe embeddings
├── examples/
│   └── working_ensemble_example.py  # Complete working example
└── README.md
```

## 🔧 Installation

### From Source (Recommended)

```bash
git clone https://github.com/fahimehgit/ensemble-ploidy-classifier.git
cd ensemble-ploidy-classifier
pip install -e .
```

### From GitHub

```bash
pip install git+https://github.com/fahimehgit/ensemble-ploidy-classifier.git
```

## 📖 Usage Examples

### Basic Usage

```python
from ensemble_ploidy_classifier import WorkingEnsembleClassifier

# Initialize with your data directories
ensemble = WorkingEnsembleClassifier(
    embeddings_dir="organized_project/embeddings/ReducedEmbeddings1",
    model_weights_dir="/path/to/ModelWeights"
)

# The ensemble automatically finds the best probe combination
ensemble.greedy_probe_selection()
ensemble.train_final_ensemble()

# Make predictions
predictions = ensemble.predict()
```

### Individual Probe Models

```python
from ensemble_ploidy_classifier import ProbeModelLoader

# Load individual probe models
loader = ProbeModelLoader("/path/to/ModelWeights")
model = loader.load_probe_model(probe_idx=0)

# Use for individual predictions
predictions = model.predict(embeddings)
```

### Advanced Configuration

```python
# Customize greedy selection
selected_probes = ensemble.greedy_probe_selection(
    max_probes=15,        # Maximum probes to select
    patience=5,           # Stop after 5 probes without improvement
    cv_folds=5            # Cross-validation folds
)

# Save and load trained ensembles
ensemble.save_ensemble("my_ensemble.pkl")
ensemble.load_ensemble("my_ensemble.pkl")
```

## 🧠 How It Works

1. **Probe Models**: 384 individual MLP classifiers, each trained on a specific probe
2. **Greedy Selection**: Algorithmically finds the best combination of probes
3. **Ensemble Training**: Trains a final classifier on the selected probe predictions
4. **Prediction**: Combines predictions from all selected probes for final output

## 📊 Performance

The ensemble automatically selects the optimal combination of probes based on:
- Cross-validation performance
- AUC-ROC scores
- Greedy optimization strategy

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👩‍🔬 Research

This implementation is based on the research presented in the GreedyModel.ipynb notebook, featuring:
- Greedy algorithm for probe selection
- 384 trained MLP models
- Reduced embeddings (700 dimensions)
- Ensemble optimization

## 🆘 Support

If you encounter any issues:
1. Check the [examples](examples/) directory
2. Review the error messages carefully
3. Ensure your data directories are correctly configured
4. Open an issue on GitHub with detailed information

---

**Note**: This is the **working implementation** that loads your actual trained models and embeddings. It's not a prototype - it's the complete system ready for production use!
