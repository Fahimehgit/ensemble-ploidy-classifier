# Contributing to Ensemble Ploidy Classifier

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- PyTorch (CPU or GPU version)

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/fahimehgit/ensemble-ploidy-classifier.git
   cd ensemble-ploidy-classifier
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Project Structure

```
ensemble_ploidy_classifier/
├── src/ensemble_ploidy_classifier/  # Main package
│   ├── models/                      # Model architectures
│   ├── data/                        # Data handling
│   ├── training/                    # Training utilities
│   ├── utils/                       # Utility functions
│   └── config/                      # Configuration classes
├── tests/                           # Test suite
├── examples/                        # Usage examples
├── docs/                           # Documentation
└── scripts/                        # Utility scripts
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag
4. GitHub Actions will handle PyPI upload

