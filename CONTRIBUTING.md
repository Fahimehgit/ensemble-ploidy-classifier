# Contributing to Ensemble Ploidy Classifier

Thank you for your interest in contributing to the Ensemble Ploidy Classifier! This document provides guidelines for contributing to this project.

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

## Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the issue template
- Include minimal reproducible examples
- Provide system information (OS, Python version, package versions)

### ✨ Feature Requests
- Describe the feature and its use case
- Explain how it fits with the project's goals
- Consider contributing the implementation

### 📝 Documentation
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve docstrings

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

## Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run these before committing:
```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Type checking
mypy src/

# Run tests
pytest
```

### Commit Messages

Use conventional commit format:
```
type(scope): description

Examples:
feat(models): add new aggregation method
fix(data): resolve memory leak in dataloader
docs(readme): update installation instructions
test(models): add tests for DynamicLeNet
```

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   pytest tests/
   python examples/basic_usage.py  # Test examples work
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(scope): your description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

### Pull Request Requirements

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New code has appropriate tests
- [ ] Documentation updated if needed
- [ ] Examples still work
- [ ] No breaking changes (or clearly documented)

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ensemble_ploidy_classifier

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Writing Tests
- Place tests in the `tests/` directory
- Mirror the source structure: `src/package/module.py` → `tests/test_module.py`
- Use descriptive test names: `test_model_loads_correctly_with_valid_config`
- Test edge cases and error conditions

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

## Questions?

- Open an issue for questions about contributing
- Check existing issues for similar questions
- Review the documentation first

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Assume good intentions

Thank you for contributing! 🎉 