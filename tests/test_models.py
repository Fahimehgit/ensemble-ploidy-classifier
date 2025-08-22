"""
Tests for the Ensemble Ploidy Classifier models.

This module contains unit tests for all model components.
"""

import pytest
import torch
import numpy as np

from ensemble_ploidy_classifier.models import (
    DynamicLeNet,
    ClassificationMLP,
    TrainableAggregator,
    DimensionalityReductionMLP,
)
from ensemble_ploidy_classifier.config import ModelConfig, TrainingConfig


class TestDynamicLeNet:
    """Test cases for DynamicLeNet model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DynamicLeNet(
            input_channels=1,
            input_size=59136,
            num_layers=3,
            num_filters=[64, 128, 256],
            kernel_sizes=[3, 3, 3],
            dropout_rate=0.5,
            activation_fn="relu",
            pool_type="max",
            fc_layer_sizes=[700],
            num_classes=1,
        )
        
        assert model is not None
        assert isinstance(model, DynamicLeNet)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = DynamicLeNet(
            input_channels=1,
            input_size=59136,
            num_layers=3,
            num_filters=[64, 128, 256],
            kernel_sizes=[3, 3, 3],
            dropout_rate=0.5,
            activation_fn="relu",
            pool_type="max",
            fc_layer_sizes=[700],
            num_classes=1,
        )
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, int(np.sqrt(59136)), int(np.sqrt(59136)))
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError):
            DynamicLeNet(
                input_channels=1,
                input_size=59136,
                num_layers=0,  # Invalid
                num_filters=[64, 128, 256],
                kernel_sizes=[3, 3, 3],
                dropout_rate=0.5,
                activation_fn="relu",
                pool_type="max",
                fc_layer_sizes=[700],
                num_classes=1,
            )
        
        with pytest.raises(ValueError):
            DynamicLeNet(
                input_channels=1,
                input_size=59136,
                num_layers=3,
                num_filters=[64, 128],  # Wrong length
                kernel_sizes=[3, 3, 3],
                dropout_rate=0.5,
                activation_fn="relu",
                pool_type="max",
                fc_layer_sizes=[700],
                num_classes=1,
            )


class TestClassificationMLP:
    """Test cases for ClassificationMLP model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ClassificationMLP(
            input_dim=700,
            hidden_dim1=512,
            hidden_dim2=256,
            dropout_rate1=0.3,
            dropout_rate2=0.3,
        )
        
        assert model is not None
        assert isinstance(model, ClassificationMLP)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = ClassificationMLP(
            input_dim=700,
            hidden_dim1=512,
            hidden_dim2=256,
            dropout_rate1=0.3,
            dropout_rate2=0.3,
        )
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 700)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
    
    def test_get_embedding(self):
        """Test getting embeddings from penultimate layer."""
        model = ClassificationMLP(
            input_dim=700,
            hidden_dim1=512,
            hidden_dim2=256,
            dropout_rate1=0.3,
            dropout_rate2=0.3,
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 700)
        
        embedding = model.get_embedding(input_tensor)
        
        # Check embedding shape
        assert embedding.shape == (batch_size, 256)  # hidden_dim2
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError):
            ClassificationMLP(
                input_dim=0,  # Invalid
                hidden_dim1=512,
                hidden_dim2=256,
                dropout_rate1=0.3,
                dropout_rate2=0.3,
            )
        
        with pytest.raises(ValueError):
            ClassificationMLP(
                input_dim=700,
                hidden_dim1=512,
                hidden_dim2=256,
                dropout_rate1=1.5,  # Invalid
                dropout_rate2=0.3,
            )


class TestTrainableAggregator:
    """Test cases for TrainableAggregator model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = TrainableAggregator(
            num_probes=3,
            input_dim=700,
            device="cpu",
        )
        
        assert model is not None
        assert isinstance(model, TrainableAggregator)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = TrainableAggregator(
            num_probes=3,
            input_dim=700,
            device="cpu",
        )
        
        # Create dummy embeddings
        batch_size = 4
        probe_embeddings = [
            torch.randn(batch_size, 700) for _ in range(3)
        ]
        
        # Forward pass
        output = model(probe_embeddings)
        
        # Check output shape
        assert output.shape == (batch_size, 700)
    
    def test_get_weights(self):
        """Test getting aggregation weights."""
        model = TrainableAggregator(
            num_probes=3,
            input_dim=700,
            device="cpu",
        )
        
        weights = model.get_weights()
        
        # Check weights shape and properties
        assert weights.shape == (3,)
        assert torch.all(weights >= 0)  # Non-negative
        assert torch.allclose(weights.sum(), torch.tensor(1.0))  # Sum to 1
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError):
            TrainableAggregator(
                num_probes=0,  # Invalid
                input_dim=700,
                device="cpu",
            )
        
        with pytest.raises(ValueError):
            TrainableAggregator(
                num_probes=3,
                input_dim=0,  # Invalid
                device="cpu",
            )


class TestDimensionalityReductionMLP:
    """Test cases for DimensionalityReductionMLP model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DimensionalityReductionMLP(
            input_dim=700,
            output_dim=100,
            hidden_dims=[512, 256],
            dropout_rate=0.3,
            activation="relu",
        )
        
        assert model is not None
        assert isinstance(model, DimensionalityReductionMLP)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = DimensionalityReductionMLP(
            input_dim=700,
            output_dim=100,
            hidden_dims=[512, 256],
            dropout_rate=0.3,
            activation="relu",
        )
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 700)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 100)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        model = DimensionalityReductionMLP(
            input_dim=700,
            output_dim=100,
        )
        
        ratio = model.get_compression_ratio()
        expected_ratio = 100 / 700
        
        assert ratio == expected_ratio
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError):
            DimensionalityReductionMLP(
                input_dim=0,  # Invalid
                output_dim=100,
            )
        
        with pytest.raises(ValueError):
            DimensionalityReductionMLP(
                input_dim=700,
                output_dim=0,  # Invalid
            )


class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_initialization(self):
        """Test configuration initialization."""
        config = ModelConfig()
        
        assert config is not None
        assert config.input_channels == 1
        assert config.input_size == 59136
    
    def test_validation(self):
        """Test configuration validation."""
        # Test invalid num_layers
        with pytest.raises(ValueError):
            ModelConfig(num_layers=0)
        
        # Test invalid dropout_rate
        with pytest.raises(ValueError):
            ModelConfig(dropout_rate=1.5)
        
        # Test mismatched filter lengths
        with pytest.raises(ValueError):
            ModelConfig(
                num_layers=3,
                num_filters=[64, 128],  # Wrong length
                kernel_sizes=[3, 3, 3],
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "input_channels" in config_dict
        assert "input_size" in config_dict


class TestTrainingConfig:
    """Test cases for TrainingConfig."""
    
    def test_initialization(self):
        """Test configuration initialization."""
        config = TrainingConfig()
        
        assert config is not None
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3
    
    def test_validation(self):
        """Test configuration validation."""
        # Test invalid batch_size
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)
        
        # Test invalid learning_rate
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)
        
        # Test invalid patience
        with pytest.raises(ValueError):
            TrainingConfig(patience=0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "batch_size" in config_dict
        assert "learning_rate" in config_dict


if __name__ == "__main__":
    pytest.main([__file__]) 