#!/usr/bin/env python3
"""
Model Loading Utility for Ensemble Ploidy Classifier

This utility provides easy access to pre-trained models and helps users
load the best performing probes for their classification tasks.
"""

import os
import torch
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..models.dynamic_lenet import DynamicLeNet
from ..models.trainable_aggregator import TrainableAggregator
from ..config.model_config import ModelConfig

class PretrainedModelLoader:
    """
    Easy-to-use loader for pre-trained ploidy classification models.
    
    Example:
        loader = PretrainedModelLoader()
        
        # Load the best performing model
        model, config = loader.load_best_model()
        
        # Load top N models for ensemble
        models = loader.load_top_models(n=5)
        
        # Load a specific probe
        model = loader.load_probe_model(probe_id=229)
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Path to the models directory. If None, uses default.
        """
        if models_dir is None:
            # Default to organized_project/models relative to package
            package_dir = Path(__file__).parent.parent.parent
            self.models_dir = package_dir / "organized_project" / "models"
        else:
            self.models_dir = Path(models_dir)
            
        self.results_dir = self.models_dir.parent / "results"
        
    def get_model_rankings(self) -> pd.DataFrame:
        """
        Get the ranking of all available models by performance.
        
        Returns:
            DataFrame with probe rankings and performance metrics
        """
        rankings_file = self.results_dir / "probe_rankings_test.csv"
        if rankings_file.exists():
            return pd.read_csv(rankings_file)
        else:
            # Fallback: scan available models
            model_files = list(self.models_dir.glob("best_model_probe_*.pt"))
            return pd.DataFrame({
                'probe_idx': [self._extract_probe_id(f.name) for f in model_files],
                'model_file': [f.name for f in model_files]
            })
    
    def load_best_model(self) -> Tuple[DynamicLeNet, Dict]:
        """
        Load the best performing model.
        
        Returns:
            Tuple of (model, config_dict)
        """
        rankings = self.get_model_rankings()
        if len(rankings) == 0:
            raise ValueError("No trained models found!")
            
        # Get the best model
        if 'test_accuracy' in rankings.columns:
            best_probe = rankings.loc[rankings['test_accuracy'].idxmax(), 'probe_idx']
        else:
            best_probe = rankings.iloc[0]['probe_idx']
            
        return self.load_probe_model(best_probe)
    
    def load_top_models(self, n: int = 5) -> List[Tuple[DynamicLeNet, Dict]]:
        """
        Load the top N performing models.
        
        Args:
            n: Number of top models to load
            
        Returns:
            List of (model, config_dict) tuples
        """
        rankings = self.get_model_rankings()
        
        if 'test_accuracy' in rankings.columns:
            top_probes = rankings.nlargest(n, 'test_accuracy')['probe_idx'].tolist()
        else:
            top_probes = rankings.head(n)['probe_idx'].tolist()
            
        models = []
        for probe_id in top_probes:
            try:
                model, config = self.load_probe_model(probe_id)
                models.append((model, config))
            except Exception as e:
                print(f"Warning: Could not load probe {probe_id}: {e}")
                continue
                
        return models
    
    def load_probe_model(self, probe_id: int) -> Tuple[DynamicLeNet, Dict]:
        """
        Load a specific probe model.
        
        Args:
            probe_id: ID of the probe to load
            
        Returns:
            Tuple of (model, config_dict)
        """
        # Find the model file
        model_files = list(self.models_dir.glob(f"best_model_probe_{probe_id}*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model found for probe {probe_id}")
            
        model_file = model_files[0]  # Take the first match
        
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_file, map_location=device)
        
        # Extract configuration from model if possible
        config = self._extract_model_config(model)
        
        return model, config
    
    def load_ensemble_models(self, probe_ids: List[int]) -> List[DynamicLeNet]:
        """
        Load multiple models for ensemble prediction.
        
        Args:
            probe_ids: List of probe IDs to load
            
        Returns:
            List of loaded models
        """
        models = []
        for probe_id in probe_ids:
            try:
                model, _ = self.load_probe_model(probe_id)
                models.append(model)
            except Exception as e:
                print(f"Warning: Could not load probe {probe_id}: {e}")
                continue
        return models
    
    def list_available_models(self) -> List[int]:
        """
        List all available probe model IDs.
        
        Returns:
            List of available probe IDs
        """
        model_files = list(self.models_dir.glob("best_model_probe_*.pt"))
        return sorted([self._extract_probe_id(f.name) for f in model_files])
    
    def _extract_probe_id(self, filename: str) -> int:
        """Extract probe ID from filename."""
        import re
        match = re.search(r'probe_(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    def _extract_model_config(self, model: DynamicLeNet) -> Dict:
        """Extract configuration from a loaded model."""
        # This is a simplified config extraction
        # In practice, you might save configs alongside models
        return {
            "model_type": "DynamicLeNet",
            "architecture": str(model),
            "device": str(next(model.parameters()).device)
        }

# Convenience function for quick model loading
def load_best_ploidy_model():
    """
    Quickly load the best performing ploidy classification model.
    
    Returns:
        Tuple of (model, config)
    """
    loader = PretrainedModelLoader()
    return loader.load_best_model()

def load_top_ploidy_models(n=5):
    """
    Quickly load the top N ploidy classification models.
    
    Args:
        n: Number of models to load
        
    Returns:
        List of (model, config) tuples
    """
    loader = PretrainedModelLoader()
    return loader.load_top_models(n)
