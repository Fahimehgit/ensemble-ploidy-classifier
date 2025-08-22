"""
Metrics utilities for the Ensemble Ploidy Classifier.

This module contains functions for calculating various evaluation metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Union


def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary containing various metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    # AUC-ROC
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics["auc_roc"] = 0.0
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
    
    # Additional metrics from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Precision
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall (Sensitivity)
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # F1-Score
    metrics["f1_score"] = (
        2 * (metrics["precision"] * metrics["recall"]) / 
        (metrics["precision"] + metrics["recall"])
        if (metrics["precision"] + metrics["recall"]) > 0 else 0.0
    )
    
    # Balanced Accuracy
    metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
    
    return metrics


def calculate_ensemble_metrics(
    y_true: Union[List, np.ndarray],
    y_pred_probes: List[Union[List, np.ndarray]],
    y_pred_ensemble: Union[List, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for individual probes and ensemble.
    
    Args:
        y_true: True labels
        y_pred_probes: List of predictions from individual probes
        y_pred_ensemble: Predictions from ensemble
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary containing metrics for each probe and ensemble
    """
    results = {}
    
    # Calculate metrics for each probe
    for i, y_pred in enumerate(y_pred_probes):
        results[f"probe_{i}"] = calculate_metrics(y_true, y_pred, threshold)
    
    # Calculate metrics for ensemble
    results["ensemble"] = calculate_metrics(y_true, y_pred_ensemble, threshold)
    
    return results


def find_optimal_threshold(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    metric: str = "f1_score",
) -> Tuple[float, float]:
    """
    Find the optimal threshold for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ("f1_score", "precision", "recall", "balanced_accuracy")
        
    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_threshold = 0.5
    best_metric = 0.0
    
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_pred, threshold)
        current_metric = metrics.get(metric, 0.0)
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
    
    return best_threshold, best_metric


def print_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    threshold: float = 0.5,
    target_names: List[str] = None,
) -> None:
    """
    Print a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Threshold for converting probabilities to binary predictions
        target_names: Names for the classes
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    if target_names is None:
        target_names = ["Class 0", "Class 1"]
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=target_names))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Class 0  Class 1")
    print(f"Actual Class 0    {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"       Class 1    {cm[1, 0]:6d}  {cm[1, 1]:6d}")


def calculate_confidence_intervals(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for metrics using bootstrap.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Dictionary containing confidence intervals for each metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    # Bootstrap samples
    bootstrap_metrics = {
        "auc_roc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        # Calculate metrics for this sample
        metrics = calculate_metrics(sample_true, sample_pred)
        
        for metric_name in bootstrap_metrics:
            bootstrap_metrics[metric_name].append(metrics[metric_name])
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    confidence_intervals = {}
    for metric_name, values in bootstrap_metrics.items():
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        confidence_intervals[metric_name] = (lower, upper)
    
    return confidence_intervals 