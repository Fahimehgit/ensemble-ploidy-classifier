"""
Visualization utilities for the Ensemble Ploidy Classifier.

This module contains functions for plotting training curves and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Plot training curves for loss and metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Training Curves", fontsize=16)
    
    # Loss curves
    if "train_loss" in history and "val_loss" in history:
        axes[0, 0].plot(history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(history["val_loss"], label="Validation Loss", color="red")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # AUC curves
    if "train_auc" in history and "val_auc" in history:
        axes[0, 1].plot(history["train_auc"], label="Train AUC", color="blue")
        axes[0, 1].plot(history["val_auc"], label="Validation AUC", color="red")
        axes[0, 1].set_title("AUC-ROC")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("AUC-ROC")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy curves
    if "train_accuracy" in history and "val_accuracy" in history:
        axes[1, 0].plot(history["train_accuracy"], label="Train Accuracy", color="blue")
        axes[1, 0].plot(history["val_accuracy"], label="Validation Accuracy", color="red")
        axes[1, 0].set_title("Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate curve (if available)
    if "learning_rate" in history:
        axes[1, 1].plot(history["learning_rate"], label="Learning Rate", color="green")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_ensemble_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot comparison of metrics across different probes and ensemble.
    
    Args:
        metrics: Dictionary containing metrics for each probe and ensemble
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    # Prepare data for plotting
    metric_names = ["auc_roc", "accuracy", "precision", "recall", "f1_score"]
    model_names = list(metrics.keys())
    
    # Create DataFrame for easier plotting
    data = []
    for model_name in model_names:
        for metric_name in metric_names:
            if metric_name in metrics[model_name]:
                data.append({
                    "Model": model_name,
                    "Metric": metric_name.replace("_", " ").title(),
                    "Value": metrics[model_name][metric_name]
                })
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot for all metrics
    pivot_df = df.pivot(index="Model", columns="Metric", values="Value")
    pivot_df.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].set_title("Metrics Comparison")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Score")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Heatmap for detailed comparison
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Metrics Heatmap")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Model")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_roc_curves(
    y_true: List[float],
    y_pred_probes: List[List[float]],
    y_pred_ensemble: List[float],
    probe_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot ROC curves for all probes and ensemble.
    
    Args:
        y_true: True labels
        y_pred_probes: List of predictions from each probe
        y_pred_ensemble: Predictions from ensemble
        probe_names: Optional names for probes
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curves for each probe
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_pred_probes)))
    
    for i, y_pred in enumerate(y_pred_probes):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = np.trapz(tpr, fpr)
        
        probe_name = probe_names[i] if probe_names else f"Probe {i}"
        plt.plot(fpr, tpr, color=colors[i], label=f"{probe_name} (AUC = {auc:.3f})")
    
    # Plot ROC curve for ensemble
    fpr, tpr, _ = roc_curve(y_true, y_pred_ensemble)
    auc = np.trapz(tpr, fpr)
    plt.plot(fpr, tpr, color="black", linewidth=2, label=f"Ensemble (AUC = {auc:.3f})")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrices(
    y_true: List[float],
    y_pred_probes: List[List[float]],
    y_pred_ensemble: List[float],
    probe_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Plot confusion matrices for all probes and ensemble.
    
    Args:
        y_true: True labels
        y_pred_probes: List of predictions from each probe
        y_pred_ensemble: Predictions from ensemble
        probe_names: Optional names for probes
        threshold: Threshold for converting probabilities to binary predictions
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate number of subplots
    n_models = len(y_pred_probes) + 1  # +1 for ensemble
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot confusion matrices for each probe
    for i, y_pred in enumerate(y_pred_probes):
        row = i // n_cols
        col = i % n_cols
        
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        probe_name = probe_names[i] if probe_names else f"Probe {i}"
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[row, col])
        axes[row, col].set_title(f"{probe_name} Confusion Matrix")
        axes[row, col].set_xlabel("Predicted")
        axes[row, col].set_ylabel("Actual")
    
    # Plot confusion matrix for ensemble
    row = len(y_pred_probes) // n_cols
    col = len(y_pred_probes) % n_cols
    
    y_pred_ensemble_binary = (np.array(y_pred_ensemble) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_ensemble_binary)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[row, col])
    axes[row, col].set_title("Ensemble Confusion Matrix")
    axes[row, col].set_xlabel("Predicted")
    axes[row, col].set_ylabel("Actual")
    
    # Hide empty subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_aggregation_weights(
    aggregator,
    probe_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot the learned aggregation weights.
    
    Args:
        aggregator: TrainableAggregator instance
        probe_names: Optional names for probes
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    weights = aggregator.get_probe_importance().numpy()
    
    plt.figure(figsize=figsize)
    
    if probe_names is None:
        probe_names = [f"Probe {i}" for i in range(len(weights))]
    
    bars = plt.bar(range(len(weights)), weights, color="skyblue", edgecolor="navy")
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{weight:.3f}", ha="center", va="bottom")
    
    plt.xlabel("Probe")
    plt.ylabel("Weight")
    plt.title("Learned Aggregation Weights")
    plt.xticks(range(len(weights)), probe_names, rotation=45)
    plt.ylim(0, max(weights) * 1.1)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show() 