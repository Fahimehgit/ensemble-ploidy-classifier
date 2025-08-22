"""
Logging utilities for the Ensemble Ploidy Classifier.

This module contains functions for setting up logging and monitoring.
"""

import logging
import os
from datetime import datetime
from typing import Optional
from loguru import logger


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_file: Path to log file (if None, uses default naming)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
    """
    # Remove default handler
    logger.remove()
    
    # Create log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ensemble_ploidy_classifier_{timestamp}.log"
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Setup console logging
    if console_output:
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True,
        )
    
    # Setup file logging
    if file_output:
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )
    
    logger.info(f"Logging setup complete. Log file: {log_file}")


def log_training_start(
    model_config: dict,
    training_config: dict,
    num_probes: int,
) -> None:
    """
    Log the start of training with configuration details.
    
    Args:
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary
        num_probes: Number of probes in the ensemble
    """
    logger.info("=" * 60)
    logger.info("ENSEMBLE PLODY CLASSIFIER TRAINING STARTED")
    logger.info("=" * 60)
    
    logger.info(f"Number of probes: {num_probes}")
    logger.info(f"Device: {training_config.get('device', 'Not specified')}")
    logger.info(f"Batch size: {training_config.get('batch_size', 'Not specified')}")
    logger.info(f"Learning rate: {training_config.get('learning_rate', 'Not specified')}")
    logger.info(f"Number of epochs: {training_config.get('num_epochs', 'Not specified')}")
    
    logger.info("Model Configuration:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")


def log_epoch_results(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    train_auc: float,
    val_auc: float,
    train_accuracy: Optional[float] = None,
    val_accuracy: Optional[float] = None,
) -> None:
    """
    Log epoch training results.
    
    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss
        train_auc: Training AUC-ROC
        val_auc: Validation AUC-ROC
        train_accuracy: Training accuracy (optional)
        val_accuracy: Validation accuracy (optional)
    """
    accuracy_str = ""
    if train_accuracy is not None and val_accuracy is not None:
        accuracy_str = f", Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
    
    logger.info(
        f"Epoch {epoch}/{total_epochs} - "
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}"
        f"{accuracy_str}"
    )


def log_probe_training_start(probe_idx: int, total_probes: int) -> None:
    """
    Log the start of probe training.
    
    Args:
        probe_idx: Current probe index
        total_probes: Total number of probes
    """
    logger.info("-" * 40)
    logger.info(f"TRAINING PROBE {probe_idx + 1}/{total_probes}")
    logger.info("-" * 40)


def log_probe_training_complete(
    probe_idx: int,
    best_val_auc: float,
    training_time: float,
) -> None:
    """
    Log the completion of probe training.
    
    Args:
        probe_idx: Probe index
        best_val_auc: Best validation AUC achieved
        training_time: Total training time in seconds
    """
    logger.info(f"Probe {probe_idx} training completed:")
    logger.info(f"  Best validation AUC: {best_val_auc:.4f}")
    logger.info(f"  Training time: {training_time:.2f} seconds")


def log_ensemble_training_start() -> None:
    """Log the start of ensemble training."""
    logger.info("-" * 40)
    logger.info("TRAINING ENSEMBLE")
    logger.info("-" * 40)


def log_ensemble_training_complete(
    best_val_auc: float,
    training_time: float,
    aggregation_weights: list,
) -> None:
    """
    Log the completion of ensemble training.
    
    Args:
        best_val_auc: Best validation AUC achieved
        training_time: Total training time in seconds
        aggregation_weights: Learned aggregation weights
    """
    logger.info("Ensemble training completed:")
    logger.info(f"  Best validation AUC: {best_val_auc:.4f}")
    logger.info(f"  Training time: {training_time:.2f} seconds")
    logger.info(f"  Aggregation weights: {[f'{w:.3f}' for w in aggregation_weights]}")


def log_evaluation_results(
    test_metrics: dict,
    probe_metrics: dict,
    ensemble_metrics: dict,
) -> None:
    """
    Log evaluation results.
    
    Args:
        test_metrics: Overall test metrics
        probe_metrics: Individual probe metrics
        ensemble_metrics: Ensemble metrics
    """
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info("Test Set Performance:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nIndividual Probe Performance:")
    for probe_name, metrics in probe_metrics.items():
        logger.info(f"  {probe_name}:")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value:.4f}")
    
    logger.info("\nEnsemble Performance:")
    for metric, value in ensemble_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


def log_model_save(save_path: str) -> None:
    """
    Log model saving.
    
    Args:
        save_path: Path where model was saved
    """
    logger.info(f"Model saved to: {save_path}")


def log_model_load(load_path: str) -> None:
    """
    Log model loading.
    
    Args:
        load_path: Path from where model was loaded
    """
    logger.info(f"Model loaded from: {load_path}")


def log_error(error: Exception, context: str = "") -> None:
    """
    Log an error with context.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    logger.error(f"Error in {context}: {str(error)}")
    logger.exception(error)


def log_warning(message: str, context: str = "") -> None:
    """
    Log a warning message.
    
    Args:
        message: Warning message
        context: Additional context
    """
    if context:
        logger.warning(f"Warning in {context}: {message}")
    else:
        logger.warning(message)


def log_info(message: str) -> None:
    """
    Log an info message.
    
    Args:
        message: Info message
    """
    logger.info(message)


def log_debug(message: str) -> None:
    """
    Log a debug message.
    
    Args:
        message: Debug message
    """
    logger.debug(message) 