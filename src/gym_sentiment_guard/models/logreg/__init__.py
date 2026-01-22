"""Logistic Regression model implementation.

This package contains LogReg-specific:
- pipeline.py: Training pipeline (TF-IDF + LogReg + Isotonic calibration)
- experiments/: Hyperparameter grid search and ablation studies
- reports/: Error analysis and ablation reporting
"""

from .training import train_from_config

__all__ = [
    'train_from_config',
]
