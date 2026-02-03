"""SVM training module.

Provides train_from_config for reproducible SVM model training.
"""

from __future__ import annotations

from .training import train_from_config
from .training_rbf import train_rbf_from_config

__all__ = ['train_from_config', 'train_rbf_from_config']

