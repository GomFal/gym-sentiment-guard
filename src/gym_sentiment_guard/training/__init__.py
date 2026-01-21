"""DEPRECATED: Use gym_sentiment_guard.models.logreg instead.

This module is maintained for backward compatibility.
New code should import from gym_sentiment_guard.models.logreg.
"""

import warnings

from ..models.logreg.pipeline import train_from_config

warnings.warn(
    'Importing from gym_sentiment_guard.training is deprecated. '
    'Use gym_sentiment_guard.models.logreg instead.',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['train_from_config']
