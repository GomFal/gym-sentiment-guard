"""DEPRECATED: Use gym_sentiment_guard.models.logreg.pipeline instead.

This module is maintained for backward compatibility.
New code should import from gym_sentiment_guard.models.logreg.pipeline.
"""

import warnings

from ..models.logreg.pipeline import *  # noqa: F401, F403

warnings.warn(
    'Importing from gym_sentiment_guard.training.model is deprecated. '
    'Use gym_sentiment_guard.models.logreg.pipeline instead.',
    DeprecationWarning,
    stacklevel=2,
)
