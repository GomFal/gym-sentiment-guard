"""DEPRECATED: Use gym_sentiment_guard.common.metrics instead."""

import warnings

from ..common.metrics import *  # noqa: F401, F403

warnings.warn(
    'Importing from experiments.metrics is deprecated. '
    'Use gym_sentiment_guard.common.metrics instead.',
    DeprecationWarning,
    stacklevel=2,
)
