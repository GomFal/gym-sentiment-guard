"""DEPRECATED: Use gym_sentiment_guard.common.threshold instead."""

import warnings

from ..common.threshold import *  # noqa: F401, F403

warnings.warn(
    'Importing from experiments.threshold is deprecated. '
    'Use gym_sentiment_guard.common.threshold instead.',
    DeprecationWarning,
    stacklevel=2,
)
