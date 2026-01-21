"""DEPRECATED: Use gym_sentiment_guard.common.artifacts instead.

This module is maintained for backward compatibility.
New code should import from gym_sentiment_guard.common.artifacts.

Note: logreg_params has been renamed to classifier_params in the new module.
For backward compatibility, RunConfig still accepts logreg_params via __init__.
"""

import warnings

from ..common.artifacts import (  # noqa: F401
    RunConfig,
    RunResult,
    generate_run_id,
    get_environment_info,
    get_git_info,
    save_calibration_plot,
    save_predictions,
    save_run_artifact,
)

warnings.warn(
    'Importing from gym_sentiment_guard.experiments.artifacts is deprecated. '
    'Use gym_sentiment_guard.common.artifacts instead. '
    'Note: logreg_params has been renamed to classifier_params.',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'RunConfig',
    'RunResult',
    'get_git_info',
    'get_environment_info',
    'generate_run_id',
    'save_run_artifact',
    'save_predictions',
    'save_calibration_plot',
]
