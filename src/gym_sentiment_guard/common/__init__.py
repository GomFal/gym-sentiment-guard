"""Common utilities shared across model implementations."""

from .artifacts import (
    RunConfig,
    RunResult,
    generate_run_id,
    get_environment_info,
    get_git_info,
    save_calibration_plot,
    save_predictions,
    save_run_artifact,
)
from .metrics import (
    ValMetrics,
    compute_brier_score,
    compute_ece,
    compute_skill_score,
    compute_test_metrics,
    compute_val_metrics,
)
from .protocols import ModelProtocol
from .stopwords import (
    STOPWORD_PRESETS,
    STOPWORDS_NEVER_REMOVE,
    STOPWORDS_SAFE,
    resolve_stop_words,
)
from .threshold import (
    ThresholdResult,
    apply_threshold,
    select_threshold,
)

__all__ = [
    # Artifacts
    'RunConfig',
    'RunResult',
    'get_git_info',
    'get_environment_info',
    'generate_run_id',
    'save_run_artifact',
    'save_predictions',
    'save_calibration_plot',
    # Metrics
    'ValMetrics',
    'compute_brier_score',
    'compute_ece',
    'compute_skill_score',
    'compute_val_metrics',
    'compute_test_metrics',
    # Threshold
    'ThresholdResult',
    'apply_threshold',
    'select_threshold',
    # Protocols
    'ModelProtocol',
    # Stopwords
    'STOPWORDS_SAFE',
    'STOPWORDS_NEVER_REMOVE',
    'STOPWORD_PRESETS',
    'resolve_stop_words',
]
