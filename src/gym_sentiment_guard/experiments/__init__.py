"""
Experiments module for LogReg ablation studies.

This module implements systematic hyperparameter search following
EXPERIMENT_PROTOCOL.md specifications.

Modules:
- grid: Parameter grid definitions (§5) and stopwords (§6)
- threshold: Threshold selection logic (§3)
- metrics: Metrics computation (§7)
- artifacts: Run persistence (§10)
- runner: Single experiment execution
- ablation: Ablation suite orchestrator (§9)
"""

from .ablation import evaluate_winner_on_test, generate_grid_configs, run_ablation_suite
from .artifacts import RunConfig, RunResult, save_calibration_plot
from .grid import CALIBRATION_CONFIG, LOGREG_GRID, STOPWORDS_SAFE, STOPWORD_PRESETS, TFIDF_GRID, resolve_stop_words
from .metrics import (
    ValMetrics,
    compute_brier_score,
    compute_ece,
    compute_skill_score,
    compute_test_metrics,
    compute_val_metrics,
)
from .runner import ExperimentConfig, run_single_experiment
from .threshold import ThresholdResult, apply_threshold, select_threshold

__all__ = [
    # Grid definitions
    'TFIDF_GRID',
    'LOGREG_GRID',
    'CALIBRATION_CONFIG',
    'STOPWORDS_SAFE',
    'STOPWORD_PRESETS',
    'resolve_stop_words',
    # Threshold selection
    'select_threshold',
    'apply_threshold',
    'ThresholdResult',
    # Metrics
    'compute_val_metrics',
    'compute_test_metrics',
    'compute_brier_score',
    'compute_ece',
    'compute_skill_score',
    'ValMetrics',
    # Artifacts
    'RunConfig',
    'RunResult',
    'save_calibration_plot',
    # Runner
    'ExperimentConfig',
    'run_single_experiment',
    # Ablation
    'generate_grid_configs',
    'run_ablation_suite',
    'evaluate_winner_on_test',
]
