"""Experiments module for Logistic Regression.

Calculates the ablation surface for LogReg parameters.
"""

from .ablation import evaluate_winner_on_test, generate_grid_configs, run_ablation_suite
from .grid import (
    CALIBRATION_CONFIG,
    FIXED_PARAMS,
    LOGREG_GRID,
    SOLVER_BY_PENALTY,
    TFIDF_GRID,
)
from .runner import ExperimentConfig, run_single_experiment

__all__ = [
    # Grid
    'TFIDF_GRID',
    'LOGREG_GRID',
    'CALIBRATION_CONFIG',
    'FIXED_PARAMS',
    'SOLVER_BY_PENALTY',
    # Runner
    'ExperimentConfig',
    'run_single_experiment',
    # Ablation
    'generate_grid_configs',
    'run_ablation_suite',
    'evaluate_winner_on_test',
]
