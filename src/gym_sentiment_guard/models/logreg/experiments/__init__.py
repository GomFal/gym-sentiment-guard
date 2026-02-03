"""Experiments module for Logistic Regression.

Calculates the ablation surface for LogReg parameters.
"""

from .ablation import (
    evaluate_winner_on_test,
    generate_grid_configs,
    rank_results,
    run_ablation_suite,
)
from .runner import ExperimentConfig, run_single_experiment

__all__ = [
    # Runner
    'ExperimentConfig',
    'run_single_experiment',
    # Ablation
    'generate_grid_configs',
    'rank_results',
    'run_ablation_suite',
    'evaluate_winner_on_test',
]
