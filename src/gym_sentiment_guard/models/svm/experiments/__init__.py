"""SVM experiments module.

Provides ablation suites for Linear SVM and RBF SVM with FeatureUnion and Isotonic calibration.
"""

from __future__ import annotations

# Linear SVM exports
from .ablation import (
    evaluate_winner_on_test,
    generate_grid_configs,
    rank_results,
    run_ablation_suite,
)
from .runner import SVMExperimentConfig, run_single_experiment

# RBF SVM exports
from .ablation_rbf import (
    evaluate_rbf_winner_on_test,
    generate_rbf_grid_configs,
    run_rbf_ablation_suite,
)
from .runner_rbf import SVCRBFExperimentConfig, run_single_rbf_experiment

# Shared utilities
from .common import rank_results, validate_label_mapping

__all__ = [
    # Linear SVM Runner
    'SVMExperimentConfig',
    'run_single_experiment',
    # Linear SVM Ablation
    'generate_grid_configs',
    'rank_results',
    'run_ablation_suite',
    'evaluate_winner_on_test',
    # RBF SVM Runner
    'SVCRBFExperimentConfig',
    'run_single_rbf_experiment',
    # RBF SVM Ablation
    'generate_rbf_grid_configs',
    'run_rbf_ablation_suite',
    'evaluate_rbf_winner_on_test',
    # Shared utilities
    'validate_label_mapping',
]
