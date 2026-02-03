"""SVM model module.

Provides Linear SVM and RBF SVM with FeatureUnion and Isotonic calibration
for sentiment classification.
"""

from __future__ import annotations

from .experiments import (
    SVMExperimentConfig,
    SVCRBFExperimentConfig,
    run_ablation_suite,
    run_rbf_ablation_suite,
    run_single_experiment,
    run_single_rbf_experiment,
)
from .training import train_from_config, train_rbf_from_config
from .reports import generate_ablation_report

__all__ = [
    # Linear SVM Experiments
    'SVMExperimentConfig',
    'run_single_experiment',
    'run_ablation_suite',
    # RBF SVM Experiments
    'SVCRBFExperimentConfig',
    'run_single_rbf_experiment',
    'run_rbf_ablation_suite',
    # Training
    'train_from_config',
    'train_rbf_from_config',
    # Reports
    'generate_ablation_report',
]

