"""
Metrics computation for experiment runs.

Implements §7 of EXPERIMENT_PROTOCOL.md:
- Required VAL metrics: F1_neg, Recall_neg, Precision_neg, Macro F1, PR AUC (Negative)
- Diagnostic metrics: confusion matrix, timing, feature counts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ValMetrics:
    """Metrics computed on VAL set (§7.1)."""

    # Required metrics at selected threshold
    f1_neg: float
    recall_neg: float
    precision_neg: float
    macro_f1: float
    pr_auc_neg: float  # Average Precision on p_neg

    # Threshold info
    threshold: float
    constraint_status: str  # "met", "not_met", or "final_test"

    # Calibration metrics
    brier_score: float = 0.0
    ece: float = 0.0
    skill_score: float = 0.0

    # Diagnostics (§7.3)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classification_report: dict[str, Any] = field(default_factory=dict)
    support_neg: int = 0
    support_pos: int = 0


def compute_brier_score(y_true: np.ndarray, p_neg: np.ndarray) -> float:
    """
    Compute Brier score for negative class probability.

    Uses sklearn's brier_score_loss for reliable, tested implementation.
    Lower is better, 0 = perfect calibration.
    """
    from sklearn.metrics import brier_score_loss

    y_neg_binary = (y_true == 0).astype(int)
    return float(brier_score_loss(y_neg_binary, p_neg))


def compute_ece(y_true: np.ndarray, p_neg: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) using sklearn's calibration_curve.

    Uses quantile strategy to ensure each bin has statistical significance.
    ECE = mean(|bin_accuracy - bin_confidence|)
    Lower is better, 0 = perfectly calibrated.
    """
    from sklearn.calibration import calibration_curve

    y_neg_binary = (y_true == 0).astype(int)

    # Use sklearn's vectorized binning with quantile strategy
    prob_true, prob_pred = calibration_curve(
        y_neg_binary, p_neg, n_bins=n_bins, strategy='quantile'
    )

    # With quantile strategy, bins have roughly equal samples, so use equal weights
    # ECE is mean absolute calibration error across bins
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    return ece


def compute_skill_score(brier_score: float, class_prior: float) -> float:
    """
    Compute skill score (improvement over baseline).

    Skill = 1 - (brier / baseline_brier)
    Higher is better, 1 = perfect, 0 = no better than baseline.
    """
    # Baseline Brier score for always predicting the class prior
    baseline_brier = class_prior * (1 - class_prior)
    if baseline_brier <= 0:
        return 0.0
    return float(1 - (brier_score / baseline_brier))


def compute_val_metrics(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    constraint_status: str,
    class_prior: float | None = None,
) -> ValMetrics:
    """
    Compute all required VAL metrics per §7.1.

    Args:
        y_true: True labels (0=negative, 1=positive)
        p_neg: Predicted probability of negative class
        y_pred: Predicted labels at selected threshold
        threshold: Selected threshold
        constraint_status: "met", "not_met", or "final_test"
        class_prior: Prior probability of negative class (for skill score)

    Returns:
        ValMetrics with all required and diagnostic metrics
    """
    # Required metrics (§7.1)
    f1_neg = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0.0))
    recall_neg = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0.0))
    precision_neg = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0.0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0.0))

    # PR AUC (Negative) - Average Precision on p_neg (§7.1)
    # y_target = 1 when y_true is negative (0), so we invert
    y_neg_binary = (y_true == 0).astype(int)
    pr_auc_neg = float(average_precision_score(y_neg_binary, p_neg))

    # Calibration metrics
    brier = compute_brier_score(y_true, p_neg)
    ece = compute_ece(y_true, p_neg)

    # Skill score (use provided prior or compute from data)
    if class_prior is None:
        class_prior = float(np.mean(y_neg_binary))
    skill = compute_skill_score(brier, class_prior)

    # Diagnostics (§7.3)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)

    # Support counts (cached mask for efficiency)
    support_neg = int(np.sum(y_neg_binary))
    support_pos = len(y_true) - support_neg

    return ValMetrics(
        f1_neg=f1_neg,
        recall_neg=recall_neg,
        precision_neg=precision_neg,
        macro_f1=macro_f1,
        pr_auc_neg=pr_auc_neg,
        threshold=threshold,
        constraint_status=constraint_status,
        brier_score=brier,
        ece=ece,
        skill_score=skill,
        confusion_matrix=cm.tolist(),
        classification_report=report,
        support_neg=support_neg,
        support_pos=support_pos,
    )


def compute_test_metrics(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    threshold: float,
) -> ValMetrics:
    """
    Compute TEST metrics for final winner (§7.2).

    Uses the same threshold selected on VAL (not re-optimized).

    Args:
        y_true: True labels (0=negative, 1=positive)
        p_neg: Predicted probability of negative class
        threshold: Threshold selected on VAL (carried over)

    Returns:
        ValMetrics computed on TEST
    """
    # Apply threshold (not re-optimized per §7.2)
    y_pred = np.where(p_neg >= threshold, 0, 1)

    return compute_val_metrics(
        y_true=y_true,
        p_neg=p_neg,
        y_pred=y_pred,
        threshold=threshold,
        constraint_status='final_test',  # Special status for TEST
    )
