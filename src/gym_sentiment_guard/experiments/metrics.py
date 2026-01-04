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

    # Diagnostics (§7.3)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classification_report: dict[str, Any] = field(default_factory=dict)
    support_neg: int = 0
    support_pos: int = 0


def compute_val_metrics(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    constraint_status: str,
) -> ValMetrics:
    """
    Compute all required VAL metrics per §7.1.

    Args:
        y_true: True labels (0=negative, 1=positive)
        p_neg: Predicted probability of negative class
        y_pred: Predicted labels at selected threshold
        threshold: Selected threshold
        constraint_status: "met", "not_met", or "final_test"

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
