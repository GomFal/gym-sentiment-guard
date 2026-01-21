"""
Threshold selection for experiment runs.

Implements §3 of EXPERIMENT_PROTOCOL.md:
- Candidate enumeration from VAL p_neg values
- Selection objective: maximize F1_neg subject to Recall_neg ≥ constraint
- Constraint failure handling with fallback rule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ThresholdResult:
    """Result of threshold selection on VAL set."""

    threshold: float
    f1_neg: float
    recall_neg: float
    precision_neg: float
    macro_f1: float
    constraint_status: Literal['met', 'not_met']
    # Fallback details (only populated if constraint not met)
    best_achievable_recall_neg: float | None = None
    fallback_f1_neg: float | None = None


def select_threshold(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    recall_constraint: float = 0.90,
    pos_label: int = 0,  # negative class is 0 in our encoding
) -> ThresholdResult:
    """
    Select optimal threshold on VAL following §3 of EXPERIMENT_PROTOCOL.md.

    Decision rule (§2.3): Predict negative if p_neg >= threshold.

    Selection objective (§3.2):
    1. Filter candidates where Recall_neg >= recall_constraint
    2. Select threshold maximizing F1_neg
    3. Tie-break by Macro F1

    Constraint failure handling (§3.3):
    If no threshold satisfies constraint, use fallback rule:
    choose threshold that maximizes Recall_neg.

    OPTIMIZED: Uses vectorized computation instead of per-threshold sklearn calls.
    Complexity reduced from O(k*n) sklearn calls to O(n log n) sort + O(k) vectorized ops.

    Args:
        y_true: True labels (0=negative, 1=positive)
        p_neg: Predicted probability of negative class
        recall_constraint: Minimum required Recall_neg (default 0.90)
        pos_label: Label for the "positive" class in metrics (0=negative)

    Returns:
        ThresholdResult with selected threshold and metrics
    """
    n = len(y_true)

    # Pre-compute class counts (cached once)
    n_neg = int(np.sum(y_true == 0))
    n_pos = n - n_neg

    # Enumerate candidate thresholds using unique p_neg values (§3.1)
    # Add boundary values for completeness
    candidates = np.unique(np.concatenate([[0.0], p_neg, [1.0]]))
    k = len(candidates)

    # VECTORIZED: Compute predictions for ALL thresholds at once
    # Shape: (k, n) - each row is predictions for one threshold
    # Broadcasting: p_neg[None, :] shape (1, n), candidates[:, None] shape (k, 1)
    y_pred_all = np.where(p_neg[None, :] >= candidates[:, None], 0, 1)

    # Pre-compute y_true broadcasts for vectorized comparison
    y_true_is_neg = y_true == 0  # Shape: (n,)
    y_true_is_pos = ~y_true_is_neg  # Shape: (n,)

    # VECTORIZED: Compute confusion matrix components for all thresholds
    # TP_neg: predicted=0 AND true=0
    # FP_neg: predicted=0 AND true=1
    # FN_neg: predicted=1 AND true=0
    pred_is_neg = y_pred_all == 0  # Shape: (k, n)
    pred_is_pos = ~pred_is_neg  # Shape: (k, n)

    tp_neg = np.sum(pred_is_neg & y_true_is_neg[None, :], axis=1)  # Shape: (k,)
    fp_neg = np.sum(pred_is_neg & y_true_is_pos[None, :], axis=1)  # Shape: (k,)
    fn_neg = np.sum(pred_is_pos & y_true_is_neg[None, :], axis=1)  # Shape: (k,)

    # For positive class metrics (needed for macro F1)
    tp_pos = np.sum(pred_is_pos & y_true_is_pos[None, :], axis=1)  # Shape: (k,)
    fp_pos = fn_neg  # FP for positive = FN for negative
    fn_pos = fp_neg  # FN for positive = FP for negative

    # VECTORIZED: Compute metrics from confusion matrix components
    # Use np.divide with where parameter to avoid divide-by-zero warnings

    # Pre-allocate output arrays for safe division (k = number of thresholds)
    zeros_k = np.zeros(k, dtype=float)

    # Recall_neg = TP_neg / (TP_neg + FN_neg) = TP_neg / n_neg
    recall_neg = (
        np.divide(tp_neg, n_neg, out=zeros_k.copy(), where=n_neg > 0)
        if n_neg > 0
        else zeros_k.copy()
    )

    # Precision_neg = TP_neg / (TP_neg + FP_neg)
    denom_prec_neg = tp_neg + fp_neg
    precision_neg = np.divide(tp_neg, denom_prec_neg, out=zeros_k.copy(), where=denom_prec_neg > 0)

    # F1_neg = 2 * precision * recall / (precision + recall)
    denom_f1_neg = precision_neg + recall_neg
    f1_neg = np.divide(
        2 * precision_neg * recall_neg, denom_f1_neg, out=zeros_k.copy(), where=denom_f1_neg > 0
    )

    # Positive class metrics for macro F1
    recall_pos = (
        np.divide(tp_pos, n_pos, out=zeros_k.copy(), where=n_pos > 0)
        if n_pos > 0
        else zeros_k.copy()
    )
    denom_prec_pos = tp_pos + fp_pos
    precision_pos = np.divide(tp_pos, denom_prec_pos, out=zeros_k.copy(), where=denom_prec_pos > 0)
    denom_f1_pos = precision_pos + recall_pos
    f1_pos = np.divide(
        2 * precision_pos * recall_pos, denom_f1_pos, out=zeros_k.copy(), where=denom_f1_pos > 0
    )

    # Macro F1 = (F1_neg + F1_pos) / 2
    macro_f1 = (f1_neg + f1_pos) / 2

    # Filter out degenerate thresholds (all same class)
    # These have either tp_neg=0 or tp_pos=0 exclusively
    valid_mask = (tp_neg + fn_neg > 0) & (tp_pos + fn_pos > 0)

    # Find best threshold meeting constraint
    constraint_mask = (recall_neg >= recall_constraint) & valid_mask

    if np.any(constraint_mask):
        # Among those meeting constraint, maximize F1_neg, tie-break by macro_f1
        # Create sort key: (-f1_neg, -macro_f1) for argmin
        meeting_indices = np.where(constraint_mask)[0]
        meeting_f1 = f1_neg[constraint_mask]
        meeting_macro = macro_f1[constraint_mask]

        # Find best: max f1_neg, then max macro_f1
        best_local = np.lexsort((-meeting_macro, -meeting_f1))[0]
        best_idx = meeting_indices[best_local]

        return ThresholdResult(
            threshold=float(candidates[best_idx]),
            f1_neg=float(f1_neg[best_idx]),
            recall_neg=float(recall_neg[best_idx]),
            precision_neg=float(precision_neg[best_idx]),
            macro_f1=float(macro_f1[best_idx]),
            constraint_status='met',
        )
    else:
        # Constraint not met - use fallback: max recall_neg
        fallback_mask = valid_mask
        if not np.any(fallback_mask):
            raise ValueError('No valid threshold found (all predictions degenerate)')

        fallback_indices = np.where(fallback_mask)[0]
        fallback_recall = recall_neg[fallback_mask]
        best_local = np.argmax(fallback_recall)
        best_idx = fallback_indices[best_local]

        return ThresholdResult(
            threshold=float(candidates[best_idx]),
            f1_neg=float(f1_neg[best_idx]),
            recall_neg=float(recall_neg[best_idx]),
            precision_neg=float(precision_neg[best_idx]),
            macro_f1=float(macro_f1[best_idx]),
            constraint_status='not_met',
            best_achievable_recall_neg=float(recall_neg[best_idx]),
            fallback_f1_neg=float(f1_neg[best_idx]),
        )


def apply_threshold(p_neg: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply decision rule to get predictions.

    Decision rule (§2.3): Predict negative (0) if p_neg >= threshold.

    Args:
        p_neg: Predicted probability of negative class
        threshold: Decision threshold

    Returns:
        Predicted labels (0=negative, 1=positive)
    """
    return np.where(p_neg >= threshold, 0, 1)
