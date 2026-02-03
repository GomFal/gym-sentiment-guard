"""
Common utilities for SVM experiment modules.

Provides shared functionality for both Linear SVM and RBF SVM experiments:
- Result ranking
- Label mapping validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from gym_sentiment_guard.common.artifacts import RunResult




def validate_label_mapping(
    df: pd.DataFrame,
    label_column: str,
    label_mapping: dict[str, int],
    source_path: str,
) -> pd.Series:
    """
    Validate and apply label mapping, raising if unmapped labels found.

    Args:
        df: DataFrame with labels
        label_column: Name of the label column
        label_mapping: Mapping from label strings to integers
        source_path: Path to the source file (for error messages)

    Returns:
        Mapped labels as integer Series

    Raises:
        ValueError: If unmapped labels are found
    """
    mapped = df[label_column].map(label_mapping)
    unmapped = df.loc[mapped.isnull(), label_column].unique()

    if len(unmapped) > 0:
        raise ValueError(
            f"Unmapped labels found in data: {unmapped.tolist()}. "
            f"Source: {source_path}, column: {label_column}"
        )

    return mapped.astype(int)


def rank_results(results: list[RunResult]) -> list[RunResult]:
    """
    Rank results by primary objective and tie-breakers (§1).

    Primary: F1_neg (maximize) subject to Recall_neg >= 0.90
    Tie-breakers (in order):
    1. Macro F1 (higher is better)
    2. PR AUC (Negative) (higher is better)
    3. Brier Score (lower is better)
    4. ECE (lower is better)

    Valid runs with constraint met > constraint not met > invalid

    Args:
        results: List of run results

    Returns:
        Sorted list (best first)
    """

    def sort_key(r: RunResult) -> tuple[Any, ...]:
        """
        Sort key: (validity, constraint, f1_neg, macro_f1, pr_auc, -brier, -ece)
        Higher is better, so we negate brier/ece.
        """
        if r.validity_status == 'invalid':
            return (0, 0, 0, 0, 0, 0, 0)

        if r.val_metrics is None:
            return (0, 0, 0, 0, 0, 0, 0)

        constraint_met = 1 if r.val_metrics.constraint_status == 'met' else 0

        return (
            1,  # valid
            constraint_met,
            r.val_metrics.f1_neg,
            r.val_metrics.macro_f1,
            r.val_metrics.pr_auc_neg,
            -r.val_metrics.brier_score,  # Lower is better
            -r.val_metrics.ece,  # Lower is better
        )

    return sorted(results, key=sort_key, reverse=True)
