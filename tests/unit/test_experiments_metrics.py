"""Unit tests for experiments metrics computation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score

from gym_sentiment_guard.common.metrics import (
    ValMetrics,
    compute_test_metrics,
    compute_val_metrics,
)


class TestComputeValMetrics:
    """Tests for compute_val_metrics function."""

    def test_returns_val_metrics_dataclass(self) -> None:
        """Returns a ValMetrics instance."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.8, 0.7, 0.3, 0.2])
        y_pred = np.array([0, 0, 1, 1])

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')

        assert isinstance(result, ValMetrics)

    def test_f1_neg_matches_sklearn(self) -> None:
        """F1_neg matches sklearn's f1_score for pos_label=0."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        p_neg = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1])

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')
        expected_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0.0)

        assert abs(result.f1_neg - expected_f1) < 1e-6

    def test_recall_neg_matches_sklearn(self) -> None:
        """Recall_neg matches sklearn's recall_score for pos_label=0."""
        y_true = np.array([0, 0, 1, 1, 1])
        p_neg = np.array([0.8, 0.3, 0.2, 0.1, 0.05])
        y_pred = np.array([0, 1, 1, 1, 1])  # 1 TP, 1 FN for negative

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')
        expected_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0.0)

        assert abs(result.recall_neg - expected_recall) < 1e-6

    def test_pr_auc_neg_computation(self) -> None:
        """PR AUC (Negative) is computed correctly using average precision."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.9, 0.8, 0.3, 0.1])
        y_pred = np.array([0, 0, 1, 1])

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')

        # y_neg_binary = 1 where y_true == 0
        y_neg_binary = (y_true == 0).astype(int)
        expected_pr_auc = average_precision_score(y_neg_binary, p_neg)

        assert abs(result.pr_auc_neg - expected_pr_auc) < 1e-6

    def test_constraint_status_preserved(self) -> None:
        """Constraint status is passed through correctly."""
        y_true = np.array([0, 1])
        p_neg = np.array([0.8, 0.2])
        y_pred = np.array([0, 1])

        result_met = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')
        result_not_met = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'not_met')

        assert result_met.constraint_status == 'met'
        assert result_not_met.constraint_status == 'not_met'

    def test_confusion_matrix_is_2x2(self) -> None:
        """Confusion matrix has shape 2x2."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.8, 0.7, 0.3, 0.2])
        y_pred = np.array([0, 0, 1, 1])

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')

        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2
        assert len(result.confusion_matrix[1]) == 2

    def test_support_counts_correct(self) -> None:
        """Support counts match actual class distribution."""
        y_true = np.array([0, 0, 0, 1, 1])  # 3 negative, 2 positive
        p_neg = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        y_pred = np.array([0, 0, 0, 1, 1])

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.5, 'met')

        assert result.support_neg == 3
        assert result.support_pos == 2

    def test_threshold_preserved(self) -> None:
        """Threshold value is preserved in result."""
        y_true = np.array([0, 1])
        p_neg = np.array([0.8, 0.2])
        y_pred = np.array([0, 1])

        result = compute_val_metrics(y_true, p_neg, y_pred, 0.42, 'met')

        assert result.threshold == 0.42


class TestComputeTestMetrics:
    """Tests for compute_test_metrics function."""

    def test_uses_final_test_status(self) -> None:
        """Test metrics have 'final_test' constraint status."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.8, 0.7, 0.3, 0.2])

        result = compute_test_metrics(y_true, p_neg, threshold=0.5)

        assert result.constraint_status == 'final_test'

    def test_applies_threshold_not_reoptimized(self) -> None:
        """Test metrics use provided threshold (no re-optimization)."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.8, 0.6, 0.4, 0.2])
        threshold = 0.5

        result = compute_test_metrics(y_true, p_neg, threshold=threshold)

        assert result.threshold == threshold

    def test_returns_val_metrics_instance(self) -> None:
        """Returns a ValMetrics instance for consistency."""
        y_true = np.array([0, 1])
        p_neg = np.array([0.8, 0.2])

        result = compute_test_metrics(y_true, p_neg, threshold=0.5)

        assert isinstance(result, ValMetrics)
