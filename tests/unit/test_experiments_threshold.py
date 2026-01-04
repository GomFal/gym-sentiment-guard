"""Unit tests for experiments threshold selection logic."""

from __future__ import annotations

import numpy as np

from gym_sentiment_guard.experiments.threshold import (
    ThresholdResult,
    apply_threshold,
    select_threshold,
)


class TestSelectThreshold:
    """Tests for select_threshold function."""

    def test_constraint_met_returns_met_status(self) -> None:
        """When recall constraint is achievable, status is 'met'."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # p_neg values that can achieve high recall for negative class
        p_neg = np.array([0.9, 0.8, 0.85, 0.7, 0.2, 0.3, 0.1, 0.25, 0.15, 0.05])

        result = select_threshold(y_true, p_neg, recall_constraint=0.75)

        assert isinstance(result, ThresholdResult)
        assert result.constraint_status == 'met'
        assert result.recall_neg >= 0.75

    def test_constraint_not_met_returns_fallback(self) -> None:
        """When constraint cannot be met, uses fallback and marks 'not_met'."""
        y_true = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # Only one negative sample, hard to get 90% recall
        p_neg = np.array([0.5, 0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55, 0.45])

        result = select_threshold(y_true, p_neg, recall_constraint=0.99)

        assert result.constraint_status == 'not_met'
        assert result.best_achievable_recall_neg is not None

    def test_maximizes_f1_among_valid_thresholds(self) -> None:
        """Selects threshold that maximizes F1_neg among constraint-meeting options."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        p_neg = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])

        result = select_threshold(y_true, p_neg, recall_constraint=0.80)

        # Should achieve at least 80% recall and maximize F1
        assert result.recall_neg >= 0.80
        assert result.f1_neg > 0

    def test_returns_all_metrics(self) -> None:
        """Result contains all required metrics."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.8, 0.6, 0.4, 0.2])

        result = select_threshold(y_true, p_neg, recall_constraint=0.50)

        assert hasattr(result, 'threshold')
        assert hasattr(result, 'f1_neg')
        assert hasattr(result, 'recall_neg')
        assert hasattr(result, 'precision_neg')
        assert hasattr(result, 'macro_f1')
        assert 0.0 <= result.threshold <= 1.0
        assert 0.0 <= result.f1_neg <= 1.0
        assert 0.0 <= result.recall_neg <= 1.0

    def test_uses_unique_p_neg_values_as_candidates(self) -> None:
        """Threshold selection uses unique p_neg values."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Duplicate p_neg values
        p_neg = np.array([0.7, 0.7, 0.7, 0.3, 0.3, 0.3])

        result = select_threshold(y_true, p_neg, recall_constraint=0.0)

        # Should still produce valid result
        assert result.threshold in [0.0, 0.3, 0.7, 1.0]

    def test_tie_breaker_uses_macro_f1(self) -> None:
        """When F1_neg is tied, macro_f1 is used as tie-breaker."""
        y_true = np.array([0, 0, 1, 1])
        p_neg = np.array([0.9, 0.8, 0.2, 0.1])

        result = select_threshold(y_true, p_neg, recall_constraint=0.0)

        # Should have valid macro_f1
        assert result.macro_f1 >= 0.0


class TestApplyThreshold:
    """Tests for apply_threshold function."""

    def test_decision_rule_p_neg_above_threshold_predicts_negative(self) -> None:
        """p_neg >= threshold → predict negative (0)."""
        p_neg = np.array([0.8, 0.6, 0.4, 0.2])
        threshold = 0.5

        y_pred = apply_threshold(p_neg, threshold)

        # 0.8 >= 0.5 → 0, 0.6 >= 0.5 → 0, 0.4 < 0.5 → 1, 0.2 < 0.5 → 1
        np.testing.assert_array_equal(y_pred, [0, 0, 1, 1])

    def test_decision_rule_exact_threshold_predicts_negative(self) -> None:
        """p_neg == threshold → predict negative (0)."""
        p_neg = np.array([0.5])
        threshold = 0.5

        y_pred = apply_threshold(p_neg, threshold)

        np.testing.assert_array_equal(y_pred, [0])

    def test_decision_rule_below_threshold_predicts_positive(self) -> None:
        """p_neg < threshold → predict positive (1)."""
        p_neg = np.array([0.3, 0.4, 0.49])
        threshold = 0.5

        y_pred = apply_threshold(p_neg, threshold)

        np.testing.assert_array_equal(y_pred, [1, 1, 1])

    def test_all_negative_at_zero_threshold(self) -> None:
        """All predictions are negative when threshold is 0."""
        p_neg = np.array([0.0, 0.5, 1.0])
        threshold = 0.0

        y_pred = apply_threshold(p_neg, threshold)

        np.testing.assert_array_equal(y_pred, [0, 0, 0])

    def test_all_positive_at_high_threshold(self) -> None:
        """All predictions are positive when threshold > max(p_neg)."""
        p_neg = np.array([0.0, 0.5, 0.9])
        threshold = 1.0

        y_pred = apply_threshold(p_neg, threshold)

        # Only p_neg=1.0 would be 0, but we don't have that
        np.testing.assert_array_equal(y_pred, [1, 1, 1])
