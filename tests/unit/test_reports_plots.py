"""Unit tests for reports plots module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_sentiment_guard.reports.plots import (
    plot_c_vs_f1neg,
    plot_calibration_curve,
    plot_confusion_matrix_heatmap,
    plot_ngram_effect,
    plot_pr_curve_neg,
    plot_stopwords_effect,
    plot_threshold_curve,
    plot_top_k_f1neg_bar,
    plot_val_vs_test_comparison,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_top5_df() -> pd.DataFrame:
    """Sample Top-5 DataFrame for plot testing."""
    return pd.DataFrame([
        {'run_id': 'run_001', 'F1_neg': 0.90, 'Recall_neg': 0.93, 'threshold': 0.40},
        {'run_id': 'run_002', 'F1_neg': 0.88, 'Recall_neg': 0.91, 'threshold': 0.45},
        {'run_id': 'run_003', 'F1_neg': 0.85, 'Recall_neg': 0.90, 'threshold': 0.50},
    ])


@pytest.fixture
def sample_ablation_df() -> pd.DataFrame:
    """Sample ablation DataFrame for factor plots."""
    return pd.DataFrame([
        {'C': 0.1, 'F1_neg': 0.80, 'ngram_range': '(1, 1)', 'stopwords_enabled': True},
        {'C': 0.1, 'F1_neg': 0.82, 'ngram_range': '(1, 1)', 'stopwords_enabled': False},
        {'C': 1.0, 'F1_neg': 0.85, 'ngram_range': '(1, 2)', 'stopwords_enabled': True},
        {'C': 1.0, 'F1_neg': 0.87, 'ngram_range': '(1, 2)', 'stopwords_enabled': False},
        {'C': 10.0, 'F1_neg': 0.83, 'ngram_range': '(1, 3)', 'stopwords_enabled': True},
        {'C': 10.0, 'F1_neg': 0.84, 'ngram_range': '(1, 3)', 'stopwords_enabled': False},
    ])


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Sample predictions for PR curve and calibration tests."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    p_neg = np.array([0.9, 0.2, 0.8, 0.1, 0.15, 0.7, 0.25, 0.05, 0.85, 0.6, 0.3, 0.75, 0.2, 0.65, 0.1])
    return y_true, p_neg


# =============================================================================
# Plot Tests - Determinism & Output
# =============================================================================


class TestPlotTopKBar:
    """Tests for plot_top_k_f1neg_bar function."""

    def test_creates_png_file(self, sample_top5_df: pd.DataFrame, tmp_path: Path):
        """Test that function creates a PNG file."""
        output = tmp_path / 'test_top5.png'

        result = plot_top_k_f1neg_bar(sample_top5_df, output)

        assert result.exists()
        assert result.suffix == '.png'
        assert result.stat().st_size > 0

    def test_deterministic_output(self, sample_top5_df: pd.DataFrame, tmp_path: Path):
        """Test that plots are deterministic (same data = same file size)."""
        output1 = tmp_path / 'test1.png'
        output2 = tmp_path / 'test2.png'

        plot_top_k_f1neg_bar(sample_top5_df, output1)
        plot_top_k_f1neg_bar(sample_top5_df, output2)

        # File sizes should be identical for deterministic output
        assert output1.stat().st_size == output2.stat().st_size


class TestPlotCvsF1neg:
    """Tests for plot_c_vs_f1neg function."""

    def test_creates_png_file(self, sample_ablation_df: pd.DataFrame, tmp_path: Path):
        """Test that function creates a PNG file."""
        output = tmp_path / 'test_c.png'

        result = plot_c_vs_f1neg(sample_ablation_df, output)

        assert result.exists()
        assert result.stat().st_size > 0


class TestPlotNgramEffect:
    """Tests for plot_ngram_effect function."""

    def test_creates_png_file(self, sample_ablation_df: pd.DataFrame, tmp_path: Path):
        """Test that function creates a PNG file."""
        output = tmp_path / 'test_ngram.png'

        result = plot_ngram_effect(sample_ablation_df, output)

        assert result.exists()


class TestPlotStopwordsEffect:
    """Tests for plot_stopwords_effect function."""

    def test_creates_png_file(self, sample_ablation_df: pd.DataFrame, tmp_path: Path):
        """Test that function creates a PNG file."""
        output = tmp_path / 'test_stop.png'

        result = plot_stopwords_effect(sample_ablation_df, output)

        assert result.exists()


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix_heatmap function."""

    def test_creates_png_file(self, tmp_path: Path):
        """Test confusion matrix plot creation."""
        cm = np.array([[400, 50], [60, 1500]])
        output = tmp_path / 'test_cm.png'

        result = plot_confusion_matrix_heatmap(cm, 'Test CM', output)

        assert result.exists()

    def test_accepts_list_input(self, tmp_path: Path):
        """Test that function accepts list as input."""
        cm = [[400, 50], [60, 1500]]
        output = tmp_path / 'test_cm_list.png'

        result = plot_confusion_matrix_heatmap(cm, 'Test CM', output)

        assert result.exists()


class TestPlotPRCurve:
    """Tests for plot_pr_curve_neg function."""

    def test_creates_png_file(self, sample_predictions: tuple, tmp_path: Path):
        """Test PR curve plot creation."""
        y_true, p_neg = sample_predictions
        output = tmp_path / 'test_pr.png'

        result = plot_pr_curve_neg(y_true, p_neg, threshold=0.5, output_path=output)

        assert result.exists()


class TestPlotThresholdCurve:
    """Tests for plot_threshold_curve function."""

    def test_creates_png_file(self, sample_predictions: tuple, tmp_path: Path):
        """Test threshold curve plot creation."""
        y_true, p_neg = sample_predictions
        output = tmp_path / 'test_thresh.png'

        result = plot_threshold_curve(y_true, p_neg, selected_threshold=0.5, output_path=output)

        assert result.exists()


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""

    def test_creates_png_file(self, sample_predictions: tuple, tmp_path: Path):
        """Test calibration curve plot creation."""
        y_true, p_neg = sample_predictions
        output = tmp_path / 'test_calib.png'

        result = plot_calibration_curve(y_true, p_neg, output_path=output, n_bins=5)

        assert result.exists()


class TestPlotValVsTest:
    """Tests for plot_val_vs_test_comparison function."""

    def test_creates_png_file(self, tmp_path: Path):
        """Test VAL vs TEST comparison plot."""
        val_metrics = {'F1_neg': 0.85, 'Recall_neg': 0.92, 'Precision_neg': 0.80}
        test_metrics = {'F1_neg': 0.83, 'Recall_neg': 0.90, 'Precision_neg': 0.78}
        output = tmp_path / 'test_valtest.png'

        result = plot_val_vs_test_comparison(val_metrics, test_metrics, output)

        assert result.exists()
