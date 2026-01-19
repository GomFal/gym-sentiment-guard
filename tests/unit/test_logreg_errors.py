"""
Unit tests for error analysis module.

TASK 11: Minimal test coverage per ERROR_ANALYSIS_MODULE.md spec.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample Spanish gym review texts."""
    return [
        'excelente gimnasio muy limpio',
        'horrible servicio pero buenas máquinas',
        'buen precio aunque algo sucio',
        'malo muy malo no recomiendo',
        'perfecto todo bien muy contento',
    ]


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Labels: 0=negative, 1=positive."""
    return np.array([1, 0, 0, 0, 1])


@pytest.fixture
def fitted_model(sample_texts: list[str], sample_labels: np.ndarray) -> Pipeline:
    """Create a simple fitted model for testing."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    base_clf = LogisticRegression(max_iter=100, random_state=42)
    # Use simple classifier (not calibrated) for faster testing
    model = Pipeline([
        ('tfidf', vectorizer),
        ('logreg', base_clf),
    ])
    model.fit(sample_texts, sample_labels)
    return model


@pytest.fixture
def sample_merged_df() -> pd.DataFrame:
    """Sample merged DataFrame as produced by loader."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            'excelente gimnasio muy limpio',
            'horrible servicio pero buenas máquinas',
            'buen precio aunque algo sucio',
            'malo muy malo no recomiendo',
            'perfecto todo bien muy contento',
        ],
        'y_true': [1, 0, 0, 0, 1],
        'y_pred': [1, 1, 0, 0, 0],  # 2 errors: ids 2 and 5
        'p_neg': [0.2, 0.35, 0.6, 0.8, 0.55],
        'p_pos': [0.8, 0.65, 0.4, 0.2, 0.45],
    })


# =============================================================================
# Error Table Tests
# =============================================================================


class TestErrorTable:
    """Tests for error_table.py."""

    def test_schema_validation(
        self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline
    ) -> None:
        """Test that error table has all required columns."""
        from gym_sentiment_guard.reports.logreg_errors.error_table import build_error_table

        result = build_error_table(sample_merged_df, fitted_model, threshold=0.37)

        required_cols = [
            'id', 'text', 'y_true', 'y_pred', 'p_neg', 'p_pos',
            'abs_margin', 'loss', 'is_error', 'nnz', 'tfidf_sum', 'low_coverage',
        ]
        for col in required_cols:
            assert col in result.columns, f'Missing column: {col}'

    def test_loss_computation(
        self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline
    ) -> None:
        """Test that loss is computed correctly."""
        from gym_sentiment_guard.reports.logreg_errors.error_table import build_error_table

        result = build_error_table(sample_merged_df, fitted_model, threshold=0.37)

        # Loss should be positive (cross-entropy)
        assert (result['loss'] >= 0).all()
        # Loss should be finite
        assert np.isfinite(result['loss']).all()


# =============================================================================
# Risk Tag Tests
# =============================================================================


class TestRiskTags:
    """Tests for risk_tags.py."""

    def test_near_threshold(self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline) -> None:
        """Test near_threshold computation."""
        from gym_sentiment_guard.reports.logreg_errors.error_table import build_error_table
        from gym_sentiment_guard.reports.logreg_errors.risk_tags import compute_risk_tags

        error_df = build_error_table(sample_merged_df, fitted_model, threshold=0.37)
        result = compute_risk_tags(error_df, threshold=0.37, near_threshold_band=0.10, contrast_keywords=set())

        assert 'near_threshold' in result.columns
        # p_neg=0.35 is within [0.27, 0.47], so should be True
        idx_35 = result[result['p_neg'] == 0.35].index
        if len(idx_35) > 0:
            assert result.loc[idx_35[0], 'near_threshold'] == True  # noqa: E712 - numpy bool

    def test_has_contrast(self) -> None:
        """Test contrast keyword detection."""
        from gym_sentiment_guard.reports.logreg_errors.risk_tags import has_contrast_marker

        keywords = {'pero', 'aunque', 'sin embargo'}

        assert has_contrast_marker('muy bueno pero sucio', keywords) is True
        assert has_contrast_marker('excelente servicio', keywords) is False
        assert has_contrast_marker('AUNQUE está lejos', keywords) is True


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for reproducibility."""

    def test_same_input_same_output(
        self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline
    ) -> None:
        """Test that same inputs produce identical outputs."""
        from gym_sentiment_guard.reports.logreg_errors.error_table import build_error_table

        result1 = build_error_table(sample_merged_df.copy(), fitted_model, threshold=0.37)
        result2 = build_error_table(sample_merged_df.copy(), fitted_model, threshold=0.37)

        pd.testing.assert_frame_equal(result1, result2)


# =============================================================================
# Slice Metrics Tests
# =============================================================================


class TestSliceMetrics:
    """Tests for slices.py."""

    def test_slice_counts(self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline) -> None:
        """Test that slice sample counts are correct."""
        from gym_sentiment_guard.reports.logreg_errors.error_table import build_error_table
        from gym_sentiment_guard.reports.logreg_errors.risk_tags import compute_risk_tags
        from gym_sentiment_guard.reports.logreg_errors.slices import compute_slice_metrics

        error_df = build_error_table(sample_merged_df, fitted_model, threshold=0.37)
        error_df = compute_risk_tags(error_df, threshold=0.37, near_threshold_band=0.10, contrast_keywords=set())

        # Use min_slice_size=1 to avoid skipping with small test data
        result = compute_slice_metrics(error_df, min_slice_size=1)

        assert 'overall' in result
        assert result['overall']['n'] == len(sample_merged_df)

    def test_min_slice_filter(self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline) -> None:
        """Test that small slices are skipped."""
        from gym_sentiment_guard.reports.logreg_errors.error_table import build_error_table
        from gym_sentiment_guard.reports.logreg_errors.risk_tags import compute_risk_tags
        from gym_sentiment_guard.reports.logreg_errors.slices import compute_slice_metrics

        error_df = build_error_table(sample_merged_df, fitted_model, threshold=0.37)
        error_df = compute_risk_tags(error_df, threshold=0.37, near_threshold_band=0.10, contrast_keywords=set())

        # With 5 samples and min_slice_size=10, only 'overall' might have enough
        result = compute_slice_metrics(error_df, min_slice_size=10)

        # Small slices should be marked as skipped
        for slice_name, metrics in result.items():
            if metrics.get('n', 0) < 10:
                assert metrics.get('skipped', False) is True


# =============================================================================
# Contributions Tests
# =============================================================================


class TestContributions:
    """Tests for contributions.py."""

    def test_non_empty_output(
        self, sample_merged_df: pd.DataFrame, fitted_model: Pipeline
    ) -> None:
        """Test that contributions are produced for errors."""
        from gym_sentiment_guard.reports.logreg_errors.contributions import compute_example_contributions

        # Get error examples
        errors = sample_merged_df[sample_merged_df['y_true'] != sample_merged_df['y_pred']]

        if len(errors) > 0:
            result = compute_example_contributions(errors, fitted_model, top_k=5)

            assert len(result) > 0
            for contrib in result:
                assert 'id' in contrib
                assert 'top_positive_contributors' in contrib
                assert 'top_negative_contributors' in contrib
