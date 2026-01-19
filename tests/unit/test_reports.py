"""Unit tests for the reports module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_sentiment_guard.reports.schema import (
    REQUIRED_COLUMNS,
    _extract_run_data,
    load_all_runs,
    load_test_predictions,
    validate_schema,
)
from gym_sentiment_guard.reports.tables import (
    export_csv,
    generate_comparison_table,
    get_top_k,
    get_winner,
    sort_ablation_table,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_run_json() -> dict:
    """Create a sample run.json structure for testing."""
    return {
        'run_id': 'run.2026-01-01_001',
        'tfidf_params': {
            'ngram_range': [1, 2],
            'min_df': 2,
            'max_df': 0.9,
            'sublinear_tf': True,
            'stop_words': 'custom',
        },
        'logreg_params': {
            'penalty': 'l2',
            'C': 1.0,
            'class_weight': None,
        },
        'diagnostics': {
            'n_features': 5000,
            'coefficient_sparsity': 0.25,
            'training_time_seconds': 1.5,
        },
        'val_metrics': {
            'constraint_status': 'met',
            'f1_neg': 0.85,
            'recall_neg': 0.92,
            'precision_neg': 0.80,
            'macro_f1': 0.90,
            'pr_auc_neg': 0.88,
            'threshold': 0.4,
            'brier_score': 0.05,
            'ece': 0.02,
            'classification_report': {
                '0': {'f1-score': 0.85},
                '1': {'f1-score': 0.95},
            },
            'confusion_matrix': [[400, 50], [60, 1500]],
        },
    }


@pytest.fixture
def sample_run_json_invalid() -> dict:
    """Create a sample run.json with constraint NOT met."""
    return {
        'run_id': 'run.2026-01-01_002',
        'tfidf_params': {
            'ngram_range': [1, 1],
            'min_df': 1,
            'max_df': 1.0,
            'sublinear_tf': False,
            'stop_words': None,
        },
        'logreg_params': {
            'penalty': 'l2',
            'C': 0.1,
            'class_weight': 'balanced',
        },
        'diagnostics': {
            'n_features': 3000,
            'coefficient_sparsity': 0.50,
            'training_time_seconds': 0.8,
        },
        'val_metrics': {
            'constraint_status': 'not_met',
            'f1_neg': 0.75,
            'recall_neg': 0.80,
            'precision_neg': 0.70,
            'macro_f1': 0.82,
            'pr_auc_neg': 0.78,
            'threshold': 0.5,
            'brier_score': 0.10,
            'ece': 0.05,
            'classification_report': {
                '0': {'f1-score': 0.75},
                '1': {'f1-score': 0.89},
            },
            'confusion_matrix': [[350, 100], [100, 1450]],
        },
    }


@pytest.fixture
def sample_ablation_df() -> pd.DataFrame:
    """Create a sample ablation DataFrame for testing sorting logic."""
    return pd.DataFrame([
        # Best: constraint met, highest F1_neg
        {'run_id': 'run_001', 'constraint_met': True, 'F1_neg': 0.90, 'Macro_F1': 0.92, 'PR_AUC_neg': 0.89, 'Brier_Score': 0.04, 'ECE': 0.01, 'Recall_neg': 0.93, 'threshold': 0.4, 'Precision_neg': 0.88},
        # Second: constraint met, lower F1_neg
        {'run_id': 'run_002', 'constraint_met': True, 'F1_neg': 0.88, 'Macro_F1': 0.90, 'PR_AUC_neg': 0.87, 'Brier_Score': 0.05, 'ECE': 0.02, 'Recall_neg': 0.91, 'threshold': 0.45, 'Precision_neg': 0.85},
        # Third: constraint met, tied F1_neg with run_004, better Macro_F1
        {'run_id': 'run_003', 'constraint_met': True, 'F1_neg': 0.85, 'Macro_F1': 0.88, 'PR_AUC_neg': 0.85, 'Brier_Score': 0.06, 'ECE': 0.03, 'Recall_neg': 0.90, 'threshold': 0.5, 'Precision_neg': 0.81},
        # Fourth: constraint met, tied F1_neg with run_003, worse Macro_F1
        {'run_id': 'run_004', 'constraint_met': True, 'F1_neg': 0.85, 'Macro_F1': 0.86, 'PR_AUC_neg': 0.84, 'Brier_Score': 0.07, 'ECE': 0.04, 'Recall_neg': 0.90, 'threshold': 0.5, 'Precision_neg': 0.80},
        # Fifth: constraint NOT met (should sort last despite high F1)
        {'run_id': 'run_005', 'constraint_met': False, 'F1_neg': 0.95, 'Macro_F1': 0.96, 'PR_AUC_neg': 0.94, 'Brier_Score': 0.02, 'ECE': 0.005, 'Recall_neg': 0.85, 'threshold': 0.6, 'Precision_neg': 0.99},
    ])


@pytest.fixture
def sample_test_predictions_df() -> pd.DataFrame:
    """Create sample test predictions for testing."""
    return pd.DataFrame({
        'y_true': [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        'y_pred': [0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        'p_neg': [0.8, 0.1, 0.3, 0.2, 0.05, 0.9, 0.15, 0.1, 0.7, 0.4],
        'p_pos': [0.2, 0.9, 0.7, 0.8, 0.95, 0.1, 0.85, 0.9, 0.3, 0.6],
    })


# =============================================================================
# Schema Tests
# =============================================================================


class TestExtractRunData:
    """Tests for _extract_run_data function."""

    def test_extract_valid_run(self, sample_run_json: dict, tmp_path: Path):
        """Test extraction from a valid run.json."""
        run_path = tmp_path / 'run.json'
        run_path.write_text(json.dumps(sample_run_json), encoding='utf-8')

        result = _extract_run_data(run_path)

        assert result is not None
        assert result['run_id'] == 'run.2026-01-01_001'
        assert result['constraint_met'] is True
        assert result['F1_neg'] == 0.85
        assert result['Recall_neg'] == 0.92
        assert result['ngram_range'] == '(1, 2)'
        assert result['stopwords_enabled'] is True
        assert result['C'] == 1.0

    def test_extract_constraint_not_met(self, sample_run_json_invalid: dict, tmp_path: Path):
        """Test extraction when constraint is not met."""
        run_path = tmp_path / 'run.json'
        run_path.write_text(json.dumps(sample_run_json_invalid), encoding='utf-8')

        result = _extract_run_data(run_path)

        assert result is not None
        assert result['constraint_met'] is False
        assert result['stopwords_enabled'] is False

    def test_extract_missing_val_metrics(self, tmp_path: Path):
        """Test handling of missing val_metrics."""
        run_path = tmp_path / 'run.json'
        run_path.write_text(json.dumps({'run_id': 'test'}), encoding='utf-8')

        result = _extract_run_data(run_path)

        assert result is None

    def test_extract_invalid_json(self, tmp_path: Path):
        """Test handling of invalid JSON."""
        run_path = tmp_path / 'run.json'
        run_path.write_text('not valid json', encoding='utf-8')

        result = _extract_run_data(run_path)

        assert result is None


class TestLoadAllRuns:
    """Tests for load_all_runs function."""

    def test_load_multiple_runs(self, sample_run_json: dict, sample_run_json_invalid: dict, tmp_path: Path):
        """Test loading multiple run directories."""
        # Create run directories
        run1_dir = tmp_path / 'run.2026-01-01_001'
        run1_dir.mkdir()
        (run1_dir / 'run.json').write_text(json.dumps(sample_run_json), encoding='utf-8')

        run2_dir = tmp_path / 'run.2026-01-01_002'
        run2_dir.mkdir()
        (run2_dir / 'run.json').write_text(json.dumps(sample_run_json_invalid), encoding='utf-8')

        df = load_all_runs(tmp_path)

        assert len(df) == 2
        assert 'run.2026-01-01_001' in df['run_id'].values
        assert 'run.2026-01-01_002' in df['run_id'].values

    def test_load_empty_directory(self, tmp_path: Path):
        """Test loading from empty directory."""
        df = load_all_runs(tmp_path)

        assert len(df) == 0


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_with_all_columns(self, sample_ablation_df: pd.DataFrame):
        """Test validation passes with minimum required columns."""
        # Add missing columns for full schema
        for col in REQUIRED_COLUMNS:
            if col not in sample_ablation_df.columns:
                sample_ablation_df[col] = 0

        # Should not raise
        validate_schema(sample_ablation_df)

    def test_validate_missing_columns_raises(self):
        """Test validation raises on missing columns."""
        df = pd.DataFrame({'run_id': ['test']})

        with pytest.raises(ValueError, match='Missing required columns'):
            validate_schema(df)


class TestLoadTestPredictions:
    """Tests for load_test_predictions function."""

    def test_load_valid_predictions(self, sample_test_predictions_df: pd.DataFrame, tmp_path: Path):
        """Test loading valid predictions CSV."""
        pred_path = tmp_path / 'test_predictions.csv'
        sample_test_predictions_df.to_csv(pred_path, index=False)

        df = load_test_predictions(pred_path)

        assert len(df) == 10
        assert 'y_true' in df.columns
        assert 'p_neg' in df.columns
        assert 'p_pos' in df.columns

    def test_load_missing_columns_raises(self, tmp_path: Path):
        """Test loading raises on missing required columns."""
        pred_path = tmp_path / 'test_predictions.csv'
        pd.DataFrame({'y_true': [0, 1]}).to_csv(pred_path, index=False)

        with pytest.raises(ValueError, match='Missing columns'):
            load_test_predictions(pred_path)


# =============================================================================
# Tables Tests
# =============================================================================


class TestSortAblationTable:
    """Tests for sort_ablation_table function."""

    def test_sort_constraint_met_first(self, sample_ablation_df: pd.DataFrame):
        """Test that constraint_met=True sorts before False."""
        df_sorted = sort_ablation_table(sample_ablation_df)

        # First 4 should have constraint_met=True
        assert all(df_sorted.iloc[:4]['constraint_met'] == True)  # noqa: E712
        # Last should have constraint_met=False
        assert df_sorted.iloc[-1]['constraint_met'] == False  # noqa: E712

    def test_sort_by_f1neg_descending(self, sample_ablation_df: pd.DataFrame):
        """Test that within constraint_met=True, F1_neg sorts descending."""
        df_sorted = sort_ablation_table(sample_ablation_df)

        valid_runs = df_sorted[df_sorted['constraint_met'] == True]  # noqa: E712
        f1_values = valid_runs['F1_neg'].tolist()

        # Should be sorted descending
        assert f1_values == sorted(f1_values, reverse=True)

    def test_sort_tiebreaker_macro_f1(self, sample_ablation_df: pd.DataFrame):
        """Test that Macro_F1 is used as tiebreaker when F1_neg is equal."""
        df_sorted = sort_ablation_table(sample_ablation_df)

        # run_003 and run_004 have same F1_neg=0.85
        # run_003 has Macro_F1=0.88, run_004 has Macro_F1=0.86
        # So run_003 should come before run_004
        run_003_idx = df_sorted[df_sorted['run_id'] == 'run_003'].index[0]
        run_004_idx = df_sorted[df_sorted['run_id'] == 'run_004'].index[0]

        assert run_003_idx < run_004_idx


class TestGetTopK:
    """Tests for get_top_k function."""

    def test_get_top_5(self, sample_ablation_df: pd.DataFrame):
        """Test selecting top 5 valid runs."""
        df_sorted = sort_ablation_table(sample_ablation_df)
        top_k = get_top_k(df_sorted, k=5)

        # Only 4 valid runs exist (constraint_met=True)
        assert len(top_k) == 4
        assert all(top_k['constraint_met'] == True)  # noqa: E712

    def test_get_top_2(self, sample_ablation_df: pd.DataFrame):
        """Test selecting top 2."""
        df_sorted = sort_ablation_table(sample_ablation_df)
        top_k = get_top_k(df_sorted, k=2)

        assert len(top_k) == 2
        assert top_k.iloc[0]['run_id'] == 'run_001'
        assert top_k.iloc[1]['run_id'] == 'run_002'


class TestGetWinner:
    """Tests for get_winner function."""

    def test_get_winner_returns_best(self, sample_ablation_df: pd.DataFrame):
        """Test winner selection returns best valid run."""
        df_sorted = sort_ablation_table(sample_ablation_df)
        winner = get_winner(df_sorted)

        assert winner is not None
        assert winner['run_id'] == 'run_001'
        assert winner['F1_neg'] == 0.90

    def test_get_winner_no_valid_runs(self):
        """Test winner selection when no valid runs exist."""
        df = pd.DataFrame([
            {'run_id': 'run_001', 'constraint_met': False, 'F1_neg': 0.90},
        ])
        winner = get_winner(df)

        assert winner is None


class TestExportCSV:
    """Tests for export_csv function."""

    def test_export_creates_file(self, sample_ablation_df: pd.DataFrame, tmp_path: Path):
        """Test CSV export creates file."""
        output_path = tmp_path / 'test.csv'

        result = export_csv(sample_ablation_df, output_path)

        assert result.exists()
        assert result == output_path

    def test_export_creates_parent_dirs(self, sample_ablation_df: pd.DataFrame, tmp_path: Path):
        """Test export creates parent directories."""
        output_path = tmp_path / 'nested' / 'dir' / 'test.csv'

        result = export_csv(sample_ablation_df, output_path)

        assert result.exists()


class TestGenerateComparisonTable:
    """Tests for generate_comparison_table function."""

    def test_generates_markdown(self, sample_ablation_df: pd.DataFrame):
        """Test Markdown table generation."""
        markdown = generate_comparison_table(sample_ablation_df.head(3))

        assert '|' in markdown
        assert 'run_001' in markdown
        assert 'run_002' in markdown
