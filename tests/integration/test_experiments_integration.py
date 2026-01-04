"""Integration tests for experiments module end-to-end."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gym_sentiment_guard.experiments import (
    ExperimentConfig,
    run_single_experiment,
)
from gym_sentiment_guard.experiments.ablation import generate_grid_configs, rank_results


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    """Create minimal train/val CSVs for testing."""
    # Create train data (20 samples)
    train_data = {
        'comment': [
            'excelente gimnasio muy bueno',
            'terrible horrible malo',
            'perfecto increible fantastico',
            'pesimo asqueroso fatal',
            'muy buen servicio',
            'mala atencion',
            'genial experiencia',
            'decepcionante',
            'super recomendado',
            'no lo recomiendo',
            'fantastico lugar',
            'muy malo',
            'increible todo',
            'horrible servicio',
            'excelente instalaciones',
            'pesimas maquinas',
            'muy limpio',
            'sucio y feo',
            'grande y espacioso',
            'pequeno y apretado',
        ],
        'sentiment': [
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
        ],
    }
    train_df = pd.DataFrame(train_data)
    train_path = tmp_path / 'train.csv'
    train_df.to_csv(train_path, index=False)

    # Create val data (10 samples)
    val_data = {
        'comment': [
            'buen gimnasio',
            'mal servicio',
            'excelente lugar',
            'horrible atencion',
            'muy recomendado',
            'no volveria',
            'fantastico',
            'terrible',
            'perfecto',
            'pesimo',
        ],
        'sentiment': [
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'negative',
        ],
    }
    val_df = pd.DataFrame(val_data)
    val_path = tmp_path / 'val.csv'
    val_df.to_csv(val_path, index=False)

    return tmp_path


class TestRunSingleExperimentE2E:
    """End-to-end tests for single experiment execution."""

    def test_run_produces_valid_result(self, fixture_data_dir: Path) -> None:
        """Running an experiment produces a valid result."""
        config = ExperimentConfig(
            train_path=str(fixture_data_dir / 'train.csv'),
            val_path=str(fixture_data_dir / 'val.csv'),
            test_path=str(fixture_data_dir / 'val.csv'),  # Use val as test for simplicity
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            C=1.0,
            penalty='l2',
            output_dir=str(fixture_data_dir / 'experiments'),
            save_predictions=False,
        )

        result = run_single_experiment(config)

        assert result is not None
        assert result.config.run_id is not None
        assert result.validity_status in ['valid', 'constraint_not_met', 'invalid']

    def test_run_produces_metrics(self, fixture_data_dir: Path) -> None:
        """Running an experiment produces val metrics."""
        config = ExperimentConfig(
            train_path=str(fixture_data_dir / 'train.csv'),
            val_path=str(fixture_data_dir / 'val.csv'),
            test_path=str(fixture_data_dir / 'val.csv'),
            ngram_range=(1, 1),
            min_df=1,
            C=1.0,
            output_dir=str(fixture_data_dir / 'experiments'),
            save_predictions=False,
        )

        result = run_single_experiment(config)

        if result.validity_status != 'invalid':
            assert result.val_metrics is not None
            assert 0.0 <= result.val_metrics.f1_neg <= 1.0
            assert 0.0 <= result.val_metrics.recall_neg <= 1.0
            assert 0.0 <= result.val_metrics.threshold <= 1.0


class TestGenerateGridConfigs:
    """Tests for grid configuration generation."""

    def test_generates_correct_number_of_configs(self, fixture_data_dir: Path) -> None:
        """Grid generation produces correct number of configurations."""
        base_config = ExperimentConfig(
            train_path=str(fixture_data_dir / 'train.csv'),
            val_path=str(fixture_data_dir / 'val.csv'),
            test_path=str(fixture_data_dir / 'val.csv'),
            output_dir=str(fixture_data_dir / 'experiments'),
        )

        tfidf_grid = {
            'ngram_range': [(1, 1), (1, 2)],
            'min_df': [1, 2],
            'max_df': [1.0],
            'sublinear_tf': [True],
        }
        logreg_grid = {
            'penalty': ['l2'],
            'C': [0.1, 1.0],
            'class_weight': [None],
        }

        configs = generate_grid_configs(base_config, tfidf_grid, logreg_grid)

        # 2 ngram × 2 min_df × 1 max_df × 1 sublinear × 1 penalty × 2 C × 1 weight = 8
        assert len(configs) == 8


class TestRankResults:
    """Tests for result ranking logic."""

    def test_valid_runs_ranked_above_invalid(self) -> None:
        """Valid runs are ranked higher than invalid runs."""
        from gym_sentiment_guard.experiments.artifacts import RunConfig, RunResult
        from gym_sentiment_guard.experiments.metrics import ValMetrics

        valid_result = RunResult(
            config=RunConfig(run_id='run_valid', timestamp='2025-01-01'),
            val_metrics=ValMetrics(
                f1_neg=0.8,
                recall_neg=0.9,
                precision_neg=0.7,
                macro_f1=0.75,
                pr_auc_neg=0.85,
                threshold=0.5,
                constraint_status='met',
            ),
            validity_status='valid',
        )

        invalid_result = RunResult(
            config=RunConfig(run_id='run_invalid', timestamp='2025-01-01'),
            validity_status='invalid',
            invalidity_reason='Convergence failed',
        )

        ranked = rank_results([invalid_result, valid_result])

        assert ranked[0].config.run_id == 'run_valid'
        assert ranked[1].config.run_id == 'run_invalid'

    def test_constraint_met_ranked_above_not_met(self) -> None:
        """Constraint-meeting runs are ranked higher."""
        from gym_sentiment_guard.experiments.artifacts import RunConfig, RunResult
        from gym_sentiment_guard.experiments.metrics import ValMetrics

        met_result = RunResult(
            config=RunConfig(run_id='run_met', timestamp='2025-01-01'),
            val_metrics=ValMetrics(
                f1_neg=0.7,
                recall_neg=0.92,
                precision_neg=0.6,
                macro_f1=0.70,
                pr_auc_neg=0.75,
                threshold=0.4,
                constraint_status='met',
            ),
            validity_status='valid',
        )

        not_met_result = RunResult(
            config=RunConfig(run_id='run_not_met', timestamp='2025-01-01'),
            val_metrics=ValMetrics(
                f1_neg=0.85,
                recall_neg=0.80,
                precision_neg=0.9,  # Higher F1 but lower recall
                macro_f1=0.80,
                pr_auc_neg=0.88,
                threshold=0.6,
                constraint_status='not_met',
            ),
            validity_status='constraint_not_met',
        )

        ranked = rank_results([not_met_result, met_result])

        assert ranked[0].config.run_id == 'run_met'

    def test_higher_f1_neg_ranked_first(self) -> None:
        """Among valid, constraint-meeting runs, higher F1_neg wins."""
        from gym_sentiment_guard.experiments.artifacts import RunConfig, RunResult
        from gym_sentiment_guard.experiments.metrics import ValMetrics

        high_f1 = RunResult(
            config=RunConfig(run_id='high_f1', timestamp='2025-01-01'),
            val_metrics=ValMetrics(
                f1_neg=0.90,
                recall_neg=0.92,
                precision_neg=0.88,
                macro_f1=0.85,
                pr_auc_neg=0.90,
                threshold=0.45,
                constraint_status='met',
            ),
            validity_status='valid',
        )

        low_f1 = RunResult(
            config=RunConfig(run_id='low_f1', timestamp='2025-01-01'),
            val_metrics=ValMetrics(
                f1_neg=0.75,
                recall_neg=0.95,
                precision_neg=0.60,
                macro_f1=0.70,
                pr_auc_neg=0.80,
                threshold=0.35,
                constraint_status='met',
            ),
            validity_status='valid',
        )

        ranked = rank_results([low_f1, high_f1])

        assert ranked[0].config.run_id == 'high_f1'
