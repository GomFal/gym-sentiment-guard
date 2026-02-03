"""Unit tests for SVM RBF runner module."""

from pathlib import Path

import pandas as pd
import pytest

from gym_sentiment_guard.models.svm.experiments.runner_rbf import (
    SVCRBFExperimentConfig,
    run_single_rbf_experiment,
)
from gym_sentiment_guard.models.svm.pipelines import build_rbf_pipeline


@pytest.fixture
def mock_rbf_data_paths(tmp_path):
    """Create minimal train/val/test CSVs for testing."""
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    # Create train data (small for fast testing)
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
        ],
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
        ],
    }
    pd.DataFrame(train_data).to_csv(train_path, index=False)

    # Create val data
    val_data = {
        'comment': [
            'buen gimnasio',
            'mal servicio',
            'excelente lugar',
            'horrible atencion',
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative'],
    }
    pd.DataFrame(val_data).to_csv(val_path, index=False)

    # Create test data
    pd.DataFrame(val_data).to_csv(test_path, index=False)

    return train_path, val_path, test_path


class TestSVCRBFExperimentConfig:
    """Tests for SVCRBFExperimentConfig dataclass."""

    def test_config_instantiation(self) -> None:
        """Config can be instantiated with required paths."""
        config = SVCRBFExperimentConfig(
            train_path='train.csv',
            val_path='val.csv',
            test_path='test.csv',
        )

        assert config.train_path == 'train.csv'
        assert config.kernel == 'rbf'
        assert config.gamma == 'scale'
        assert config.C == 1.0

    def test_config_with_custom_gamma(self) -> None:
        """Config accepts gamma as string or float."""
        config_scale = SVCRBFExperimentConfig(
            train_path='a', val_path='b', test_path='c', gamma='scale'
        )
        assert config_scale.gamma == 'scale'

        config_auto = SVCRBFExperimentConfig(
            train_path='a', val_path='b', test_path='c', gamma='auto'
        )
        assert config_auto.gamma == 'auto'

        config_float = SVCRBFExperimentConfig(
            train_path='a', val_path='b', test_path='c', gamma=0.1
        )
        assert config_float.gamma == 0.1

    def test_config_defaults(self) -> None:
        """Config has sensible defaults for RBF SVM."""
        config = SVCRBFExperimentConfig(
            train_path='a', val_path='b', test_path='c'
        )

        assert config.kernel == 'rbf'
        assert config.tol == 0.001
        assert config.cache_size == 1000
        assert config.max_iter == -1
        assert config.output_dir == 'artifacts/experiments/svm_rbf'


class TestBuildRBFPipeline:
    """Tests for build_rbf_pipeline function."""

    def test_build_pipeline_returns_pipeline(self) -> None:
        """Pipeline builder returns a valid sklearn Pipeline."""
        pipeline = build_rbf_pipeline()

        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict_proba')

    def test_pipeline_has_expected_steps(self) -> None:
        """Pipeline has features and classifier steps."""
        pipeline = build_rbf_pipeline()

        assert 'features' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps


class TestLabelMappingValidation:
    """Tests for label mapping validation in RBF runner."""

    def test_unmapped_train_labels_raises(self, mock_rbf_data_paths, tmp_path) -> None:
        """ValueError raised for unmapped train labels."""
        train_path, val_path, test_path = mock_rbf_data_paths

        # Create invalid train data
        invalid_train_path = tmp_path / "invalid_train.csv"
        pd.DataFrame({
            'comment': ['test comment'],
            'sentiment': ['unknown_label']
        }).to_csv(invalid_train_path, index=False)

        config = SVCRBFExperimentConfig(
            train_path=str(invalid_train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            label_mapping={'positive': 1, 'negative': 0},
        )

        with pytest.raises(ValueError) as excinfo:
            run_single_rbf_experiment(config)

        assert "Unmapped labels" in str(excinfo.value)
        assert "unknown_label" in str(excinfo.value)

    def test_unmapped_val_labels_raises(self, mock_rbf_data_paths, tmp_path) -> None:
        """ValueError raised for unmapped val labels."""
        train_path, val_path, test_path = mock_rbf_data_paths

        # Create invalid val data
        invalid_val_path = tmp_path / "invalid_val.csv"
        pd.DataFrame({
            'comment': ['test comment'],
            'sentiment': ['buggy_label']
        }).to_csv(invalid_val_path, index=False)

        config = SVCRBFExperimentConfig(
            train_path=str(train_path),
            val_path=str(invalid_val_path),
            test_path=str(test_path),
            label_mapping={'positive': 1, 'negative': 0},
        )

        with pytest.raises(ValueError) as excinfo:
            run_single_rbf_experiment(config)

        assert "Unmapped labels" in str(excinfo.value)
        assert "buggy_label" in str(excinfo.value)
