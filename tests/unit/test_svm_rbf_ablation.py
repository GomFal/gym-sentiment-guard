"""Unit tests for SVM RBF ablation module."""

from pathlib import Path

import pytest

from gym_sentiment_guard.models.svm.experiments.runner_rbf import SVCRBFExperimentConfig
from gym_sentiment_guard.models.svm.experiments.ablation_rbf import generate_rbf_grid_configs


class TestGenerateRBFGridConfigs:
    """Tests for generate_rbf_grid_configs function."""

    def test_generates_correct_number_of_configs(self) -> None:
        """Grid generation produces correct number of configurations."""
        base_config = SVCRBFExperimentConfig(
            train_path='train.csv',
            val_path='val.csv',
            test_path='test.csv',
        )

        unigram_grid = {
            'ngram_range': [(1, 1)],
            'min_df': [10],
            'max_df': [0.90],
            'sublinear_tf': [True],
            'stop_words': ['curated_safe'],
        }

        multigram_grid = {
            'ngram_range': [(2, 3)],
            'min_df': [2],
            'max_df': [0.90],
            'sublinear_tf': [True],
            'stop_words': [None],
        }

        # 4 C values × 4 gamma values = 16 configs
        svc_grid = {
            'kernel': ['rbf'],
            'C': [1.0, 3.0, 5.0, 10.0],
            'gamma': ['scale', 0.1, 0.01, 0.001],
            'tol': [0.001],
            'cache_size': [1000],
            'max_iter': [-1],
        }

        configs = generate_rbf_grid_configs(base_config, unigram_grid, multigram_grid, svc_grid)

        assert len(configs) == 16  # 4 C × 4 gamma

    def test_configs_have_correct_types(self) -> None:
        """Generated configs are of correct type."""
        base_config = SVCRBFExperimentConfig(
            train_path='train.csv',
            val_path='val.csv',
            test_path='test.csv',
        )

        unigram_grid = {'ngram_range': [(1, 1)], 'min_df': [10], 'max_df': [0.90], 'sublinear_tf': [True], 'stop_words': ['curated_safe']}
        multigram_grid = {'ngram_range': [(2, 3)], 'min_df': [2], 'max_df': [0.90], 'sublinear_tf': [True], 'stop_words': [None]}
        svc_grid = {'kernel': ['rbf'], 'C': [1.0], 'gamma': ['scale'], 'tol': [0.001], 'cache_size': [1000], 'max_iter': [-1]}

        configs = generate_rbf_grid_configs(base_config, unigram_grid, multigram_grid, svc_grid)

        assert len(configs) == 1
        assert isinstance(configs[0], SVCRBFExperimentConfig)
        assert configs[0].kernel == 'rbf'
        assert configs[0].C == 1.0
        assert configs[0].gamma == 'scale'

    def test_configs_inherit_base_paths(self) -> None:
        """Generated configs inherit data paths from base config."""
        base_config = SVCRBFExperimentConfig(
            train_path='my_train.csv',
            val_path='my_val.csv',
            test_path='my_test.csv',
            output_dir='my_output',
        )

        unigram_grid = {'ngram_range': [(1, 1)], 'min_df': [10], 'max_df': [0.90], 'sublinear_tf': [True], 'stop_words': ['curated_safe']}
        multigram_grid = {'ngram_range': [(2, 3)], 'min_df': [2], 'max_df': [0.90], 'sublinear_tf': [True], 'stop_words': [None]}
        svc_grid = {'kernel': ['rbf'], 'C': [1.0, 2.0], 'gamma': ['scale'], 'tol': [0.001], 'cache_size': [1000], 'max_iter': [-1]}

        configs = generate_rbf_grid_configs(base_config, unigram_grid, multigram_grid, svc_grid)

        for config in configs:
            assert config.train_path == 'my_train.csv'
            assert config.val_path == 'my_val.csv'
            assert config.test_path == 'my_test.csv'
            assert config.output_dir == 'my_output'
