"""Unit tests for SVM pipeline builders.

Tests the pipeline construction functions in gym_sentiment_guard.models.svm.pipelines.
"""

from __future__ import annotations

import pytest
from sklearn.preprocessing import MaxAbsScaler

from gym_sentiment_guard.models.svm.pipelines import (
    build_feature_union,
    build_linear_pipeline,
    build_rbf_pipeline,
)


class TestBuildFeatureUnion:
    """Tests for build_feature_union function."""

    def test_returns_feature_union_with_two_transformers(self):
        """FeatureUnion should contain unigrams and multigrams."""
        fu = build_feature_union()
        
        assert len(fu.transformer_list) == 2
        names = [name for name, _ in fu.transformer_list]
        assert 'unigrams' in names
        assert 'multigrams' in names

    def test_custom_ngram_ranges(self):
        """Should accept custom ngram ranges."""
        fu = build_feature_union(
            unigram_ngram_range=(1, 2),
            multigram_ngram_range=(3, 4),
        )
        
        unigram = fu.transformer_list[0][1]
        multigram = fu.transformer_list[1][1]
        
        assert unigram.ngram_range == (1, 2)
        assert multigram.ngram_range == (3, 4)


class TestBuildLinearPipeline:
    """Tests for build_linear_pipeline function."""

    def test_returns_pipeline_with_features_and_classifier(self):
        """Linear pipeline should have features and classifier steps."""
        pipeline = build_linear_pipeline()
        
        assert 'features' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps

    def test_no_scaler_in_linear_pipeline(self):
        """Linear pipeline should NOT have a scaler step."""
        pipeline = build_linear_pipeline()
        
        assert 'scaler' not in pipeline.named_steps


class TestBuildRBFPipeline:
    """Tests for build_rbf_pipeline function."""

    def test_returns_pipeline_with_required_steps(self):
        """RBF pipeline should have features, scaler, and classifier steps."""
        pipeline = build_rbf_pipeline(use_scaler=True)
        
        assert 'features' in pipeline.named_steps
        assert 'scaler' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps

    def test_scaler_is_maxabs_scaler(self):
        """Scaler step should be MaxAbsScaler for sparsity preservation."""
        pipeline = build_rbf_pipeline(use_scaler=True)
        
        assert isinstance(pipeline.named_steps['scaler'], MaxAbsScaler)

    def test_rbf_pipeline_without_scaler(self):
        """RBF pipeline with use_scaler=False should NOT have scaler step."""
        pipeline = build_rbf_pipeline(use_scaler=False)
        
        assert 'features' in pipeline.named_steps
        assert 'scaler' not in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps

    def test_default_use_scaler_is_true(self):
        """Default use_scaler should be True."""
        pipeline = build_rbf_pipeline()  # No explicit use_scaler
        
        assert 'scaler' in pipeline.named_steps

    def test_pipeline_step_order(self):
        """Steps should be in order: features -> scaler -> classifier."""
        pipeline = build_rbf_pipeline(use_scaler=True)
        
        step_names = [name for name, _ in pipeline.steps]
        
        assert step_names == ['features', 'scaler', 'classifier']

    def test_custom_svc_params(self):
        """Should accept custom SVC parameters."""
        pipeline = build_rbf_pipeline(
            C=0.5,
            gamma=0.1,
            cache_size=500,
        )
        
        # The classifier is a CalibratedClassifierCV wrapping SVC
        calibrated = pipeline.named_steps['classifier']
        base_svc = calibrated.estimator
        
        assert base_svc.C == 0.5
        assert base_svc.gamma == 0.1
        assert base_svc.cache_size == 500
