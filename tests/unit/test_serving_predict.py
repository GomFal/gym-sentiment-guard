"""Unit tests for serving module prediction logic."""

from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.serving.loader import ModelArtifact
from gym_sentiment_guard.serving.predict import (
    PredictionResult,
    predict_batch,
    predict_single,
    preprocess_text,
)


@pytest.fixture
def model_artifact() -> ModelArtifact:
    """Create a model artifact for testing."""
    vectorizer = TfidfVectorizer(lowercase=True)
    classifier = LogisticRegression()
    pipeline = Pipeline([('tfidf', vectorizer), ('logreg', classifier)])

    # Train on polar examples
    texts = [
        'excelente gimnasio muy bueno',
        'terrible horrible malo',
        'perfecto increible fantastico',
        'pesimo asqueroso fatal',
    ]
    labels = [1, 0, 1, 0]
    pipeline.fit(texts, labels)

    return ModelArtifact(
        model=pipeline,
        metadata={'model_name': 'test'},
        version='test_v1',
        threshold=0.5,
        target_class='negative',
        label_mapping={'negative': 0, 'positive': 1},
    )


class TestPreprocessText:
    """Tests for preprocess_text function."""

    def test_lowercase(self) -> None:
        result = preprocess_text('HELLO WORLD')
        assert result == 'hello world'

    def test_collapse_whitespace(self) -> None:
        result = preprocess_text('hello    world')
        assert result == 'hello world'

    def test_strip_whitespace(self) -> None:
        result = preprocess_text('  hello world  ')
        assert result == 'hello world'

    def test_remove_punctuation(self) -> None:
        result = preprocess_text('hello, world!')
        # Note: ! is not in DEFAULT_STRUCTURAL_PUNCTUATION
        assert 'hello' in result
        assert 'world' in result
        assert ',' not in result

    def test_emoji_removal(self) -> None:
        result = preprocess_text('great gym! 🏋️💪')
        assert '🏋' not in result
        assert '💪' not in result
        assert 'great' in result


class TestPredictSingle:
    """Tests for predict_single function."""

    def test_returns_prediction_result(self, model_artifact: ModelArtifact) -> None:
        result = predict_single('muy buen gimnasio', model_artifact)

        assert isinstance(result, PredictionResult)
        assert result.sentiment in ('positive', 'negative')
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.probability_positive <= 1.0
        assert 0.0 <= result.probability_negative <= 1.0

    def test_probabilities_sum_to_one(self, model_artifact: ModelArtifact) -> None:
        result = predict_single('test text', model_artifact)

        total = result.probability_positive + result.probability_negative
        assert abs(total - 1.0) < 0.001

    def test_positive_text(self, model_artifact: ModelArtifact) -> None:
        result = predict_single('excelente fantastico increible', model_artifact)
        # Model should predict positive for positive text
        assert result.probability_positive > 0.5

    def test_negative_text(self, model_artifact: ModelArtifact) -> None:
        result = predict_single('terrible horrible malo', model_artifact)
        # Model should predict negative for negative text
        assert result.probability_negative > 0.5

    def test_without_preprocessing(self, model_artifact: ModelArtifact) -> None:
        result = predict_single(
            'TEST TEXT',
            model_artifact,
            apply_preprocessing=False,
        )
        assert isinstance(result, PredictionResult)


class TestPredictBatch:
    """Tests for predict_batch function."""

    def test_returns_list(self, model_artifact: ModelArtifact) -> None:
        texts = ['good', 'bad', 'great']
        results = predict_batch(texts, model_artifact)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_result_is_prediction_result(self, model_artifact: ModelArtifact) -> None:
        texts = ['text one', 'text two']
        results = predict_batch(texts, model_artifact)

        for result in results:
            assert isinstance(result, PredictionResult)

    def test_empty_list(self, model_artifact: ModelArtifact) -> None:
        results = predict_batch([], model_artifact)
        assert results == []

    def test_single_item_batch(self, model_artifact: ModelArtifact) -> None:
        results = predict_batch(['single text'], model_artifact)
        assert len(results) == 1
