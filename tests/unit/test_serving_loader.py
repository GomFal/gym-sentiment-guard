"""Unit tests for serving module loader."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.serving.loader import (
    ModelArtifact,
    ModelLoadError,
    load_model,
)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory with valid artifacts."""
    model_path = tmp_path / 'model'
    model_path.mkdir()

    # Create a simple model
    vectorizer = TfidfVectorizer(lowercase=True)
    classifier = LogisticRegression()
    pipeline = Pipeline([('tfidf', vectorizer), ('logreg', classifier)])

    # Fit on minimal data
    texts = ['good gym', 'bad gym', 'great service', 'terrible place']
    labels = [1, 0, 1, 0]
    pipeline.fit(texts, labels)

    # Save model
    joblib.dump(pipeline, model_path / 'logreg.joblib')

    # Save metadata
    metadata = {
        'model_name': 'test_model',
        'version': '2025-01-01_001',
        'threshold': 0.5,
        'threshold_target_class': 'negative',
        'label_mapping': {'negative': 0, 'positive': 1},
    }
    (model_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    return model_path


def test_load_model_success(model_dir: Path) -> None:
    """Test successful model loading."""
    artifact = load_model(model_dir)

    assert isinstance(artifact, ModelArtifact)
    assert artifact.version == '2025-01-01_001'
    assert artifact.threshold == 0.5
    assert artifact.target_class == 'negative'
    assert artifact.label_mapping == {'negative': 0, 'positive': 1}
    assert artifact.model_name == 'test_model'


def test_load_model_missing_directory() -> None:
    """Test error when model directory doesn't exist."""
    with pytest.raises(ModelLoadError, match='Model directory not found'):
        load_model('/nonexistent/path')


def test_load_model_missing_joblib(tmp_path: Path) -> None:
    """Test error when logreg.joblib is missing."""
    model_path = tmp_path / 'model'
    model_path.mkdir()
    (model_path / 'metadata.json').write_text('{}', encoding='utf-8')

    with pytest.raises(ModelLoadError, match='Model file not found'):
        load_model(model_path)


def test_load_model_missing_metadata(tmp_path: Path) -> None:
    """Test error when metadata.json is missing."""
    model_path = tmp_path / 'model'
    model_path.mkdir()

    # Create minimal model with two classes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('logreg', LogisticRegression()),
    ])
    pipeline.fit(['good', 'bad'], [1, 0])
    joblib.dump(pipeline, model_path / 'logreg.joblib')

    with pytest.raises(ModelLoadError, match='Metadata file not found'):
        load_model(model_path)


def test_load_model_invalid_json(tmp_path: Path) -> None:
    """Test error when metadata.json is invalid."""
    model_path = tmp_path / 'model'
    model_path.mkdir()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('logreg', LogisticRegression()),
    ])
    pipeline.fit(['good', 'bad'], [1, 0])
    joblib.dump(pipeline, model_path / 'logreg.joblib')
    (model_path / 'metadata.json').write_text('invalid json', encoding='utf-8')

    with pytest.raises(ModelLoadError, match='Failed to load metadata'):
        load_model(model_path)
