"""Integration tests for serving API endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory with valid artifacts."""
    model_path = tmp_path / 'model'
    model_path.mkdir()

    vectorizer = TfidfVectorizer(lowercase=True)
    classifier = LogisticRegression()
    pipeline = Pipeline([('tfidf', vectorizer), ('logreg', classifier)])

    texts = ['excelente gym bueno', 'terrible malo horrible', 'fantastico genial', 'pesimo fatal']
    labels = [1, 0, 1, 0]
    pipeline.fit(texts, labels)

    joblib.dump(pipeline, model_path / 'logreg.joblib')

    metadata = {
        'model_name': 'test_sentiment',
        'version': '2025-01-01_001',
        'threshold': 0.5,
        'threshold_target_class': 'negative',
        'label_mapping': {'negative': 0, 'positive': 1},
    }
    (model_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    return model_path


@pytest.fixture
def serving_config(tmp_path: Path, model_dir: Path) -> Path:
    """Create a serving config file."""
    config_path = tmp_path / 'serving.yaml'
    config_content = f"""
model:
  path: {model_dir}

server:
  host: 0.0.0.0
  port: 8001

preprocessing:
  enabled: true

validation:
  max_text_bytes: 51200
  min_text_length: 1

batch:
  max_items: 100
  max_text_bytes_per_item: 5120

logging:
  mode: minimal
"""
    config_path.write_text(config_content, encoding='utf-8')
    return config_path


@pytest.fixture
def client(serving_config: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a test client with configured app."""
    # Set env var before importing app module
    monkeypatch.setenv('GSG_SERVING_CONFIG', str(serving_config))

    # Remove cached app module to force re-import with new env var
    modules_to_remove = [key for key in sys.modules if 'gym_sentiment_guard.serving' in key]
    for module in modules_to_remove:
        del sys.modules[module]

    # Now import the app (will use new env var)
    from fastapi.testclient import TestClient

    from gym_sentiment_guard.serving.app import app

    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_returns_ok(self, client) -> None:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'ok'

    def test_ready_returns_ready(self, client) -> None:
        response = client.get('/ready')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ready'
        assert data['model_loaded'] is True


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_returns_metadata(self, client) -> None:
        response = client.get('/model/info')
        assert response.status_code == 200
        data = response.json()
        assert data['model_name'] == 'test_sentiment'
        assert data['version'] == '2025-01-01_001'
        assert data['threshold'] == 0.5
        assert data['target_class'] == 'negative'


class TestPredictEndpoint:
    """Tests for unified prediction endpoint."""

    def test_predict_single_text(self, client) -> None:
        response = client.post('/predict', json={'texts': ['excelente fantastico genial']})
        assert response.status_code == 200
        data = response.json()

        # Should return a list
        assert isinstance(data, list)
        assert len(data) == 1

        # Check prediction structure
        prediction = data[0]
        assert 'sentiment' in prediction
        assert 'confidence' in prediction
        assert 'probability_positive' in prediction
        assert 'probability_negative' in prediction
        assert 'model_version' in prediction

    def test_predict_multiple_texts(self, client) -> None:
        response = client.post(
            '/predict',
            json={'texts': ['good review', 'bad review', 'neutral']},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_predict_returns_probabilities(self, client) -> None:
        response = client.post('/predict', json={'texts': ['test review']})
        assert response.status_code == 200
        prediction = response.json()[0]

        # Probabilities should sum to ~1
        total = prediction['probability_positive'] + prediction['probability_negative']
        assert abs(total - 1.0) < 0.01

    def test_predict_empty_list_rejected(self, client) -> None:
        response = client.post('/predict', json={'texts': []})
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_whitespace_only_rejected(self, client) -> None:
        response = client.post('/predict', json={'texts': ['   ']})
        assert response.status_code == 422


class TestValidation:
    """Tests for input validation."""

    def test_text_too_large_rejected(self, client) -> None:
        # Create text larger than 5KB per item
        large_text = 'a' * 6000
        response = client.post('/predict', json={'texts': [large_text]})
        assert response.status_code == 400
        assert 'exceeds maximum size' in response.json()['detail']
