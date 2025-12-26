"""Serving module for gym_sentiment_guard."""

from .app import app
from .loader import ModelArtifact, ModelExplainError, ModelLoadError, load_model
from .predict import (
    ExplanationResult,
    PredictionResult,
    explain_predictions,
    predict,
    preprocess_text,
)
from .schemas import (
    ErrorResponse,
    ExplainResponse,
    FeatureImportance,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

__all__ = [
    'app',
    'ModelArtifact',
    'ModelExplainError',
    'ModelLoadError',
    'load_model',
    'ExplanationResult',
    'PredictionResult',
    'explain_predictions',
    'predict',
    'preprocess_text',
    'ErrorResponse',
    'ExplainResponse',
    'FeatureImportance',
    'HealthResponse',
    'ModelInfoResponse',
    'PredictRequest',
    'PredictResponse',
    'ReadyResponse',
]
