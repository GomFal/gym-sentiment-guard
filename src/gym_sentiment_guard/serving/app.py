"""FastAPI application for serving sentiment predictions."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ..config import ServingConfig, load_serving_config
from ..data.cleaning import load_structural_punctuation
from ..utils import get_logger, json_log
from .loader import ModelArtifact, ModelExplainError, ModelLoadError, load_model
from .predict import explain_predictions, predict
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

# Global state
_artifact: ModelArtifact | None = None
_config: ServingConfig | None = None
_structural_punctuation: str | None = None

log = get_logger(__name__)


def _get_config_path() -> Path:
    """Get the config path from environment or default."""
    import os

    config_path = os.getenv('GSG_SERVING_CONFIG', 'configs/serving.yaml')
    return Path(config_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _artifact, _config, _structural_punctuation

    config_path = _get_config_path()
    log.info(
        json_log(
            'serving.startup',
            component='serving.app',
            config_path=str(config_path),
        )
    )

    try:
        _config = load_serving_config(config_path)
        _artifact = load_model(_config.model.path)

        if _config.preprocessing.structural_punctuation_path:
            _structural_punctuation = load_structural_punctuation(
                _config.preprocessing.structural_punctuation_path
            )

        log.info(
            json_log(
                'serving.ready',
                component='serving.app',
                model_version=_artifact.version,
                model_name=_artifact.model_name,
            )
        )
    except (FileNotFoundError, ModelLoadError) as exc:
        log.error(
            json_log(
                'serving.startup_error',
                component='serving.app',
                error=str(exc),
            )
        )
        raise

    yield

    log.info(json_log('serving.shutdown', component='serving.app'))


app = FastAPI(
    title='Gym Sentiment Guard API',
    description='Sentiment analysis API for gym reviews.',
    version='0.2.0',
    lifespan=lifespan,
)


def _validate_text_size(text: str, max_bytes: int, context: str = 'text') -> None:
    """Validate text size against limit."""
    text_bytes = len(text.encode('utf-8'))
    if text_bytes > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f'{context} exceeds maximum size of {max_bytes} bytes '
            f'({text_bytes} bytes provided)',
        )


def _log_request(
    endpoint: str,
    input_count: int,
    latency_ms: float,
) -> None:
    """Log request if request logging is enabled."""
    if _config and _config.logging.mode == 'requests':
        log.info(
            json_log(
                'serving.request',
                component='serving.app',
                endpoint=endpoint,
                input_count=input_count,
                latency_ms=round(latency_ms, 2),
            )
        )


@app.get('/health', response_model=HealthResponse, tags=['Health'])
def health() -> HealthResponse:
    """Liveness check endpoint."""
    return HealthResponse(status='ok')


@app.get('/ready', response_model=ReadyResponse, tags=['Health'])
def ready() -> ReadyResponse:
    """Readiness check endpoint."""
    model_loaded = _artifact is not None
    status = 'ready' if model_loaded else 'not_ready'
    return ReadyResponse(status=status, model_loaded=model_loaded)


@app.get('/model/info', response_model=ModelInfoResponse, tags=['Model'])
def model_info() -> ModelInfoResponse:
    """Get information about the loaded model."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    return ModelInfoResponse(
        model_name=_artifact.model_name,
        version=_artifact.version,
        threshold=_artifact.threshold,
        target_class=_artifact.target_class,
        label_mapping=_artifact.label_mapping,
    )


@app.post(
    '/predict',
    response_model=list[PredictResponse],
    responses={400: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    tags=['Prediction'],
)
def predict_endpoint(request: PredictRequest) -> list[PredictResponse]:
    """
    Predict sentiment for one or more reviews.

    Accepts 1 to 100 texts and returns a list of predictions.
    """
    if _artifact is None or _config is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    start_time = time.perf_counter()

    # Validate batch size
    if len(request.texts) > _config.batch.max_items:
        raise HTTPException(
            status_code=400,
            detail=f'Batch size {len(request.texts)} exceeds maximum of {_config.batch.max_items}',
        )

    # Validate individual text sizes
    for i, text in enumerate(request.texts):
        _validate_text_size(
            text,
            _config.batch.max_text_bytes_per_item,
            context=f'Text at index {i}',
        )

    # Make predictions
    results = predict(
        texts=request.texts,
        artifact=_artifact,
        apply_preprocessing=_config.preprocessing.enabled,
        structural_punctuation=_structural_punctuation,
    )

    predictions = [
        PredictResponse(
            sentiment=result.sentiment,
            confidence=result.confidence,
            probability_positive=result.probability_positive,
            probability_negative=result.probability_negative,
            model_version=_artifact.version,
        )
        for result in results
    ]

    latency_ms = (time.perf_counter() - start_time) * 1000
    _log_request('/predict', len(request.texts), latency_ms)

    return predictions


@app.post(
    '/predict/explain',
    response_model=list[ExplainResponse],
    responses={400: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    tags=['Prediction', 'Explainability'],
)
def explain_endpoint(request: PredictRequest) -> list[ExplainResponse]:
    """
    Predict sentiment with feature importance explanations.

    Accepts 1 to 100 texts and returns predictions with the top contributing
    features for each prediction.
    """
    if _artifact is None or _config is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    start_time = time.perf_counter()

    # Validate batch size
    if len(request.texts) > _config.batch.max_items:
        raise HTTPException(
            status_code=400,
            detail=f'Batch size {len(request.texts)} exceeds maximum of {_config.batch.max_items}',
        )

    # Validate individual text sizes
    for i, text in enumerate(request.texts):
        _validate_text_size(
            text,
            _config.batch.max_text_bytes_per_item,
            context=f'Text at index {i}',
        )

    try:
        results = explain_predictions(
            texts=request.texts,
            artifact=_artifact,
            apply_preprocessing=_config.preprocessing.enabled,
            structural_punctuation=_structural_punctuation,
        )
    except ModelExplainError as exc:
        raise HTTPException(
            status_code=400,
            detail=f'Model does not support explanation: {exc}',
        ) from exc

    responses = [
        ExplainResponse(
            sentiment=result.sentiment,
            confidence=result.confidence,
            probability_positive=result.probability_positive,
            probability_negative=result.probability_negative,
            model_version=_artifact.version,
            explanation=[
                FeatureImportance(feature=feat, importance=imp) for feat, imp in result.explanation
            ],
        )
        for result in results
    ]

    latency_ms = (time.perf_counter() - start_time) * 1000
    _log_request('/predict/explain', len(request.texts), latency_ms)

    return responses
