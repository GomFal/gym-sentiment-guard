"""FastAPI application for serving sentiment predictions."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ..config import ServingConfig, load_serving_config
from ..data.cleaning import load_structural_punctuation
from ..utils import get_logger, json_log
from .loader import ModelArtifact, ModelLoadError, load_model
from .predict import predict_batch, predict_single
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
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
    version='0.1.0',
    lifespan=lifespan,
)


def _validate_text_size(text: str, max_bytes: int, context: str = 'text') -> None:
    """Validate text size against limit."""
    text_bytes = len(text.encode('utf-8'))
    if text_bytes > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f'{context} exceeds maximum size of {max_bytes} bytes ({text_bytes} bytes provided)',
        )


def _log_request(
    endpoint: str,
    input_length: int,
    prediction: str | None,
    latency_ms: float,
) -> None:
    """Log request if request logging is enabled."""
    if _config and _config.logging.mode == 'requests':
        log.info(
            json_log(
                'serving.request',
                component='serving.app',
                endpoint=endpoint,
                input_length=input_length,
                prediction=prediction,
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
    response_model=PredictResponse,
    responses={400: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    tags=['Prediction'],
)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a single review."""
    if _artifact is None or _config is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    start_time = time.perf_counter()

    # Validate text size
    _validate_text_size(request.text, _config.validation.max_text_bytes)

    # Make prediction
    result = predict_single(
        text=request.text,
        artifact=_artifact,
        apply_preprocessing=_config.preprocessing.enabled,
        structural_punctuation=_structural_punctuation,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000
    _log_request('/predict', len(request.text), result.sentiment, latency_ms)

    return PredictResponse(
        sentiment=result.sentiment,
        confidence=result.confidence,
        probability_positive=result.probability_positive,
        probability_negative=result.probability_negative,
        model_version=_artifact.version,
    )


@app.post(
    '/predict/batch',
    response_model=BatchPredictResponse,
    responses={400: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    tags=['Prediction'],
)
def predict_batch_endpoint(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict sentiment for multiple reviews."""
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
    results = predict_batch(
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
    _log_request('/predict/batch', len(request.texts), None, latency_ms)

    return BatchPredictResponse(predictions=predictions)
