"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request schema for single prediction."""

    text: str = Field(
        ...,
        min_length=1,
        description='Review text to classify.',
        examples=['Excelente gimnasio, muy limpio y buen ambiente.'],
    )

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError('Text cannot be empty or whitespace only')
        return v


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description='List of review texts to classify.',
    )

    @field_validator('texts')
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty or whitespace only')
        return v


class PredictResponse(BaseModel):
    """Response schema for single prediction with full probabilities."""

    sentiment: str = Field(
        ...,
        description='Predicted sentiment: "positive" or "negative".',
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Confidence of the predicted sentiment.',
    )
    probability_positive: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Probability of positive sentiment.',
    )
    probability_negative: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Probability of negative sentiment.',
    )
    model_version: str = Field(
        ...,
        description='Version of the model used for prediction.',
    )


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: list[PredictResponse] = Field(
        ...,
        description='List of predictions for each input text.',
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(default='ok', description='Service health status.')


class ReadyResponse(BaseModel):
    """Response schema for readiness check endpoint."""

    status: str = Field(..., description='Readiness status.')
    model_loaded: bool = Field(..., description='Whether model is loaded.')


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""

    model_name: str = Field(..., description='Name of the model.')
    version: str = Field(..., description='Model version.')
    threshold: float = Field(..., description='Decision threshold.')
    target_class: str = Field(..., description='Class targeted by threshold.')
    label_mapping: dict[str, int] = Field(..., description='Label to integer mapping.')


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    detail: str = Field(..., description='Error message.')
