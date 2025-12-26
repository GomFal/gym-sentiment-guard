"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request schema for prediction (1 to N texts)."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description='List of review texts to classify (1 to 100 texts).',
        examples=[['Excelente gimnasio, muy limpio.', 'Pésimo servicio, no volvería.']],
    )

    @field_validator('texts')
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty or whitespace only')
        return v


class PredictResponse(BaseModel):
    """Response schema for a single prediction with full probabilities."""

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


class FeatureImportance(BaseModel):
    """Schema for a single feature's contribution to the prediction."""

    feature: str = Field(..., description='The word/token.')
    importance: float = Field(..., description='Signed contribution score.')


class ExplainResponse(PredictResponse):
    """Prediction response with feature importance explanation."""

    explanation: list[FeatureImportance] = Field(
        default_factory=list,
        description='Top contributing features, sorted by absolute importance.',
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
