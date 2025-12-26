"""Model loading utilities for the serving module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from ..utils import get_logger, json_log

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelArtifact:
    """Container for loaded model and its metadata."""

    model: Pipeline
    metadata: dict[str, Any]
    version: str
    threshold: float
    target_class: str
    label_mapping: dict[str, int]
    model_name: str


class ModelLoadError(RuntimeError):
    """Raised when model loading fails."""


class ModelExplainError(RuntimeError):
    """Raised when model does not support explanation (e.g., non-linear, no coef_)."""


def load_model(model_dir: str | Path) -> ModelArtifact:
    """
    Load a trained model and its metadata from a directory.

    Args:
        model_dir: Path to directory containing logreg.joblib and metadata.json.

    Returns:
        ModelArtifact containing the model and its configuration.

    Raises:
        ModelLoadError: If model files are missing or corrupted.
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise ModelLoadError(f'Model directory not found: {model_path}')

    joblib_path = model_path / 'logreg.joblib'
    metadata_path = model_path / 'metadata.json'

    if not joblib_path.exists():
        raise ModelLoadError(f'Model file not found: {joblib_path}')

    if not metadata_path.exists():
        raise ModelLoadError(f'Metadata file not found: {metadata_path}')

    try:
        model = joblib.load(joblib_path)
    except Exception as exc:
        raise ModelLoadError(f'Failed to load model: {exc}') from exc

    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelLoadError(f'Failed to load metadata: {exc}') from exc

    version = metadata.get('version', 'unknown')
    threshold = metadata.get('threshold', 0.5)
    target_class = metadata.get('threshold_target_class', 'negative')
    label_mapping = metadata.get('label_mapping', {'negative': 0, 'positive': 1})

    log.info(
        json_log(
            'model.loaded',
            component='serving.loader',
            model_dir=str(model_path),
            version=version,
            threshold=threshold,
            target_class=target_class,
        )
    )

    return ModelArtifact(
        model=model,
        metadata=metadata,
        version=version,
        threshold=threshold,
        target_class=target_class,
        label_mapping=label_mapping,
        model_name=metadata.get('model_name', 'unknown'),
    )
