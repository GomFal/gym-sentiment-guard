"""Model protocols defining standard interfaces."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.pipeline import Pipeline


class ModelProtocol(Protocol):
    """Protocol for sentiment model implementations.

    All model types (LogReg, SVM, etc.) should implement this interface.
    """

    def train(self, config_path: str | Path) -> dict[str, Any]:
        """Train model from config file."""
        ...

    def build_pipeline(self, config: Any) -> Pipeline:
        """Build sklearn pipeline for this model type."""
        ...

    def predict_proba(self, pipeline: Pipeline, X: Any) -> np.ndarray:
        """Get probability predictions."""
        ...
