"""Core prediction logic for the serving module."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.cleaning import DEFAULT_STRUCTURAL_PUNCTUATION, EMOJI_PATTERN
from .loader import ModelArtifact


@dataclass(frozen=True)
class PredictionResult:
    """Result of a single prediction."""

    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float


def preprocess_text(
    text: str,
    structural_punctuation: str | None = None,
) -> str:
    """
    Apply the same preprocessing as training (single text).

    Steps:
    1. Strip emojis
    2. Lowercase
    3. Replace structural punctuation with spaces
    4. Collapse whitespace
    5. Strip leading/trailing whitespace
    """
    # Strip emojis
    cleaned = EMOJI_PATTERN.sub('', text)

    # Lowercase
    cleaned = cleaned.lower()

    # Replace structural punctuation with spaces
    pattern = structural_punctuation or DEFAULT_STRUCTURAL_PUNCTUATION
    cleaned = re.sub(pattern, ' ', cleaned)

    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Strip
    cleaned = cleaned.strip()

    return cleaned


def preprocess_texts_vectorized(
    texts: list[str],
    structural_punctuation: str | None = None,
) -> list[str]:
    """
    Apply preprocessing to multiple texts using vectorized pandas operations.

    This is significantly faster than calling preprocess_text() in a loop
    for large batches (10x+ speedup for 100+ texts).

    Args:
        texts: List of input texts.
        structural_punctuation: Optional regex pattern for punctuation.

    Returns:
        List of preprocessed texts.
    """
    pattern = structural_punctuation or DEFAULT_STRUCTURAL_PUNCTUATION

    series = pd.Series(texts)

    # Vectorized operations (all performed on entire series at once)
    processed = (
        series.str.replace(EMOJI_PATTERN, '', regex=True)  # Strip emojis
        .str.lower()  # Lowercase
        .str.replace(pattern, ' ', regex=True)  # Structural punctuation
        .str.replace(r'\s+', ' ', regex=True)  # Collapse whitespace
        .str.strip()  # Strip leading/trailing
    )

    return processed.tolist()


def predict(
    texts: list[str],
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
    structural_punctuation: str | None = None,
) -> list[PredictionResult]:
    """
    Make predictions for one or more texts.

    This is the unified prediction function that always returns a list,
    even for single-text predictions.

    Args:
        texts: List of input review texts (1 to N texts).
        artifact: Loaded model artifact.
        apply_preprocessing: Whether to apply preprocessing.
        structural_punctuation: Optional custom punctuation pattern.

    Returns:
        List of PredictionResult objects (same length as input texts).
    """
    # Handle empty list case
    if not texts:
        return []

    if apply_preprocessing:
        processed_texts = preprocess_texts_vectorized(texts, structural_punctuation)
    else:
        processed_texts = texts

    # Get probabilities from model (returns numpy array)
    probas = artifact.model.predict_proba(processed_texts)

    # Get class indices
    classes = artifact.model.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Map labels to indices
    label_mapping = artifact.label_mapping
    negative_label = label_mapping.get('negative', 0)
    positive_label = label_mapping.get('positive', 1)

    neg_idx = class_to_idx[negative_label]
    pos_idx = class_to_idx[positive_label]

    # Vectorized probability extraction (single numpy operation, not N loops)
    prob_negatives = probas[:, neg_idx]
    prob_positives = probas[:, pos_idx]

    # Vectorized threshold comparison
    threshold = artifact.threshold
    target_class = artifact.target_class

    if target_class == 'negative':
        is_target_class = prob_negatives >= threshold
        sentiments = np.where(is_target_class, 'negative', 'positive')
        confidences = np.where(is_target_class, prob_negatives, prob_positives)
    else:
        is_target_class = prob_positives >= threshold
        sentiments = np.where(is_target_class, 'positive', 'negative')
        confidences = np.where(is_target_class, prob_positives, prob_negatives)

    # Create PredictionResult objects
    results = [
        PredictionResult(
            sentiment=str(sentiments[i]),
            confidence=float(confidences[i]),
            probability_positive=float(prob_positives[i]),
            probability_negative=float(prob_negatives[i]),
        )
        for i in range(len(texts))
    ]

    return results

