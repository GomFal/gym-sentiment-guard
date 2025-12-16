"""Core prediction logic for the serving module."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data.cleaning import DEFAULT_STRUCTURAL_PUNCTUATION, EMOJI_PATTERN
from .loader import ModelArtifact


@dataclass
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
    Apply the same preprocessing as training.

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


def predict_single(
    text: str,
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
    structural_punctuation: str | None = None,
) -> PredictionResult:
    """
    Make a prediction for a single text.

    Args:
        text: Input review text.
        artifact: Loaded model artifact.
        apply_preprocessing: Whether to apply preprocessing.
        structural_punctuation: Optional custom punctuation pattern.

    Returns:
        PredictionResult with sentiment and probabilities.
    """
    processed_text = (
        preprocess_text(text, structural_punctuation) if apply_preprocessing else text
    )

    # Get probabilities from model
    proba = artifact.model.predict_proba([processed_text])[0]

    # Get class indices
    classes = artifact.model.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Map labels to indices
    label_mapping = artifact.label_mapping
    negative_label = label_mapping.get('negative', 0)
    positive_label = label_mapping.get('positive', 1)

    prob_negative = float(proba[class_to_idx[negative_label]])
    prob_positive = float(proba[class_to_idx[positive_label]])

    # Apply threshold-based classification
    threshold = artifact.threshold
    target_class = artifact.target_class

    if target_class == 'negative':
        # If targeting negative, predict negative when prob_negative >= threshold
        if prob_negative >= threshold:
            sentiment = 'negative'
            confidence = prob_negative
        else:
            sentiment = 'positive'
            confidence = prob_positive
    else:
        # If targeting positive, predict positive when prob_positive >= threshold
        if prob_positive >= threshold:
            sentiment = 'positive'
            confidence = prob_positive
        else:
            sentiment = 'negative'
            confidence = prob_negative

    return PredictionResult(
        sentiment=sentiment,
        confidence=confidence,
        probability_positive=prob_positive,
        probability_negative=prob_negative,
    )


def predict_batch(
    texts: list[str],
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
    structural_punctuation: str | None = None,
) -> list[PredictionResult]:
    """
    Make predictions for multiple texts.

    Args:
        texts: List of input review texts.
        artifact: Loaded model artifact.
        apply_preprocessing: Whether to apply preprocessing.
        structural_punctuation: Optional custom punctuation pattern.

    Returns:
        List of PredictionResult objects.
    """
    # Handle empty list case
    if not texts:
        return []

    if apply_preprocessing:
        processed_texts = [
            preprocess_text(text, structural_punctuation) for text in texts
        ]
    else:
        processed_texts = texts

    # Get probabilities from model
    probas = artifact.model.predict_proba(processed_texts)

    # Get class indices
    classes = artifact.model.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Map labels to indices
    label_mapping = artifact.label_mapping
    negative_label = label_mapping.get('negative', 0)
    positive_label = label_mapping.get('positive', 1)

    threshold = artifact.threshold
    target_class = artifact.target_class

    results = []
    for proba in probas:
        prob_negative = float(proba[class_to_idx[negative_label]])
        prob_positive = float(proba[class_to_idx[positive_label]])

        if target_class == 'negative':
            if prob_negative >= threshold:
                sentiment = 'negative'
                confidence = prob_negative
            else:
                sentiment = 'positive'
                confidence = prob_positive
        else:
            if prob_positive >= threshold:
                sentiment = 'positive'
                confidence = prob_positive
            else:
                sentiment = 'negative'
                confidence = prob_negative

        results.append(
            PredictionResult(
                sentiment=sentiment,
                confidence=confidence,
                probability_positive=prob_positive,
                probability_negative=prob_negative,
            )
        )

    return results
