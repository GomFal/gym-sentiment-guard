"""Core prediction logic for the serving module."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.cleaning import DEFAULT_STRUCTURAL_PUNCTUATION, EMOJI_PATTERN
from .loader import ModelArtifact, ModelExplainError


@dataclass(frozen=True)
class PredictionResult:
    """Result of a single prediction."""

    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float


@dataclass(frozen=True)
class ExplanationResult(PredictionResult):
    """Prediction result with feature importance explanation."""

    explanation: tuple[tuple[str, float], ...]  # (feature, importance) pairs


def preprocess_text(
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
        processed_texts = preprocess_text(texts, structural_punctuation)
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


def _get_classifier_coefficients(classifier) -> np.ndarray:
    """
    Extract coefficients from a classifier, handling CalibratedClassifierCV.

    Args:
        classifier: A sklearn classifier (LogisticRegression, CalibratedClassifierCV, etc.)

    Returns:
        1D numpy array of shape (n_features,)

    Raises:
        ModelExplainError: If coefficients cannot be extracted.
    """
    # Case 1: CalibratedClassifierCV (e.g., from cross-validation calibration)
    if hasattr(classifier, 'calibrated_classifiers_'):
        calibrated = classifier.calibrated_classifiers_
        if not calibrated:
            raise ModelExplainError('CalibratedClassifierCV has no calibrated classifiers')

        # Check that underlying estimators have coef_
        if not hasattr(calibrated[0].estimator, 'coef_'):
            raise ModelExplainError(
                f'Base estimator {type(calibrated[0].estimator).__name__} has no coefficients'
            )

        # Average coefficients across all folds
        coefs_sum = sum(cc.estimator.coef_ for cc in calibrated)
        coefs_avg = (coefs_sum / len(calibrated)).ravel()
        return coefs_avg

    # Case 2: Direct linear model (e.g., LogisticRegression)
    if hasattr(classifier, 'coef_'):
        return classifier.coef_[0]

    raise ModelExplainError(
        f'Classifier {type(classifier).__name__} does not support coefficient extraction'
    )


def explain_predictions(
    texts: list[str],
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
    structural_punctuation: str | None = None,
    top_k: int = 10,
) -> list[ExplanationResult]:
    """
    Make predictions with feature importance explanations.

    Args:
        texts: List of input review texts (1 to N texts).
        artifact: Loaded model artifact.
        apply_preprocessing: Whether to apply preprocessing.
        structural_punctuation: Optional custom punctuation pattern.
        top_k: Number of top features to return per prediction.

    Returns:
        List of ExplanationResult objects with predictions and explanations.

    Raises:
        ModelExplainError: If the model does not support explanation.
    """
    if not texts:
        return []

    # Validate model compatibility
    pipeline = artifact.model
    if len(pipeline.steps) < 2:
        raise ModelExplainError(
            'Pipeline must have at least 2 steps (vectorizer + classifier)'
        )

    vectorizer = pipeline.steps[0][1]
    classifier = pipeline.steps[-1][1]

    if not hasattr(vectorizer, 'get_feature_names_out'):
        raise ModelExplainError(
            f'Vectorizer {type(vectorizer).__name__} does not support feature names'
        )

    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = _get_classifier_coefficients(classifier)

    # Preprocess texts
    if apply_preprocessing:
        processed_texts = preprocess_text(texts, structural_punctuation)
    else:
        processed_texts = texts

    # Batch transform all texts at once
    sparse_matrix = vectorizer.transform(processed_texts)

    # Get predictions using existing function (reuse logic)
    predictions = predict(
        texts=texts,
        artifact=artifact,
        apply_preprocessing=apply_preprocessing,
        structural_punctuation=structural_punctuation,
    )

    # Build explanations for each text
    results: list[ExplanationResult] = []
    for i, pred in enumerate(predictions):
        row = sparse_matrix.getrow(i)
        indices = row.indices
        values = row.data

        # Calculate contributions: tfidf_value * coefficient
        contributions = [
            (feature_names[j], float(values[k] * coefficients[j]))
            for k, j in enumerate(indices)
        ]

        # Get top-k by absolute importance (O(n log k) instead of O(n log n))
        top_features = heapq.nlargest(
            min(top_k, len(contributions)),
            contributions,
            key=lambda x: abs(x[1]),
        )

        results.append(
            ExplanationResult(
                sentiment=pred.sentiment,
                confidence=pred.confidence,
                probability_positive=pred.probability_positive,
                probability_negative=pred.probability_negative,
                explanation=tuple(top_features),
            )
        )

    return results
