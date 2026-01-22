"""
Model coefficient inspection for error analysis.

TASK 7: Extract and document global model coefficients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.utils.logging import get_logger, json_log

from .vectorizer_utils import get_feature_names, get_vectorizer_from_pipeline

log = get_logger(__name__)


def extract_coefficients(
    model: Pipeline,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Extract top model coefficients.

    For CalibratedClassifierCV, computes the mean coefficients across
    all calibrated estimators (not just the first one).

    Args:
        model: Fitted Pipeline with TfidfVectorizer
        top_k: Number of top coefficients per direction

    Returns:
        Dict with positive/negative coefficients grouped by n-gram length
    """
    vectorizer = get_vectorizer_from_pipeline(model)
    classifier = model.named_steps['logreg']

    # Get feature names (works for both TfidfVectorizer and FeatureUnion)
    feature_names = get_feature_names(vectorizer)

    # Extract coefficients
    # For CalibratedClassifierCV, average across all base estimators
    if isinstance(classifier, CalibratedClassifierCV):
        # Get coefficients from all calibrated classifiers
        all_coefs = []
        for calibrated_clf in classifier.calibrated_classifiers_:
            base_clf = calibrated_clf.estimator
            if hasattr(base_clf, 'coef_'):
                all_coefs.append(base_clf.coef_.ravel())

        if not all_coefs:
            raise ValueError('No coefficients found in calibrated estimators')

        # Compute mean coefficients across all estimators
        # Stack and compute mean along axis 0
        coef_matrix = np.vstack(all_coefs)
        coef = np.mean(coef_matrix, axis=0)

        log.info(
            json_log(
                'coefficients.averaged',
                component='error_analysis',
                n_estimators=len(all_coefs),
            )
        )
    elif hasattr(classifier, 'coef_'):
        coef = classifier.coef_.ravel()
    else:
        raise ValueError('Classifier has no coef_ attribute')

    # Create DataFrame for sorting
    n_features = len(coef)

    # Determine n-gram length for each term
    def ngram_length(term: str) -> int:
        return len(term.split())

    # Get top positive and negative coefficients
    sorted_indices = np.argsort(coef)

    # Top positive (highest values)
    positive_indices = sorted_indices[-top_k:][::-1]
    positive = [
        {
            'term': str(feature_names[i]),
            'coef': round(float(coef[i]), 4),
            'ngram_len': ngram_length(str(feature_names[i])),
        }
        for i in positive_indices
    ]

    # Top negative (lowest values)
    negative_indices = sorted_indices[:top_k]
    negative = [
        {
            'term': str(feature_names[i]),
            'coef': round(float(coef[i]), 4),
            'ngram_len': ngram_length(str(feature_names[i])),
        }
        for i in negative_indices
    ]

    result = {
        'positive': positive,
        'negative': negative,
        'metadata': {
            'n_features': n_features,
            'top_k': top_k,
        },
    }

    log.info(
        json_log(
            'coefficients.extracted',
            component='error_analysis',
            n_features=n_features,
            top_positive_coef=positive[0]['coef'] if positive else 0,
            top_negative_coef=negative[0]['coef'] if negative else 0,
        )
    )

    return result


def save_coefficients(coefficients: dict[str, Any], output_path: Path) -> Path:
    """
    Save model coefficients as JSON.

    Args:
        coefficients: Coefficient dict from extract_coefficients
        output_path: Path to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(coefficients, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    log.info(json_log('coefficients.saved', component='error_analysis', path=str(output_path)))

    return output_path
