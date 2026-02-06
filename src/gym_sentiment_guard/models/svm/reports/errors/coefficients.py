"""
Model coefficient inspection for SVM error analysis.

Extracts and documents global model coefficients for Linear SVM.
Note: RBF SVM does not have coef_ attribute - use support_vectors.py instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def extract_coefficients(
    model: Pipeline,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Extract top model coefficients for Linear SVM.

    For CalibratedClassifierCV, computes the mean coefficients across
    all calibrated estimators (not just the first one).

    Args:
        model: Fitted Pipeline with TfidfVectorizer and Linear SVM
        top_k: Number of top coefficients per direction

    Returns:
        Dict with positive/negative coefficients grouped by n-gram length

    Raises:
        ValueError: If model is RBF SVM (no coef_ attribute)
    """
    # Get vectorizer (check common step names)
    vectorizer = None
    for name in ('tfidf', 'vectorizer', 'features'):
        if name in model.named_steps:
            vectorizer = model.named_steps[name]
            break
    if vectorizer is None:
        raise ValueError(f"No vectorizer found. Available steps: {list(model.named_steps.keys())}")

    # Get classifier (check common step names)
    classifier = model.named_steps.get('svm') or model.named_steps.get('classifier')
    if classifier is None:
        raise ValueError(f"No classifier found. Available steps: {list(model.named_steps.keys())}")

    # Get feature names directly
    feature_names = vectorizer.get_feature_names_out()

    # Extract coefficients
    # For CalibratedClassifierCV, average across all base estimators
    if isinstance(classifier, CalibratedClassifierCV):
        # Pre-allocate matrix for coefficients (avoid list resizing + vstack)
        calibrated_clfs = classifier.calibrated_classifiers_
        n_estimators = len(calibrated_clfs)
        n_features = calibrated_clfs[0].estimator.coef_.shape[1]
        coef_matrix = np.zeros((n_estimators, n_features), dtype=np.float64)

        for i, calibrated_clf in enumerate(calibrated_clfs):
            base_clf = calibrated_clf.estimator
            if hasattr(base_clf, 'coef_'):
                coef_matrix[i] = base_clf.coef_.ravel()

        if coef_matrix.sum() == 0:
            raise ValueError(
                'No coefficients found in calibrated estimators. '
                'This may be an RBF SVM - use support_vectors.py instead.'
            )

        # Compute mean coefficients across all estimators
        coef = coef_matrix.mean(axis=0)

        log.info(
            json_log(
                'coefficients.averaged',
                component='error_analysis',
                model_type='svm_linear',
                n_estimators=n_estimators,
            )
        )
    elif hasattr(classifier, 'coef_'):
        coef = classifier.coef_.ravel()
    else:
        raise ValueError(
            'Classifier has no coef_ attribute. '
            'This may be an RBF SVM - use support_vectors.py instead.'
        )

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
        'model_type': 'svm_linear',
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
            model_type='svm_linear',
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
