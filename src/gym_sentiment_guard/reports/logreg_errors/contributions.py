"""
Example-level contribution analysis for error analysis.

TASK 8: Explain individual misclassifications using feature contributions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ...utils.logging import get_logger, json_log
from .vectorizer_utils import get_feature_names, get_vectorizer_from_pipeline

log = get_logger(__name__)


def _get_mean_coefficients(classifier: CalibratedClassifierCV | Any) -> np.ndarray:
    """
    Extract mean coefficients from classifier.

    For CalibratedClassifierCV, averages across all base estimators.
    """
    if isinstance(classifier, CalibratedClassifierCV):
        all_coefs = []
        for calibrated_clf in classifier.calibrated_classifiers_:
            base_clf = calibrated_clf.estimator
            if hasattr(base_clf, 'coef_'):
                all_coefs.append(base_clf.coef_.ravel())
        if not all_coefs:
            raise ValueError('No coefficients found')
        return np.mean(np.vstack(all_coefs), axis=0)
    elif hasattr(classifier, 'coef_'):
        return classifier.coef_.ravel()
    else:
        raise ValueError('Classifier has no coef_ attribute')


def _get_mean_intercept(classifier: CalibratedClassifierCV | Any) -> float:
    """
    Extract mean intercept from classifier.

    For CalibratedClassifierCV, averages across all base estimators.
    """
    if isinstance(classifier, CalibratedClassifierCV):
        all_intercepts = []
        for calibrated_clf in classifier.calibrated_classifiers_:
            base_clf = calibrated_clf.estimator
            if hasattr(base_clf, 'intercept_'):
                all_intercepts.append(base_clf.intercept_.ravel()[0])
        if not all_intercepts:
            raise ValueError('No intercepts found')
        return float(np.mean(all_intercepts))
    elif hasattr(classifier, 'intercept_'):
        return float(classifier.intercept_.ravel()[0])
    else:
        raise ValueError('Classifier has no intercept_ attribute')


def compute_example_contributions(
    examples: pd.DataFrame,
    model: Pipeline,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Compute feature contributions for individual examples.

    Method: contribution[i] = tfidf[i] * coefficient[i]

    Args:
        examples: DataFrame with id, text columns
        model: Fitted Pipeline
        top_k: Number of top contributors per direction

    Returns:
        List of contribution dicts per example
    """
    vectorizer = get_vectorizer_from_pipeline(model)
    classifier = model.named_steps['logreg']

    # Get feature names (works for both TfidfVectorizer and FeatureUnion)
    feature_names = get_feature_names(vectorizer)

    coef = _get_mean_coefficients(classifier)
    intercept = _get_mean_intercept(classifier)

    # Transform texts
    texts = examples['text'].astype(str).tolist()
    tfidf_matrix = vectorizer.transform(texts)

    results = []

    for i, (_, row) in enumerate(examples.iterrows()):
        # Get sparse row
        if issparse(tfidf_matrix):
            tfidf_row = tfidf_matrix.getrow(i)
            # Convert to dense for element-wise operations
            indices = tfidf_row.indices
            tfidf_values = tfidf_row.data
        else:
            tfidf_values = tfidf_matrix[i]
            indices = np.nonzero(tfidf_values)[0]
            tfidf_values = tfidf_values[indices]

        # Compute contributions: tfidf * coef
        contributions = tfidf_values * coef[indices]

        # Total sum of all contributions + intercept = raw score
        total_contribution_sum = float(np.sum(contributions))
        raw_score = total_contribution_sum + intercept

        # Sort by absolute value to get top contributors
        sorted_idx = np.argsort(contributions)

        # Top positive contributors (highest values)
        top_pos_idx = sorted_idx[-top_k:][::-1]
        positive = [
            {
                'term': str(feature_names[indices[j]]),
                'tfidf': round(float(tfidf_values[j]), 4),
                'coef': round(float(coef[indices[j]]), 4),
                'contrib': round(float(contributions[j]), 4),
            }
            for j in top_pos_idx
            if contributions[j] > 0
        ]

        # Top negative contributors (lowest values)
        top_neg_idx = sorted_idx[:top_k]
        negative = [
            {
                'term': str(feature_names[indices[j]]),
                'tfidf': round(float(tfidf_values[j]), 4),
                'coef': round(float(coef[indices[j]]), 4),
                'contrib': round(float(contributions[j]), 4),
            }
            for j in top_neg_idx
            if contributions[j] < 0
        ]

        example_result = {
            'id': row['id'],
            'text': str(row['text'])[:500],  # Truncate for readability
            'y_true': int(row['y_true']),
            'y_pred': int(row['y_pred']),
            'model_intercept': round(intercept, 4),
            'total_contribution_sum': round(total_contribution_sum, 4),
            'raw_score': round(raw_score, 4),
            'top_positive_contributors': positive,
            'top_negative_contributors': negative,
        }
        results.append(example_result)

    log.info(
        json_log(
            'contributions.computed',
            component='error_analysis',
            n_examples=len(results),
        )
    )

    return results


def save_contributions(
    contributions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Save individual example contributions as separate JSON files.

    Args:
        contributions: List of contribution dicts
        output_dir: Directory for output

    Returns:
        Dict mapping example_id to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for contrib in contributions:
        example_id = contrib['id']
        output_path = output_dir / f'{example_id}.json'
        output_path.write_text(
            json.dumps(contrib, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        results[str(example_id)] = output_path

    log.info(
        json_log(
            'contributions.saved',
            component='error_analysis',
            n_files=len(results),
            output_dir=str(output_dir),
        )
    )

    return results
