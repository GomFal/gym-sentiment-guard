"""
Example-level contribution analysis for SVM error analysis.

Explains individual misclassifications using feature contributions.
Note: Only works for Linear SVM (requires coef_ attribute).
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

from gym_sentiment_guard.utils.logging import get_logger, json_log

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
            raise ValueError(
                'No coefficients found. '
                'This may be an RBF SVM - contributions not supported.'
            )
        return np.mean(np.vstack(all_coefs), axis=0)
    elif hasattr(classifier, 'coef_'):
        return classifier.coef_.ravel()
    else:
        raise ValueError(
            'Classifier has no coef_ attribute. '
            'This may be an RBF SVM - contributions not supported.'
        )


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
        model: Fitted Pipeline with Linear SVM
        top_k: Number of top contributors per direction

    Returns:
        List of contribution dicts per example

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

    coef = _get_mean_coefficients(classifier)
    intercept = _get_mean_intercept(classifier)

    # Transform texts
    texts = examples['text'].astype(str).tolist()
    tfidf_matrix = vectorizer.transform(texts)

    # Pre-extract columns for fast access (avoid iterrows overhead)
    n_examples = len(examples)
    ids = examples['id'].values
    texts_arr = examples['text'].values
    y_trues = examples['y_true'].values
    y_preds = examples['y_pred'].values

    # Pre-allocate results list
    results = [None] * n_examples

    # Cache feature names as list for faster indexing
    feature_names_list = feature_names.tolist()

    # Get CSR internal arrays for direct access (avoid getrow overhead)
    if issparse(tfidf_matrix):
        indptr = tfidf_matrix.indptr
        all_indices = tfidf_matrix.indices
        all_data = tfidf_matrix.data

    for i in range(n_examples):
        # Direct CSR access without creating temporary objects
        if issparse(tfidf_matrix):
            start, end = indptr[i], indptr[i + 1]
            indices = all_indices[start:end]
            tfidf_values = all_data[start:end]
        else:
            tfidf_values = tfidf_matrix[i]
            indices = np.nonzero(tfidf_values)[0]
            tfidf_values = tfidf_values[indices]

        # Compute contributions: tfidf * coef
        contributions = tfidf_values * coef[indices]

        # Total sum of all contributions + intercept = raw score
        total_contribution_sum = float(np.sum(contributions))
        raw_score = total_contribution_sum + intercept

        # Sort by value to get top contributors
        sorted_idx = np.argsort(contributions)

        # Top positive contributors (highest values)
        top_pos_idx = sorted_idx[-top_k:][::-1]
        positive = [
            {
                'term': feature_names_list[indices[j]],
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
                'term': feature_names_list[indices[j]],
                'tfidf': round(float(tfidf_values[j]), 4),
                'coef': round(float(coef[indices[j]]), 4),
                'contrib': round(float(contributions[j]), 4),
            }
            for j in top_neg_idx
            if contributions[j] < 0
        ]

        results[i] = {
            'id': int(ids[i]),
            'text': str(texts_arr[i])[:500],  # Truncate for readability
            'y_true': int(y_trues[i]),
            'y_pred': int(y_preds[i]),
            'model_type': 'svm_linear',
            'model_intercept': round(intercept, 4),
            'total_contribution_sum': round(total_contribution_sum, 4),
            'raw_score': round(raw_score, 4),
            'top_positive_contributors': positive,
            'top_negative_contributors': negative,
        }

    log.info(
        json_log(
            'contributions.computed',
            component='error_analysis',
            model_type='svm_linear',
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
