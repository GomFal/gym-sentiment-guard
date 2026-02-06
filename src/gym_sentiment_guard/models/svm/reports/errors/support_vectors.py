"""
Support vector analysis for RBF SVM error analysis.

Provides model interpretability via support vector statistics for RBF kernels,
which lack the coef_ attribute available in linear models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def _get_base_svc(classifier: CalibratedClassifierCV | Any) -> Any:
    """
    Extract base SVC from classifier (handles CalibratedClassifierCV wrapper).

    Returns the first base estimator for support vector access.
    """
    if isinstance(classifier, CalibratedClassifierCV):
        if classifier.calibrated_classifiers_:
            return classifier.calibrated_classifiers_[0].estimator
        raise ValueError('No calibrated classifiers found')
    return classifier


def extract_support_vector_stats(
    model: Pipeline,
    training_data: pd.DataFrame | None = None,
    n_example_svs: int = 5,
) -> dict[str, Any]:
    """
    Extract support vector statistics for RBF SVM.

    Provides interpretability metrics for RBF models:
    - Total support vector count
    - Per-class support vector counts
    - SV ratio (fraction of training data that are SVs)
    - Example support vectors (if training data provided)

    Args:
        model: Fitted Pipeline with RBF SVM
        training_data: Optional training DataFrame for example SV lookup
        n_example_svs: Number of example SVs per class to include

    Returns:
        Dict with support vector statistics

    Raises:
        ValueError: If model is Linear SVM (use coefficients.py instead)
    """
    # Get classifier from pipeline (handles different step names)
    classifier = model.named_steps.get('svm') or model.named_steps.get('classifier')
    if classifier is None:
        raise ValueError(
            "Model pipeline does not have 'svm' or 'classifier' step. "
            f"Available steps: {list(model.named_steps.keys())}"
        )
    base_svc = _get_base_svc(classifier)

    # Verify this is an RBF SVM (has support_vectors_ but no coef_)
    if not hasattr(base_svc, 'support_vectors_'):
        raise ValueError(
            'Classifier has no support_vectors_ attribute. '
            'This may be a Linear SVM - use coefficients.py instead.'
        )

    if hasattr(base_svc, 'coef_'):
        log.warning(
            json_log(
                'support_vectors.linear_detected',
                component='error_analysis',
                message='Linear SVM detected. coefficients.py is more appropriate.',
            )
        )

    # Extract support vector statistics
    support_vectors = base_svc.support_vectors_
    n_support = base_svc.n_support_  # Array: [n_sv_class_0, n_sv_class_1]
    support_indices = base_svc.support_  # Indices into training data

    total_sv = int(support_vectors.shape[0])
    n_sv_neg = int(n_support[0])  # Class 0 (negative)
    n_sv_pos = int(n_support[1])  # Class 1 (positive)

    # Compute SV ratio if we have calibrated classifiers with training info
    sv_ratio = None
    if isinstance(classifier, CalibratedClassifierCV):
        # Estimate training size from all calibrated estimators
        # Each uses a different fold, so we sum unique indices
        all_indices = set()
        for cal_clf in classifier.calibrated_classifiers_:
            base = cal_clf.estimator
            if hasattr(base, 'support_'):
                all_indices.update(base.support_.tolist())
        if total_sv > 0:
            # Rough estimate: assume training size ≈ total_sv / typical_sv_ratio
            # This is an approximation since we don't have exact training size
            sv_ratio = None  # Cannot compute without training size

    result = {
        'model_type': 'svm_rbf',
        'total_support_vectors': total_sv,
        'n_support_per_class': {
            'negative': n_sv_neg,
            'positive': n_sv_pos,
        },
        'sv_ratio': sv_ratio,
        'support_vector_indices': support_indices.tolist()[:100],  # First 100 for reference
        'metadata': {
            'kernel': 'rbf',
            'n_example_svs': n_example_svs,
        },
    }

    # Add example support vectors if training data provided
    if training_data is not None and len(training_data) > 0:
        example_svs = []

        # Get some example SV indices per class
        neg_sv_indices = support_indices[:n_sv_neg][:n_example_svs]
        pos_sv_indices = support_indices[n_sv_neg:][:n_example_svs]

        for idx in neg_sv_indices:
            if idx < len(training_data):
                row = training_data.iloc[idx]
                example_svs.append({
                    'index': int(idx),
                    'text': str(row.get('text', row.get('comment', '')))[:200],
                    'class': 0,
                    'class_name': 'negative',
                })

        for idx in pos_sv_indices:
            if idx < len(training_data):
                row = training_data.iloc[idx]
                example_svs.append({
                    'index': int(idx),
                    'text': str(row.get('text', row.get('comment', '')))[:200],
                    'class': 1,
                    'class_name': 'positive',
                })

        result['example_svs'] = example_svs

    log.info(
        json_log(
            'support_vectors.extracted',
            component='error_analysis',
            model_type='svm_rbf',
            total_svs=total_sv,
            n_sv_neg=n_sv_neg,
            n_sv_pos=n_sv_pos,
        )
    )

    return result


def save_support_vector_stats(stats: dict[str, Any], output_path: Path) -> Path:
    """
    Save support vector statistics as JSON.

    Args:
        stats: Support vector stats dict from extract_support_vector_stats
        output_path: Path to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    log.info(
        json_log('support_vectors.saved', component='error_analysis', path=str(output_path))
    )

    return output_path
