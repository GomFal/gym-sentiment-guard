"""
Schema validation and data loading for SVM ablation run.json files.

Implements Layer 1 data loading per REPORTING_STANDARDS.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)

# Common columns for both Linear and RBF
COMMON_COLUMNS: list[str] = [
    'run_id',
    'constraint_met',
    'F1_neg',
    'F1_pos',
    'Recall_neg',
    'Precision_neg',
    'Macro_F1',
    'PR_AUC_neg',
    'threshold',
    # FeatureUnion params
    'unigram_ngram_range',
    'unigram_min_df',
    'unigram_stop_words',
    'multigram_ngram_range',
    'multigram_min_df',
    # Diagnostics
    'n_features',
    'runtime_seconds',
    'Brier_Score',
    'ECE',
]

# Linear SVM specific columns
LINEAR_COLUMNS: list[str] = [
    'C',
    'tol',
    'max_iter',
    'coef_sparsity',
]

# RBF SVM specific columns
RBF_COLUMNS: list[str] = [
    'kernel',
    'C',
    'gamma',
    'tol',
    'cache_size',
    'max_iter',
    'avg_support_vectors',
    'use_scaler',
]


def _extract_linear_run_data(run_path: Path) -> dict[str, Any] | None:
    """
    Extract flat row data from a single Linear SVM run.json file.

    Args:
        run_path: Path to run.json file

    Returns:
        Dictionary with flattened column values, or None if invalid
    """
    try:
        data = json.loads(run_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(json_log('schema.load_error', path=str(run_path), error=str(e)))
        return None

    val_metrics = data.get('val_metrics')
    if val_metrics is None:
        log.warning(json_log('schema.missing_val_metrics', run_id=data.get('run_id')))
        return None

    tfidf_params = data.get('tfidf_params', {})
    classifier_params = data.get('classifier_params', {})
    diagnostics = data.get('diagnostics', {})
    classification_report = val_metrics.get('classification_report', {})

    # Extract positive class metrics
    pos_class_report = classification_report.get('1', {})
    f1_pos = pos_class_report.get('f1-score', 0.0)

    # Format parameter strings
    unigram_ngram = tuple(tfidf_params.get('unigram_ngram_range', [1, 1]))
    multigram_ngram = tuple(tfidf_params.get('multigram_ngram_range', [2, 3]))

    return {
        'run_id': data.get('run_id', ''),
        'constraint_met': val_metrics.get('constraint_status') == 'met',
        'F1_neg': val_metrics.get('f1_neg', 0.0),
        'F1_pos': f1_pos,
        'Recall_neg': val_metrics.get('recall_neg', 0.0),
        'Precision_neg': val_metrics.get('precision_neg', 0.0),
        'Macro_F1': val_metrics.get('macro_f1', 0.0),
        'PR_AUC_neg': val_metrics.get('pr_auc_neg', 0.0),
        'threshold': val_metrics.get('threshold', 0.5),
        # Linear Hyperparams
        'C': classifier_params.get('C', 1.0),
        'tol': classifier_params.get('tol', 1e-4),
        'max_iter': classifier_params.get('max_iter', 1000),
        'coef_sparsity': data.get('coefficient_sparsity', 0.0),
        # FeatureUnion
        'unigram_ngram_range': str(unigram_ngram),
        'unigram_min_df': tfidf_params.get('unigram_min_df', 1),
        'unigram_stop_words': str(tfidf_params.get('unigram_stop_words')),
        'multigram_ngram_range': str(multigram_ngram),
        'multigram_min_df': tfidf_params.get('multigram_min_df', 1),
        # Diagnostics
        'n_features': diagnostics.get('n_features', 0),
        'runtime_seconds': diagnostics.get('training_time_seconds', 0.0),
        'Brier_Score': val_metrics.get('brier_score', 0.0),
        'ECE': val_metrics.get('ece', 0.0),
        # Additional fields
        'confusion_matrix': val_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
    }


def _extract_rbf_run_data(run_path: Path) -> dict[str, Any] | None:
    """
    Extract flat row data from a single RBF SVM run.json file.

    Args:
        run_path: Path to run.json file

    Returns:
        Dictionary with flattened column values, or None if invalid
    """
    try:
        data = json.loads(run_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(json_log('schema.load_error', path=str(run_path), error=str(e)))
        return None

    val_metrics = data.get('val_metrics')
    if val_metrics is None:
        log.warning(json_log('schema.missing_val_metrics', run_id=data.get('run_id')))
        return None

    tfidf_params = data.get('tfidf_params', {})
    classifier_params = data.get('classifier_params', {})
    diagnostics = data.get('diagnostics', {})
    classification_report = val_metrics.get('classification_report', {})

    # Extract positive class metrics
    pos_class_report = classification_report.get('1', {})
    f1_pos = pos_class_report.get('f1-score', 0.0)

    # Format parameter strings
    unigram_ngram = tuple(tfidf_params.get('unigram_ngram_range', [1, 1]))
    multigram_ngram = tuple(tfidf_params.get('multigram_ngram_range', [2, 3]))

    return {
        'run_id': data.get('run_id', ''),
        'constraint_met': val_metrics.get('constraint_status') == 'met',
        'F1_neg': val_metrics.get('f1_neg', 0.0),
        'F1_pos': f1_pos,
        'Recall_neg': val_metrics.get('recall_neg', 0.0),
        'Precision_neg': val_metrics.get('precision_neg', 0.0),
        'Macro_F1': val_metrics.get('macro_f1', 0.0),
        'PR_AUC_neg': val_metrics.get('pr_auc_neg', 0.0),
        'threshold': val_metrics.get('threshold', 0.5),
        # RBF Hyperparams
        'kernel': classifier_params.get('kernel', 'rbf'),
        'C': classifier_params.get('C', 1.0),
        'gamma': classifier_params.get('gamma', 'scale'),
        'tol': classifier_params.get('tol', 1e-3),
        'cache_size': classifier_params.get('cache_size', 1000),
        'max_iter': classifier_params.get('max_iter', -1),
        'use_scaler': classifier_params.get('use_scaler', True),
        'avg_support_vectors': data.get('diagnostics', {}).get('coefficient_sparsity', 0),
        # FeatureUnion
        'unigram_ngram_range': str(unigram_ngram),
        'unigram_min_df': tfidf_params.get('unigram_min_df', 1),
        'unigram_stop_words': str(tfidf_params.get('unigram_stop_words')),
        'multigram_ngram_range': str(multigram_ngram),
        'multigram_min_df': tfidf_params.get('multigram_min_df', 1),
        # Diagnostics
        'n_features': diagnostics.get('n_features', 0),
        'runtime_seconds': diagnostics.get('training_time_seconds', 0.0),
        'Brier_Score': val_metrics.get('brier_score', 0.0),
        'ECE': val_metrics.get('ece', 0.0),
        # Additional fields
        'confusion_matrix': val_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
    }


def load_all_runs(experiments_dir: Path, model_type: str = 'linear') -> pd.DataFrame:
    """
    Load all run.json files from experiments directory into a DataFrame.

    Args:
        experiments_dir: Path to experiments directory containing run.* folders
        model_type: 'linear' or 'rbf'

    Returns:
        DataFrame with one row per run
    """
    experiments_dir = Path(experiments_dir)
    run_dirs = sorted(
        d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('run.')
    )

    log.info(
        json_log(
            'schema.loading_runs',
            component='reports',
            model_type=model_type,
            n_dirs=len(run_dirs),
        )
    )

    rows: list[dict[str, Any]] = []
    extract_fn = _extract_linear_run_data if model_type == 'linear' else _extract_rbf_run_data

    for run_dir in run_dirs:
        run_json = run_dir / 'run.json'
        if not run_json.exists():
            continue

        row = extract_fn(run_json)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def validate_schema(df: pd.DataFrame, model_type: str = 'linear') -> None:
    """
    Validate that DataFrame contains all required columns for the model type.

    Args:
        df: DataFrame to validate
        model_type: 'linear' or 'rbf'

    Raises:
        ValueError: If required columns are missing
    """
    required = set(COMMON_COLUMNS)
    if model_type == 'linear':
        required.update(LINEAR_COLUMNS)
    else:
        required.update(RBF_COLUMNS)

    existing = set(df.columns)
    missing = required - existing

    if missing:
        raise ValueError(f'Missing required columns for {model_type}: {sorted(missing)}')


def load_test_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load test predictions CSV.

    Args:
        predictions_path: Path to test_predictions.csv

    Returns:
        DataFrame with y_true, p_neg, etc.
    """
    df = pd.read_csv(predictions_path)
    required = {'y_true', 'p_neg', 'p_pos'}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f'Missing columns in test predictions: {sorted(missing)}')

    return df
