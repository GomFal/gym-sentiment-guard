"""
Schema validation and data loading for ablation run.json files.

Implements Layer 1 data loading per REPORTING_STANDARDS.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.logging import get_logger, json_log

log = get_logger(__name__)

# Required columns per REPORTING_STANDARDS.md §Layer 1
REQUIRED_COLUMNS: list[str] = [
    'run_id',
    'constraint_met',
    'F1_neg',
    'F1_pos',
    'Recall_neg',
    'Macro_F1',
    'PR_AUC_neg',
    'threshold',
    'penalty',
    'C',
    'class_weight',
    'ngram_range',
    'min_df',
    'max_df',
    'sublinear_tf',
    'stopwords_enabled',
    'n_features',
    'coef_sparsity',
    'runtime_seconds',
    'Brier_Score',
    'ECE',
]


def _extract_run_data(run_path: Path) -> dict[str, Any] | None:
    """
    Extract flat row data from a single run.json file.

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
    logreg_params = data.get('logreg_params', {})
    diagnostics = data.get('diagnostics', {})
    classification_report = val_metrics.get('classification_report', {})

    # Extract positive class metrics from classification report
    pos_class_report = classification_report.get('1', {})
    f1_pos = pos_class_report.get('f1-score', 0.0)
    recall_pos = pos_class_report.get('recall', 0.0)
    precision_pos = pos_class_report.get('precision', 0.0)

    # Convert ngram_range list to tuple string for readability
    ngram_range = tfidf_params.get('ngram_range', [1, 1])
    ngram_str = f"({ngram_range[0]}, {ngram_range[1]})"

    # Convert stop_words to boolean: None/null means disabled, 'curated_safe' means enabled
    stop_words = tfidf_params.get('stop_words')
    stopwords_enabled = stop_words == 'curated_safe'

    return {
        'run_id': data.get('run_id', ''),
        'constraint_met': val_metrics.get('constraint_status') == 'met',
        'F1_neg': val_metrics.get('f1_neg', 0.0),
        'F1_pos': f1_pos,
        'Recall_neg': val_metrics.get('recall_neg', 0.0),
        'Recall_pos': recall_pos,
        'Precision_neg': val_metrics.get('precision_neg', 0.0),
        'Precision_pos': precision_pos,
        'Macro_F1': val_metrics.get('macro_f1', 0.0),
        'PR_AUC_neg': val_metrics.get('pr_auc_neg', 0.0),
        'threshold': val_metrics.get('threshold', 0.5),
        'penalty': logreg_params.get('penalty', 'l2'),
        'C': logreg_params.get('C', 1.0),
        'class_weight': logreg_params.get('class_weight'),
        'ngram_range': ngram_str,
        'min_df': tfidf_params.get('min_df', 1),
        'max_df': tfidf_params.get('max_df', 1.0),
        'sublinear_tf': tfidf_params.get('sublinear_tf', False),
        'stopwords_enabled': stopwords_enabled,
        'n_features': diagnostics.get('n_features', 0),
        'coef_sparsity': diagnostics.get('coefficient_sparsity', 0.0),
        'runtime_seconds': diagnostics.get('training_time_seconds', 0.0),
        'Brier_Score': val_metrics.get('brier_score', 0.0),
        'ECE': val_metrics.get('ece', 0.0),
        # Additional fields for Layer 4 deep dive
        'confusion_matrix': val_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
    }


def load_all_runs(experiments_dir: Path) -> pd.DataFrame:
    """
    Load all run.json files from experiments directory into a DataFrame.

    Uses set-based filtering for O(1) pattern matching per Python performance guidelines.

    Args:
        experiments_dir: Path to experiments directory containing run.* folders

    Returns:
        DataFrame with one row per run, columns per REQUIRED_COLUMNS
    """
    experiments_dir = Path(experiments_dir)

    # Find all run directories (pattern: run.YYYY-MM-DD_NNN)
    run_dirs = sorted(
        d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('run.')
    )

    log.info(
        json_log(
            'schema.loading_runs',
            component='reports',
            experiments_dir=str(experiments_dir),
            n_dirs=len(run_dirs),
        )
    )

    # Pre-allocate list for efficiency (per python-performance guidelines)
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_json = run_dir / 'run.json'
        if not run_json.exists():
            log.warning(json_log('schema.missing_run_json', dir=str(run_dir)))
            continue

        row = _extract_run_data(run_json)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    log.info(
        json_log(
            'schema.loaded_runs',
            component='reports',
            n_runs=len(df),
            n_valid=int(df['constraint_met'].sum()) if 'constraint_met' in df.columns else 0,
        )
    )

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    # Use set for O(1) membership testing (per python-performance guidelines)
    existing_cols = set(df.columns)
    required_set = set(REQUIRED_COLUMNS)
    missing = required_set - existing_cols

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    log.info(
        json_log(
            'schema.validated',
            component='reports',
            n_columns=len(df.columns),
            n_rows=len(df),
        )
    )


def load_test_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load test predictions CSV for Layer 4 PR curve.

    Args:
        predictions_path: Path to test_predictions.csv

    Returns:
        DataFrame with columns: y_true, y_pred, p_neg, p_pos
    """
    df = pd.read_csv(predictions_path)

    required = {'y_true', 'p_neg', 'p_pos'}
    existing = set(df.columns)
    missing = required - existing

    if missing:
        raise ValueError(f"Missing columns in test_predictions.csv: {sorted(missing)}")

    log.info(
        json_log(
            'schema.loaded_predictions',
            component='reports',
            n_samples=len(df),
            path=str(predictions_path),
        )
    )

    return df
