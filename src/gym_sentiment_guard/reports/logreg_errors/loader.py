"""
Model and data loading utilities for error analysis.

TASK 2: Load trained artifacts and merge text with predictions.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ...utils.logging import get_logger, json_log

log = get_logger(__name__)


def load_model_bundle(model_path: Path) -> Pipeline:
    """
    Load trained model pipeline from joblib.

    Args:
        model_path: Path to logreg.joblib

    Returns:
        Fitted sklearn Pipeline (TfidfVectorizer + CalibratedClassifierCV)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    model = joblib.load(model_path)
    log.info(json_log('loader.model_loaded', component='error_analysis', path=str(model_path)))

    return model


def load_merged_data(
    test_csv_path: Path,
    predictions_csv_path: Path,
    text_column: str = 'comment',
    id_column: str = 'id',
) -> pd.DataFrame:
    """
    Load and merge test data with predictions.

    Joins by row index since predictions_csv has no ID column.

    Args:
        test_csv_path: Path to frozen test.csv with text and labels
        predictions_csv_path: Path to test_predictions.csv with probabilities
        text_column: Name of text column in test.csv
        id_column: Name of ID column in test.csv

    Returns:
        Merged DataFrame with columns:
        - id, text, y_true, y_pred, p_neg, p_pos
    """
    test_csv_path = Path(test_csv_path)
    predictions_csv_path = Path(predictions_csv_path)

    if not test_csv_path.exists():
        raise FileNotFoundError(f'Test CSV not found: {test_csv_path}')
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f'Predictions CSV not found: {predictions_csv_path}')

    # Load datasets
    test_df = pd.read_csv(test_csv_path)
    pred_df = pd.read_csv(predictions_csv_path)

    # Validate row counts match
    if len(test_df) != len(pred_df):
        raise ValueError(
            f'Row count mismatch: test_csv={len(test_df)}, predictions_csv={len(pred_df)}'
        )

    # Merge by index (row order assumed to match)
    merged = pd.DataFrame(
        {
            'id': test_df[id_column].values,
            'text': test_df[text_column].astype(str).values,
            'y_true': pred_df['y_true'].values,
            'y_pred': pred_df['y_pred'].values,
            'p_neg': pred_df['p_neg'].values,
            'p_pos': pred_df['p_pos'].values,
        }
    )

    log.info(
        json_log(
            'loader.data_merged',
            component='error_analysis',
            n_samples=len(merged),
            test_path=str(test_csv_path),
            pred_path=str(predictions_csv_path),
        )
    )

    return merged
