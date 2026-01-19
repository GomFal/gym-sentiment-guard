"""
Error table construction for error analysis.

TASK 3: Create the main error analysis table with predictions, margins, and coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline

from ...utils.logging import get_logger, json_log
from .vectorizer_utils import get_vectorizer_from_pipeline

log = get_logger(__name__)


def build_error_table(
    df: pd.DataFrame,
    model: Pipeline,
    threshold: float,
    low_coverage_nnz_threshold: int = 5,
    low_coverage_tfidf_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build the main error analysis table.

    Adds computed columns for loss, margin, coverage, and error flags.

    Args:
        df: Merged DataFrame from loader (id, text, y_true, y_pred, p_neg, p_pos)
        model: Fitted Pipeline with TfidfVectorizer or FeatureUnion
        threshold: Decision threshold
        low_coverage_nnz_threshold: Min non-zero features for "covered"
        low_coverage_tfidf_threshold: Min TF-IDF sum for "covered"

    Returns:
        DataFrame with additional columns:
        - abs_margin, loss, is_error, nnz, tfidf_sum, low_coverage
    """
    # Get TF-IDF matrix for coverage computation
    vectorizer = get_vectorizer_from_pipeline(model)
    tfidf_matrix = vectorizer.transform(df['text'].astype(str))

    # Compute coverage metrics (sparse-safe)
    # Pre-allocate arrays for performance (per python-performance guidelines)
    n_samples = len(df)
    nnz_values = np.zeros(n_samples, dtype=np.int32)
    tfidf_sum_values = np.zeros(n_samples, dtype=np.float64)

    if issparse(tfidf_matrix):
        # Use sparse matrix operations for efficiency
        for i in range(n_samples):
            row = tfidf_matrix.getrow(i)
            nnz_values[i] = row.nnz
            tfidf_sum_values[i] = row.sum()
    else:
        nnz_values = np.count_nonzero(tfidf_matrix, axis=1)
        tfidf_sum_values = tfidf_matrix.sum(axis=1)

    # Compute derived columns
    p_neg = df['p_neg'].values
    p_pos = df['p_pos'].values
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values

    # Absolute margin from threshold
    abs_margin = np.abs(p_neg - threshold)

    # Cross-entropy loss: -log(p_correct)
    # p_correct = p_neg if y_true==0 else p_pos
    p_correct = np.where(y_true == 0, p_neg, p_pos)
    # Clip to avoid log(0)
    p_correct_clipped = np.clip(p_correct, 1e-15, 1.0)
    loss = -np.log(p_correct_clipped)

    # Error flag
    is_error = y_true != y_pred

    # Low coverage flag
    low_coverage = (nnz_values < low_coverage_nnz_threshold) | (
        tfidf_sum_values < low_coverage_tfidf_threshold
    )

    # Build output DataFrame (preserve original columns)
    result = df.copy()
    result['abs_margin'] = abs_margin
    result['loss'] = loss
    result['is_error'] = is_error
    result['nnz'] = nnz_values
    result['tfidf_sum'] = tfidf_sum_values
    result['low_coverage'] = low_coverage

    n_errors = is_error.sum()
    error_rate = n_errors / n_samples if n_samples > 0 else 0.0

    log.info(
        json_log(
            'error_table.built',
            component='error_analysis',
            n_samples=n_samples,
            n_errors=int(n_errors),
            error_rate=round(error_rate, 4),
        )
    )

    return result


def save_error_table(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Save error table as Parquet.

    Args:
        df: Error table DataFrame
        output_path: Path to save (should end in .parquet)

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    log.info(json_log('error_table.saved', component='error_analysis', path=str(output_path)))

    return output_path
