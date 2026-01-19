"""
Slice metrics computation for error analysis.

TASK 6: Quantify error rates across data slices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from ...utils.logging import get_logger, json_log

log = get_logger(__name__)


def compute_slice_metrics(
    df: pd.DataFrame,
    min_slice_size: int = 50,
) -> dict[str, Any]:
    """
    Compute metrics per data slice.

    Slices:
    - overall
    - near_threshold == True
    - low_coverage == True
    - has_contrast == True

    Args:
        df: Error table with all risk tags
        min_slice_size: Ignore slices with fewer samples

    Returns:
        Dict with slice metrics
    """
    slices = {
        'overall': df,
        'near_threshold': df[df['near_threshold']],
        'low_coverage': df[df['low_coverage']],
        'has_contrast': df[df['has_contrast']],
    }

    results: dict[str, Any] = {}

    for slice_name, slice_df in slices.items():
        n = len(slice_df)

        # Skip small slices
        if n < min_slice_size:
            results[slice_name] = {
                'n': n,
                'skipped': True,
                'reason': f'n < {min_slice_size}',
            }
            continue

        y_true = slice_df['y_true'].values
        y_pred = slice_df['y_pred'].values
        abs_margin = slice_df['abs_margin'].values

        n_errors = (y_true != y_pred).sum()
        error_rate = n_errors / n if n > 0 else 0.0

        # Class-specific metrics
        # pos_label=0 for negative class, pos_label=1 for positive class
        metrics = {
            'n': n,
            'skipped': False,
            'n_errors': int(n_errors),
            'error_rate': round(error_rate, 4),
            'mean_confidence': round(float(np.mean(abs_margin)), 4),
            # Negative class metrics (pos_label=0)
            'precision_neg': round(
                precision_score(y_true, y_pred, pos_label=0, zero_division=0.0), 4
            ),
            'recall_neg': round(recall_score(y_true, y_pred, pos_label=0, zero_division=0.0), 4),
            'f1_neg': round(f1_score(y_true, y_pred, pos_label=0, zero_division=0.0), 4),
            # Positive class metrics (pos_label=1)
            'precision_pos': round(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0.0), 4
            ),
            'recall_pos': round(recall_score(y_true, y_pred, pos_label=1, zero_division=0.0), 4),
            'f1_pos': round(f1_score(y_true, y_pred, pos_label=1, zero_division=0.0), 4),
        }

        results[slice_name] = metrics

    log.info(
        json_log(
            'slices.computed',
            component='error_analysis',
            n_slices=len(results),
            overall_error_rate=results.get('overall', {}).get('error_rate', 0),
        )
    )

    return results


def save_slice_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    """
    Save slice metrics as JSON.

    Args:
        metrics: Slice metrics dict
        output_path: Path to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

    log.info(json_log('slices.saved', component='error_analysis', path=str(output_path)))

    return output_path
