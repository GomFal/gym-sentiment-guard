"""
Misclassification ranking for error analysis.

TASK 5: Generate ranked error CSVs for different failure types.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils.logging import get_logger, json_log

log = get_logger(__name__)


def generate_rankings(
    df: pd.DataFrame,
    output_dir: Path,
    confidence_threshold: float = 0.30,
    top_k: int = 50,
) -> dict[str, Path]:
    """
    Generate ranked error CSV files.

    Produces three rankings:
    1. high_confidence_wrong: confident but incorrect
    2. top_loss_wrong: highest loss errors
    3. near_threshold_wrong: uncertain and incorrect

    Args:
        df: Error table with is_error, abs_margin, loss, near_threshold columns
        output_dir: Directory for output CSVs
        confidence_threshold: Margin for "high confidence"
        top_k: Max errors per ranking

    Returns:
        Dict mapping ranking name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to errors only
    errors = df[df['is_error']].copy()
    n_errors = len(errors)

    # Columns to include in output
    output_cols = [
        'id',
        'text',
        'y_true',
        'y_pred',
        'p_neg',
        'p_pos',
        'abs_margin',
        'loss',
        'near_threshold',
        'low_coverage',
        'has_contrast',
    ]

    results: dict[str, Path] = {}

    # 1. High-confidence wrong: abs_margin >= confidence_threshold
    high_conf = errors[errors['abs_margin'] >= confidence_threshold].copy()
    high_conf = high_conf.sort_values('abs_margin', ascending=False).head(top_k)
    high_conf_path = output_dir / 'high_confidence_wrong.csv'
    high_conf[output_cols].to_csv(high_conf_path, index=False)
    results['high_confidence_wrong'] = high_conf_path

    # 2. Top-loss wrong: sorted by loss DESC
    top_loss = errors.sort_values('loss', ascending=False).head(top_k)
    top_loss_path = output_dir / 'top_loss_wrong.csv'
    top_loss[output_cols].to_csv(top_loss_path, index=False)
    results['top_loss_wrong'] = top_loss_path

    # 3. Near-threshold wrong: near_threshold == True
    near_thresh = errors[errors['near_threshold']].copy()
    near_thresh = near_thresh.sort_values('abs_margin', ascending=True).head(top_k)
    near_thresh_path = output_dir / 'near_threshold_wrong.csv'
    near_thresh[output_cols].to_csv(near_thresh_path, index=False)
    results['near_threshold_wrong'] = near_thresh_path

    log.info(
        json_log(
            'rankings.generated',
            component='error_analysis',
            n_errors=n_errors,
            n_high_confidence=len(high_conf),
            n_top_loss=len(top_loss),
            n_near_threshold=len(near_thresh),
        )
    )

    return results
