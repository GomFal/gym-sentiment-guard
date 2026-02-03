"""
Table generation for Layers 1 & 2 of SVM ablation reporting.

Implements protocol-based sorting and Top-K selection per REPORTING_STANDARDS.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def sort_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort ablation table according to experiment protocol.

    Sorting order:
    1. constraint_met (True first)
    2. F1_neg (descending)
    3. Macro_F1 (descending)
    4. PR_AUC_neg (descending)
    5. Brier_Score (ascending)
    6. ECE (ascending)
    """
    df_sorted = df.sort_values(
        by=['constraint_met', 'F1_neg', 'Macro_F1', 'PR_AUC_neg', 'Brier_Score', 'ECE'],
        ascending=[False, False, False, False, True, True],
    )

    log.info(
        json_log(
            'tables.sorted',
            component='reports',
            n_rows=len(df_sorted),
            top_run_id=df_sorted.iloc[0]['run_id'] if len(df_sorted) > 0 else None,
        )
    )

    return df_sorted.reset_index(drop=True)


def get_top_k(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Select Top-K runs with constraint_met == True.

    Must be called on a pre-sorted DataFrame.
    """
    valid_df = df[df['constraint_met'] == True].copy()  # noqa: E712

    if len(valid_df) == 0:
        log.warning(json_log('tables.no_valid_runs', component='reports'))
        return valid_df

    top_k = valid_df.head(k)
    return top_k


def get_winner(df: pd.DataFrame) -> pd.Series | None:
    """Get the winning run (first row after protocol sorting)."""
    valid_df = df[df['constraint_met'] == True]  # noqa: E712

    if len(valid_df) == 0:
        return None

    return valid_df.iloc[0]


def export_csv(df: pd.DataFrame, path: Path) -> Path:
    """Export DataFrame to CSV with deterministic formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, float_format='%.6f')
    log.info(json_log('tables.exported', path=str(path)))
    return path


def generate_comparison_table(df_top_k: pd.DataFrame, model_type: str) -> str:
    """
    Generate Markdown comparison table for Top-K results.
    """
    base_cols = [
        'run_id',
        'F1_neg',
        'Recall_neg',
        'Macro_F1',
        'PR_AUC_neg',
        'Brier_Score',
        'ECE',
        'threshold',
    ]

    if model_type == 'linear':
        extra_cols = ['C', 'coef_sparsity']
    else:  # rbf
        extra_cols = ['C', 'gamma', 'avg_support_vectors']

    display_cols = [c for c in (base_cols + extra_cols) if c in df_top_k.columns]
    display_df = df_top_k[display_cols].copy()

    # Format numeric columns
    format_cols = [
        'F1_neg',
        'Recall_neg',
        'Macro_F1',
        'PR_AUC_neg',
        'Brier_Score',
        'ECE',
        'threshold',
        'coef_sparsity',
    ]
    for col in format_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')

    # Add rank
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

    return display_df.to_markdown(index=False)
