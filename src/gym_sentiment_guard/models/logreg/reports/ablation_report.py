"""
Main orchestrator for ablation suite report generation.

Implements the complete 4-layer reporting workflow per REPORTING_STANDARDS.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from gym_sentiment_guard.utils.logging import get_logger, json_log

from .markdown import (
    generate_ablation_analysis,
    generate_final_model_report,
    generate_top5_report,
    write_report,
)
from .plots import (
    plot_c_vs_f1neg,
    plot_calibration_curve,
    plot_confusion_matrix_heatmap,
    plot_ngram_effect,
    plot_pr_curve_neg,
    plot_stopwords_effect,
    plot_threshold_curve,
    plot_top_k_f1neg_bar,
    plot_val_vs_test_comparison,
)
from .schema import load_all_runs, load_test_predictions, validate_schema
from .tables import (
    export_csv,
    get_top_k,
    get_winner,
    sort_ablation_table,
)

log = get_logger(__name__)


def _generate_factor_conclusions(df: pd.DataFrame) -> dict[str, str]:
    """
    Generate automated conclusions for factor-level analysis.

    Args:
        df: Full ablation table

    Returns:
        Dict mapping factor name to conclusion text
    """
    conclusions = {}

    # C analysis
    c_grouped = df.groupby('C')['F1_neg'].agg(['mean', 'std'])
    best_c = c_grouped['mean'].idxmax()
    conclusions['C'] = (
        f'Best mean F1_neg achieved at C={best_c}. '
        f'Performance varies by {c_grouped["std"].mean():.4f} std across configurations. '
        f'Model shows {"sensitivity" if c_grouped["std"].max() > 0.02 else "stability"} to regularization.'
    )

    # ngram analysis
    ngram_grouped = df.groupby('ngram_range')['F1_neg'].mean()
    best_ngram = ngram_grouped.idxmax()
    conclusions['ngram_range'] = (
        f'Best performance with ngram_range={best_ngram}. '
        f'Higher-order n-grams provide '
        f'{"marginal improvements" if ngram_grouped.max() - ngram_grouped.min() < 0.01 else "noticeable gains"}.'
    )

    # Stopwords analysis
    stop_grouped = df.groupby('stopwords_enabled')['F1_neg'].mean()
    if len(stop_grouped) >= 2:
        diff = stop_grouped[True] - stop_grouped[False]
        conclusions['stopwords'] = (
            f'Stopword removal {"improves" if diff > 0 else "decreases"} F1_neg by {abs(diff):.4f}. '
            f'Effect is {"significant" if abs(diff) > 0.01 else "marginal"}.'
        )
    else:
        conclusions['stopwords'] = 'Insufficient data for paired comparison.'

    return conclusions


def _compute_test_metrics_from_predictions(
    predictions_df: pd.DataFrame, threshold: float
) -> dict[str, float]:
    """
    Compute TEST metrics from predictions DataFrame.

    Args:
        predictions_df: DataFrame with y_true, p_neg columns
        threshold: Operating threshold from VAL

    Returns:
        Dict with F1_neg, Recall_neg, Precision_neg
    """
    y_true = predictions_df['y_true'].values
    p_neg = predictions_df['p_neg'].values

    # Apply threshold
    y_pred = np.where(p_neg >= threshold, 0, 1)

    return {
        'F1_neg': f1_score(y_true, y_pred, pos_label=0, zero_division=0.0),
        'Recall_neg': recall_score(y_true, y_pred, pos_label=0, zero_division=0.0),
        'Precision_neg': precision_score(y_true, y_pred, pos_label=0, zero_division=0.0),
        'F1_pos': f1_score(y_true, y_pred, pos_label=1, zero_division=0.0),
        'Recall_pos': recall_score(y_true, y_pred, pos_label=1, zero_division=0.0),
        'Precision_pos': precision_score(y_true, y_pred, pos_label=1, zero_division=0.0),
    }


def generate_ablation_report(
    experiments_dir: Path,
    output_dir: Path,
    test_predictions_path: Path | None = None,
    winner_run_id: str | None = None,
) -> dict[str, Path]:
    """
    Generate complete 4-layer ablation report.

    Args:
        experiments_dir: Path to experiments directory
        output_dir: Path to output directory for reports
        test_predictions_path: Optional path to test_predictions.csv
        winner_run_id: Optional explicit winner run_id (auto-detect if None)

    Returns:
        Dictionary mapping artifact names to output paths
    """
    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'

    log.info(
        json_log(
            'report.start',
            component='reports',
            experiments_dir=str(experiments_dir),
            output_dir=str(output_dir),
        )
    )

    # ==========================================================================
    # Layer 1: Load and sort ablation table
    # ==========================================================================
    df = load_all_runs(experiments_dir)
    validate_schema(df)
    df_sorted = sort_ablation_table(df)

    # Export full sorted table
    ablation_table_path = export_csv(df_sorted, tables_dir / 'ablation_table_sorted.csv')

    # Get winner
    if winner_run_id is not None:
        winner_df = df_sorted[df_sorted['run_id'] == winner_run_id]
        if len(winner_df) == 0:
            raise ValueError(f"Winner run_id '{winner_run_id}' not found in ablation table")
        winner = winner_df.iloc[0]
    else:
        winner = get_winner(df_sorted)
        if winner is None:
            raise ValueError('No valid runs found (all constraint_met == False)')

    log.info(
        json_log(
            'report.winner',
            component='reports',
            run_id=winner['run_id'],
            f1_neg=winner['F1_neg'],
        )
    )

    artifacts: dict[str, Path] = {'ablation_table': ablation_table_path}

    # ==========================================================================
    # Layer 2: Top-K results
    # ==========================================================================
    df_top5 = get_top_k(df_sorted, k=5)
    top5_table_path = export_csv(df_top5, tables_dir / 'top5_table.csv')
    artifacts['top5_table'] = top5_table_path

    # Top-5 bar chart
    top5_fig_path = plot_top_k_f1neg_bar(df_top5, figures_dir / 'layer2_top5_f1neg.png')
    artifacts['top5_figure'] = top5_fig_path

    # TOP5_RESULTS.md
    top5_md = generate_top5_report(df_top5, winner, figures_dir='figures')
    top5_report_path = write_report(top5_md, output_dir / 'TOP5_RESULTS.md')
    artifacts['top5_report'] = top5_report_path

    # ==========================================================================
    # Layer 3: Factor-level analysis
    # ==========================================================================
    c_fig_path = plot_c_vs_f1neg(df_sorted, figures_dir / 'layer3_C_vs_f1neg.png')
    ngram_fig_path = plot_ngram_effect(df_sorted, figures_dir / 'layer3_ngram_effect.png')
    stopwords_fig_path = plot_stopwords_effect(
        df_sorted, figures_dir / 'layer3_stopwords_effect.png'
    )

    artifacts['c_figure'] = c_fig_path
    artifacts['ngram_figure'] = ngram_fig_path
    artifacts['stopwords_figure'] = stopwords_fig_path

    # Generate conclusions
    factor_conclusions = _generate_factor_conclusions(df_sorted)

    # ABLATION_ANALYSIS.md
    analysis_md = generate_ablation_analysis(factor_conclusions, figures_dir='figures')
    analysis_report_path = write_report(analysis_md, output_dir / 'ABLATION_ANALYSIS.md')
    artifacts['analysis_report'] = analysis_report_path

    # ==========================================================================
    # Layer 4: Final model deep dive
    # ==========================================================================
    # VAL confusion matrix (from winner)
    val_cm = winner.get('confusion_matrix', [[0, 0], [0, 0]])
    val_cm_path = plot_confusion_matrix_heatmap(
        val_cm, 'VAL Confusion Matrix', figures_dir / 'layer4_val_confusion_matrix.png'
    )
    artifacts['val_confusion_matrix'] = val_cm_path

    # VAL metrics
    val_metrics = {
        'F1_neg': winner['F1_neg'],
        'Recall_neg': winner['Recall_neg'],
        'Precision_neg': winner['Precision_neg'],
        'F1_pos': winner['F1_pos'],
        'Recall_pos': winner['Recall_pos'],
        'Precision_pos': winner['Precision_pos'],
        'Brier_Score': winner['Brier_Score'],
        'ECE': winner['ECE'],
    }

    # TEST predictions (if available)
    test_metrics: dict[str, float] = {}
    if test_predictions_path is not None:
        test_predictions_path = Path(test_predictions_path)
        if test_predictions_path.exists():
            predictions_df = load_test_predictions(test_predictions_path)
            threshold = winner['threshold']

            # Compute TEST metrics
            test_metrics = _compute_test_metrics_from_predictions(predictions_df, threshold)

            y_true = predictions_df['y_true'].values
            p_neg = predictions_df['p_neg'].values

            # PR curve
            pr_path = plot_pr_curve_neg(
                y_true, p_neg, threshold, figures_dir / 'layer4_pr_curve_neg.png'
            )
            artifacts['pr_curve'] = pr_path

            # Threshold curve
            thresh_path = plot_threshold_curve(
                y_true, p_neg, threshold, figures_dir / 'layer4_threshold_curve.png'
            )
            artifacts['threshold_curve'] = thresh_path

            # Calibration curve
            cal_path = plot_calibration_curve(
                y_true, p_neg, figures_dir / 'layer4_calibration_curve.png'
            )
            artifacts['calibration_curve'] = cal_path

            # VAL vs TEST comparison
            val_test_path = plot_val_vs_test_comparison(
                val_metrics, test_metrics, figures_dir / 'layer4_val_vs_test.png'
            )
            artifacts['val_vs_test'] = val_test_path

    # FINAL_MODEL_REPORT.md
    final_md = generate_final_model_report(winner, val_metrics, test_metrics, figures_dir='figures')
    final_report_path = write_report(final_md, output_dir / 'FINAL_MODEL_REPORT.md')
    artifacts['final_report'] = final_report_path

    log.info(
        json_log(
            'report.completed',
            component='reports',
            n_artifacts=len(artifacts),
            output_dir=str(output_dir),
        )
    )

    return artifacts
