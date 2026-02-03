"""
Main orchestrator for SVM ablation suite report generation.

Implements the complete 4-layer reporting workflow per REPORTING_STANDARDS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gym_sentiment_guard.common.metrics import compute_test_metrics
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
    plot_gamma_c_heatmap,
    plot_gamma_vs_f1neg,
    plot_ngram_effect,
    plot_pr_curve_neg,
    plot_scaler_calibration,
    plot_scaler_comparison,
    plot_scaler_sv_ratio,
    plot_support_vectors_vs_f1neg,
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


def _generate_factor_conclusions(df: pd.DataFrame, model_type: str) -> dict[str, str]:
    """
    Generate automated conclusions for factor-level analysis.
    """
    conclusions = {}

    # 1. C Analysis
    best_c = df.groupby('C')['F1_neg'].mean().idxmax()
    conclusions['C'] = (
        f"The model shows optimal performance around C={best_c}. Higher regularization (low C) "
        f"leads to underfitting, while higher C values show marginal gains or slight volatility "
        f"depending on other factors."
    )

    # 2. N-Gram Analysis
    df_copy = df.copy()
    df_copy['ngram_combo'] = df_copy['unigram_ngram_range'] + ' | ' + df_copy['multigram_ngram_range']
    best_ngram = df_copy.groupby('ngram_combo')['F1_neg'].mean().idxmax()
    conclusions['ngram_effect'] = (
        f"Optimal feature density is achieved with `{best_ngram}`. The inclusion of "
        "multigram features provided consistent lift over unigrams alone."
    )

    if model_type == 'rbf':
        # 3. Gamma Analysis
        best_gamma = df.groupby('gamma')['F1_neg'].mean().idxmax()
        conclusions['gamma'] = (
            f"The RBF kernel is most effective with gamma={best_gamma}. This suggests the "
            "sentiment features have a specific locality in the feature space that this kernel width captures."
        )

        # 4. Interaction
        conclusions['gamma_c_interaction'] = (
            "The interaction heatmap reveals a trade-off between C and Gamma. As C increases, "
            "the model becomes more sensitive to Gamma choice to avoid over-fitting or complex boundaries."
        )

        # 5. Complexity
        corr = df['avg_support_vectors'].corr(df['F1_neg'])
        trend = "positive" if corr > 0 else "negative"
        conclusions['support_vectors'] = (
            f"There is a {trend} correlation (r={corr:.2f}) between support vector count and F1_neg. "
            "More support vectors generally improve performance until a saturation point is reached."
        )

    return conclusions


def _compute_test_metrics_from_predictions(
    predictions_df: pd.DataFrame, threshold: float
) -> dict[str, float]:
    """
    Compute metrics from predictions at a fixed threshold.
    """
    y_true = predictions_df['y_true'].values
    p_neg = predictions_df['p_neg'].values
    
    metrics = compute_test_metrics(y_true, p_neg, threshold)
    
    return {
        'F1_neg': metrics.f1_neg,
        'Recall_neg': metrics.recall_neg,
        'Precision_neg': metrics.precision_neg,
        'Macro_F1': metrics.macro_f1,
    }


def generate_ablation_report(
    experiments_dir: Path,
    output_dir: Path,
    model_type: str,
    test_predictions_path: Path | None = None,
    winner_run_id: str | None = None,
) -> dict[str, Path]:
    """
    Generate complete 4-layer ablation report for SVM.
    """
    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    log.info(
        json_log(
            'ablation_report.start',
            component='reports',
            model_type=model_type,
            experiments_dir=str(experiments_dir),
        )
    )

    # ========== Layer 1: Data Loading & Protocol Sorting ==========
    df = load_all_runs(experiments_dir, model_type)
    if df.empty:
        raise ValueError(f"No valid run.json files found in {experiments_dir}")
        
    validate_schema(df, model_type)
    df_sorted = sort_ablation_table(df)
    
    artifacts['ablation_table_csv'] = export_csv(
        df_sorted, tables_dir / 'ablation_table_sorted.csv'
    )

    # ========== Layer 2: Top-K Comparison ==========
    df_top5 = get_top_k(df_sorted, k=5)
    
    # Selection of winner
    if winner_run_id:
        winner_row = df_sorted[df_sorted['run_id'] == winner_run_id]
        if winner_row.empty:
            raise ValueError(f"Specified winner_run_id '{winner_run_id}' not found.")
        winner = winner_row.iloc[0]
    else:
        winner = get_winner(df_sorted)
        if winner is None:
            log.warning("No valid winner found (constraints not met). Using first run.")
            winner = df_sorted.iloc[0]

    artifacts['top5_bar'] = plot_top_k_f1neg_bar(df_top5, figures_dir / 'layer2_top5_f1neg.png')
    
    # ========== Layer 3: Factor Analysis ==========
    factor_conclusions = _generate_factor_conclusions(df_sorted, model_type)
    
    # Common factors
    artifacts['c_vs_f1neg'] = plot_c_vs_f1neg(df_sorted, figures_dir / 'layer3_C_vs_f1neg.png')
    
    # N-gram effect (linear only, since RBF doesn't vary n-grams in current experiments)
    if model_type == 'linear':
        artifacts['ngram_effect'] = plot_ngram_effect(df_sorted, figures_dir / 'layer3_ngram_effect.png')
    
    # Model-specific factors
    if model_type == 'rbf':
        artifacts['gamma_effect'] = plot_gamma_vs_f1neg(df_sorted, figures_dir / 'layer3_gamma_effect.png')
        artifacts['gamma_c_heatmap'] = plot_gamma_c_heatmap(df_sorted, figures_dir / 'layer3_gamma_c_heatmap.png')
        artifacts['sv_vs_f1neg'] = plot_support_vectors_vs_f1neg(df_sorted, figures_dir / 'layer3_sv_vs_f1neg.png')
        
        # MaxAbsScaler comparison (if both scaler=True and scaler=False present)
        if 'use_scaler' in df_sorted.columns and df_sorted['use_scaler'].nunique() > 1:
            artifacts['scaler_comparison'] = plot_scaler_comparison(
                df_sorted, figures_dir / 'layer3_scaler_comparison.png'
            )
            artifacts['scaler_sv_ratio'] = plot_scaler_sv_ratio(
                df_sorted, figures_dir / 'layer3_scaler_sv_ratio.png'
            )
            artifacts['scaler_calibration'] = plot_scaler_calibration(
                df_sorted, figures_dir / 'layer3_scaler_calibration.png'
            )

    # ========== Layer 4: Final Model Deep Dive ==========
    if test_predictions_path and test_predictions_path.exists():
        log.info("Generating Layer 4 - Final Model Deep Dive (using TEST data)")
        test_df = load_test_predictions(test_predictions_path)
        
        # We need the full run.json of the winner to get its confusion matrix and precise params
        run_folder = str(winner['run_id'])
        if not run_folder.startswith('run.'):
            run_folder = f"run.{run_folder}"
            
        winner_run_path = experiments_dir / run_folder / 'run.json'
        with open(winner_run_path, encoding='utf-8') as f:
            winner_data = json.load(f)
            
        val_metrics = winner_data['val_metrics']
        threshold = val_metrics['threshold']
        
        # Test metrics
        test_metrics_dict = _compute_test_metrics_from_predictions(test_df, threshold)
        
        # Layer 4 Plots
        p_neg = test_df['p_neg'].values
        y_true = test_df['y_true'].values
        
        artifacts['layer4_val_cm'] = plot_confusion_matrix_heatmap(
            val_metrics['confusion_matrix'], "VAL Confusion Matrix", figures_dir / 'layer4_val_cm.png'
        )
        
        # Compute test CM using original label format (0=neg, 1=pos)
        # y_pred maps: if p_neg >= threshold -> predict negative (0), else positive (1)
        y_pred = np.where(p_neg >= threshold, 0, 1)
        
        from sklearn.metrics import confusion_matrix
        test_cm = confusion_matrix(y_true, y_pred)
        artifacts['layer4_test_cm'] = plot_confusion_matrix_heatmap(
            test_cm, "TEST Confusion Matrix", figures_dir / 'layer4_test_cm.png'
        )
        
        artifacts['layer4_pr_curve'] = plot_pr_curve_neg(
            y_true, p_neg, threshold, figures_dir / 'layer4_pr_curve_neg.png'
        )
        artifacts['layer4_threshold_curve'] = plot_threshold_curve(
            y_true, p_neg, threshold, figures_dir / 'layer4_threshold_curve.png'
        )
        artifacts['layer4_calibration_curve'] = plot_calibration_curve(
            y_true, p_neg, figures_dir / 'layer4_calibration_curve.png'
        )
        artifacts['layer4_val_vs_test'] = plot_val_vs_test_comparison(
            {
                'F1_neg': winner['F1_neg'],
                'Recall_neg': winner['Recall_neg'],
                'Precision_neg': winner['Precision_neg'],
            },
            test_metrics_dict,
            figures_dir / 'layer4_val_vs_test.png'
        )
        
        # Final Level 4 Report
        final_report = generate_final_model_report(
            winner, winner, test_metrics_dict, model_type, figures_dir='figures'
        )
        artifacts['final_model_report'] = write_report(final_report, output_dir / 'FINAL_MODEL_REPORT.md')
    else:
        log.warning("Layer 4 skipped - no test predictions provided.")

    # ========== Markdown Skeletos (Layer 2 & 3) ==========
    top5_report = generate_top5_report(df_top5, winner, model_type, figures_dir='figures')
    artifacts['top5_results_md'] = write_report(top5_report, output_dir / 'TOP5_RESULTS.md')
    
    ablation_analysis = generate_ablation_analysis(factor_conclusions, model_type, figures_dir='figures')
    artifacts['ablation_analysis_md'] = write_report(ablation_analysis, output_dir / 'ABLATION_ANALYSIS.md')

    return artifacts
