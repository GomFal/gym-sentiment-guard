"""
Visualization generation for Layers 2-4 of SVM ablation reporting.

Implements all required plots per REPORTING_STANDARDS.md.
Includes RBF-specific plots (Gamma+C heatmap, SV counts) and shared plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
)

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)

# Consistent styling
FIGURE_DPI = 150
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
COLOR_VALID = '#2ca02c'
COLOR_INVALID = '#d62728'


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save figure with consistent settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info(json_log('plots.saved', component='reports', path=str(path)))
    return path


# =============================================================================
# Layer 2: Top-K Visualizations
# =============================================================================


def plot_top_k_f1neg_bar(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate horizontal bar chart for Top-K F1_neg.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.sort_values('F1_neg', ascending=True).copy()

    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot['F1_neg'], color=COLOR_PRIMARY, alpha=0.8, height=0.6)

    # Annotations
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(
            row['F1_neg'] + 0.005,
            i,
            f'R={row["Recall_neg"]:.3f}, t={row["threshold"]:.3f}',
            va='center',
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['run_id'], fontsize=9)
    ax.set_xlabel('F1_neg (VAL)', fontsize=11)
    ax.set_title('Top Ablation Results by F1_neg', fontsize=12, fontweight='bold')
    # Set x-axis to start near min value for better visualization of differences
    x_min = max(0.8, df_plot['F1_neg'].min() - 0.01)
    ax.set_xlim(x_min, 1.0)
    ax.grid(axis='x', alpha=0.3)

    return _save_figure(fig, output_path)


# =============================================================================
# Layer 3: Factor-Level Analysis (Shared)
# =============================================================================


def plot_c_vs_f1neg(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot regularization strength (C) vs mean F1_neg.
    """
    grouped = df.groupby('C')['F1_neg'].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values('C')

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(grouped))
    
    ax.errorbar(
        x_pos,
        grouped['mean'],
        yerr=grouped['std'],
        marker='o', capsize=4, linewidth=2, color=COLOR_PRIMARY
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c:.4g}" for c in grouped['C']], fontsize=10)
    ax.set_xlabel('Regularization Strength (C)', fontsize=11)
    ax.set_ylabel('Mean F1_neg (VAL)', fontsize=11)
    ax.set_title('C Parameter Effect on F1_neg', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    return _save_figure(fig, output_path)


def plot_ngram_effect(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot combined effect of Unigram and Multigram ngram_ranges.
    """
    # Create combined label
    df = df.copy()
    df['ngram_combo'] = df['unigram_ngram_range'] + ' | ' + df['multigram_ngram_range']
    
    grouped = df.groupby('ngram_combo')['F1_neg'].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(grouped))
    
    ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=4, color=COLOR_PRIMARY, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped['ngram_combo'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean F1_neg (VAL)', fontsize=11)
    ax.set_title('N-gram Range Sensitivity', fontsize=12, fontweight='bold')
    
    y_min = max(0.0, grouped['mean'].min() - 0.05)
    ax.set_ylim(y_min, 1.0)
    ax.grid(axis='y', alpha=0.3)

    return _save_figure(fig, output_path)


# =============================================================================
# Layer 3: Factor-Level Analysis (RBF Specific)
# =============================================================================


def plot_gamma_vs_f1neg(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot Gamma vs mean F1_neg for RBF SVM.
    """
    grouped = df.groupby('gamma')['F1_neg'].agg(['mean', 'std']).reset_index()
    
    # Handle mixed types (floats + 'scale' string) by sorting numerics first, then strings
    def gamma_sort_key(val):
        try:
            return (0, float(val))  # Numerics first, sorted by value
        except (ValueError, TypeError):
            return (1, str(val))  # Strings last, sorted alphabetically
    
    grouped['_sort_key'] = grouped['gamma'].apply(gamma_sort_key)
    grouped = grouped.sort_values('_sort_key').drop(columns=['_sort_key'])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(grouped))
    
    ax.errorbar(
        x_pos,
        grouped['mean'],
        yerr=grouped['std'],
        marker='s', capsize=4, linewidth=2, color=COLOR_SECONDARY
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped['gamma'], fontsize=10)
    ax.set_xlabel('Kernel Coefficient (Gamma)', fontsize=11)
    ax.set_ylabel('Mean F1_neg (VAL)', fontsize=11)
    ax.set_title('Gamma Parameter Effect on F1_neg', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    return _save_figure(fig, output_path)


def plot_gamma_c_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Interaction heatmap of Gamma and C.
    """
    # Pivot for heatmap
    pivot = df.pivot_table(index='C', columns='gamma', values='F1_neg', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'F1_neg'})
    
    ax.set_title('Interaction of C and Gamma (Mean F1_neg)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Gamma', fontsize=11)
    ax.set_ylabel('C (Inverse Regularization)', fontsize=11)

    return _save_figure(fig, output_path)


def plot_support_vectors_vs_f1neg(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot avg_support_vectors vs F1_neg.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scatter plot with regression line trend
    sns.regplot(
        data=df, x='avg_support_vectors', y='F1_neg',
        scatter_kws={'alpha': 0.5, 'color': COLOR_PRIMARY},
        line_kws={'color': COLOR_INVALID}, ax=ax
    )
    
    ax.set_xlabel('Average Support Vector Count', fontsize=11)
    ax.set_ylabel('F1_neg (VAL)', fontsize=11)
    ax.set_title('Model Complexity (SV Count) vs Performance', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    return _save_figure(fig, output_path)


# =============================================================================
# Layer 3: MaxAbsScaler Comparison (RBF Specific)
# =============================================================================


def plot_scaler_comparison(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Compare key metrics between use_scaler=True vs False.
    
    Creates a grouped bar chart showing mean ± std for:
    - F1_neg
    - Recall_neg
    - Brier_Score
    - ECE
    """
    if 'use_scaler' not in df.columns or df['use_scaler'].nunique() < 2:
        log.warning(json_log('plots.scaler_comparison.skipped', reason='insufficient_groups'))
        # Create placeholder plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Insufficient data: need both scaler=True and False',
                ha='center', va='center', fontsize=12)
        ax.set_title('MaxAbsScaler Comparison (Insufficient Data)')
        return _save_figure(fig, output_path)
    
    metrics = ['F1_neg', 'Recall_neg', 'Brier_Score', 'ECE']
    
    # Group and aggregate
    scaled = df[df['use_scaler'] == True]  # noqa: E712
    unscaled = df[df['use_scaler'] == False]  # noqa: E712
    
    scaled_means = [scaled[m].mean() for m in metrics]
    scaled_stds = [scaled[m].std() for m in metrics]
    unscaled_means = [unscaled[m].mean() for m in metrics]
    unscaled_stds = [unscaled[m].std() for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, scaled_means, width, yerr=scaled_stds, 
                   label='MaxAbsScaler=True', color=COLOR_PRIMARY, alpha=0.8, capsize=4)
    bars2 = ax.bar(x + width/2, unscaled_means, width, yerr=unscaled_stds,
                   label='MaxAbsScaler=False', color=COLOR_SECONDARY, alpha=0.8, capsize=4)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('MaxAbsScaler Effect on Key Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis to accommodate all metrics (some near 0, some near 1)
    ax.set_ylim(0, max(max(scaled_means), max(unscaled_means)) * 1.15)
    
    log.info(json_log('plots.scaler_comparison.generated', 
                      scaled_count=len(scaled), unscaled_count=len(unscaled)))
    
    return _save_figure(fig, output_path)


def plot_scaler_sv_ratio(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Box plot comparing avg_support_vectors between scaler groups.
    
    Shows distribution of support vector counts for:
    - use_scaler=True
    - use_scaler=False
    """
    if 'use_scaler' not in df.columns or df['use_scaler'].nunique() < 2:
        log.warning(json_log('plots.scaler_sv_ratio.skipped', reason='insufficient_groups'))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Insufficient data: need both scaler=True and False',
                ha='center', va='center', fontsize=12)
        ax.set_title('Support Vector Distribution (Insufficient Data)')
        return _save_figure(fig, output_path)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for box plot
    df_plot = df.copy()
    df_plot['Scaler'] = df_plot['use_scaler'].map({True: 'Scaled', False: 'Unscaled'})
    
    sns.boxplot(data=df_plot, x='Scaler', y='avg_support_vectors', 
                palette=[COLOR_PRIMARY, COLOR_SECONDARY], ax=ax)
    
    # Add individual points
    sns.stripplot(data=df_plot, x='Scaler', y='avg_support_vectors',
                  color='black', alpha=0.4, size=4, ax=ax)
    
    ax.set_xlabel('MaxAbsScaler', fontsize=11)
    ax.set_ylabel('Average Support Vector Count', fontsize=11)
    ax.set_title('Support Vector Distribution by Scaler Setting', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add summary stats
    for i, scaler in enumerate(['Scaled', 'Unscaled']):
        subset = df_plot[df_plot['Scaler'] == scaler]['avg_support_vectors']
        median = subset.median()
        ax.annotate(f'med={median:.0f}', xy=(i, median), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, color='darkblue')
    
    return _save_figure(fig, output_path)


def plot_scaler_calibration(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Side-by-side comparison of calibration metrics.
    
    Creates 1x2 subplot:
    - Left: Brier Score distribution by scaler
    - Right: ECE distribution by scaler
    """
    if 'use_scaler' not in df.columns or df['use_scaler'].nunique() < 2:
        log.warning(json_log('plots.scaler_calibration.skipped', reason='insufficient_groups'))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'Insufficient data: need both scaler=True and False',
                ha='center', va='center', fontsize=12)
        ax.set_title('Calibration Comparison (Insufficient Data)')
        return _save_figure(fig, output_path)
    
    df_plot = df.copy()
    df_plot['Scaler'] = df_plot['use_scaler'].map({True: 'Scaled', False: 'Unscaled'})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Brier Score
    sns.boxplot(data=df_plot, x='Scaler', y='Brier_Score',
                palette=[COLOR_PRIMARY, COLOR_SECONDARY], ax=axes[0])
    sns.stripplot(data=df_plot, x='Scaler', y='Brier_Score',
                  color='black', alpha=0.4, size=4, ax=axes[0])
    axes[0].set_xlabel('MaxAbsScaler', fontsize=11)
    axes[0].set_ylabel('Brier Score', fontsize=11)
    axes[0].set_title('Brier Score (lower = better)', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Right: ECE
    sns.boxplot(data=df_plot, x='Scaler', y='ECE',
                palette=[COLOR_PRIMARY, COLOR_SECONDARY], ax=axes[1])
    sns.stripplot(data=df_plot, x='Scaler', y='ECE',
                  color='black', alpha=0.4, size=4, ax=axes[1])
    axes[1].set_xlabel('MaxAbsScaler', fontsize=11)
    axes[1].set_ylabel('ECE', fontsize=11)
    axes[1].set_title('Expected Calibration Error (lower = better)', fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    fig.suptitle('Calibration Quality by Scaler Setting', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return _save_figure(fig, output_path)


# =============================================================================
# Layer 4: Final Model Deep Dive
# =============================================================================


def plot_confusion_matrix_heatmap(
    cm: np.ndarray | list, title: str, output_path: Path
) -> Path:
    """Plot confusion matrix heatmap."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return _save_figure(fig, output_path)


def plot_pr_curve_neg(
    y_true: np.ndarray, p_neg: np.ndarray, threshold: float, output_path: Path
) -> Path:
    """Plot Precision-Recall curve for negative class."""
    y_neg = (y_true == 0).astype(int)
    precision, recall, _ = precision_recall_curve(y_neg, p_neg)
    ap = average_precision_score(y_neg, p_neg)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, color=COLOR_PRIMARY, label=f'PR Curve (AP={ap:.3f})')
    
    # Mark operating point
    # Find nearest threshold index or just re-apply threshold to get recall/precision
    y_pred = (p_neg >= threshold).astype(int)
    op_recall = recall_score(y_neg, y_pred)
    # Precision is trickier to get exactly from precision array without matching threshold,
    # but we can recompute it
    tp = np.sum((y_pred == 1) & (y_neg == 1))
    fp = np.sum((y_pred == 1) & (y_neg == 0))
    op_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    ax.scatter(op_recall, op_precision, s=150, c=COLOR_SECONDARY, zorder=5, 
               label=f'Selected Point (t={threshold:.3f})')
    
    ax.set_xlabel('Recall (Negative Class)', fontsize=11)
    ax.set_ylabel('Precision (Negative Class)', fontsize=11)
    ax.set_title('Precision-Recall Curve (Negative)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    return _save_figure(fig, output_path)


def plot_threshold_curve(
    y_true: np.ndarray, p_neg: np.ndarray, selected_threshold: float, output_path: Path
) -> Path:
    """Plot threshold vs metrics sensitivity."""
    thresholds = np.linspace(0, 1, 100)
    y_neg = (y_true == 0).astype(int)
    
    recalls = []
    f1s = []
    for t in thresholds:
        y_pred = (p_neg >= t).astype(int)
        recalls.append(recall_score(y_neg, y_pred, zero_division=0))
        f1s.append(f1_score(y_neg, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, recalls, linewidth=2, label='Recall_neg', color=COLOR_PRIMARY)
    ax.plot(thresholds, f1s, linewidth=2, label='F1_neg', color=COLOR_SECONDARY)
    
    ax.axvline(selected_threshold, color='red', linestyle='--', label=f'Chosen t={selected_threshold:.3f}')
    
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Threshold Sensitivity Analysis', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    return _save_figure(fig, output_path)


def plot_calibration_curve(
    y_true: np.ndarray, p_neg: np.ndarray, output_path: Path
) -> Path:
    """Plot reliability diagram."""
    y_neg = (y_true == 0).astype(int)
    prob_true, prob_pred = calibration_curve(y_neg, p_neg, n_bins=10, strategy='quantile')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, color=COLOR_PRIMARY, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives (Negative Class)', fontsize=11)
    ax.set_title('Calibration Reliability Diagram', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return _save_figure(fig, output_path)


def plot_val_vs_test_comparison(
    val_metrics: dict, test_metrics: dict, output_path: Path
) -> Path:
    """Side-by-side performance comparison."""
    metrics = ['F1_neg', 'Recall_neg', 'Precision_neg']
    val_vals = [val_metrics.get(m, 0) for m in metrics]
    test_vals = [test_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_val = ax.bar(x - width/2, val_vals, width, label='VAL', color=COLOR_PRIMARY, alpha=0.8)
    bars_test = ax.bar(x + width/2, test_vals, width, label='TEST', color=COLOR_SECONDARY, alpha=0.8)

    # Add value labels on bars
    for bars in [bars_val, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('VAL vs TEST Generalization', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])

    return _save_figure(fig, output_path)
