"""
Visualization generation for Layers 2-4 of ablation reporting.

Implements all required plots per REPORTING_STANDARDS.md.
Uses Agg backend for CI/headless compatibility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Headless backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
)

from ..utils.logging import get_logger, json_log

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

    Annotated with Recall_neg and threshold values.

    Args:
        df: DataFrame with top K runs (must have run_id, F1_neg, Recall_neg, threshold)
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by F1_neg for visual clarity (already sorted, but ensure order)
    df_plot = df.sort_values('F1_neg', ascending=True).copy()

    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot['F1_neg'], color=COLOR_PRIMARY, alpha=0.8, height=0.6)

    # Annotations
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(
            row['F1_neg'] + 0.005,
            i,
            f"R={row['Recall_neg']:.3f}, t={row['threshold']:.3f}",
            va='center',
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['run_id'], fontsize=9)
    ax.set_xlabel('F1_neg (VAL)', fontsize=11)
    ax.set_title('Top-5 Ablation Results by F1_neg', fontsize=12, fontweight='bold')
    # Dynamic X-axis scaling for better visual distinction
    x_min = max(0.8, df_plot['F1_neg'].min() - 0.02)
    ax.set_xlim(x_min, df_plot['F1_neg'].max() + 0.05)
    ax.grid(axis='x', alpha=0.3)

    # Highlight winner
    ax.barh(y_pos[-1], df_plot.iloc[-1]['F1_neg'], color=COLOR_VALID, alpha=0.9, height=0.6)

    return _save_figure(fig, output_path)


# =============================================================================
# Layer 3: Factor-Level Analysis
# =============================================================================


def plot_c_vs_f1neg(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot regularization strength (C) vs mean F1_neg.

    Uses categorical X-axis to clearly show all C values.

    Args:
        df: Full ablation table
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    # Group by C value
    grouped = df.groupby('C')['F1_neg'].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values('C')

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use categorical positions for clear visualization of all C values
    x_pos = np.arange(len(grouped))
    ax.errorbar(
        x_pos,
        grouped['mean'],
        yerr=grouped['std'],
        marker='o',
        capsize=4,
        capthick=1.5,
        linewidth=2,
        color=COLOR_PRIMARY,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in grouped['C']], fontsize=10)
    ax.set_xlabel('Regularization Strength (C)', fontsize=11)
    ax.set_ylabel('Mean F1_neg (VAL)', fontsize=11)
    ax.set_title('Regularization Effect on F1_neg', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    return _save_figure(fig, output_path)


def plot_ngram_effect(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot ngram_range effect on F1_neg.

    Uses dynamic Y-axis scaling to make differences visually distinguishable.

    Args:
        df: Full ablation table
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    grouped = df.groupby('ngram_range')['F1_neg'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(grouped))
    ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=4, color=COLOR_PRIMARY, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped['ngram_range'], fontsize=10)
    ax.set_xlabel('ngram_range', fontsize=11)
    ax.set_ylabel('Mean F1_neg (VAL)', fontsize=11)
    ax.set_title('N-gram Range Effect on F1_neg', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Dynamic Y-axis scaling for better visual distinction
    y_min = grouped['mean'].min()
    y_upper = (grouped['mean'] + grouped['std']).max() + 0.01
    y_lower = max(0, y_min - 0.02)  # Start slightly below min, but not below 0
    ax.set_ylim([y_lower, y_upper])

    return _save_figure(fig, output_path)


def plot_stopwords_effect(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot stopwords ON vs OFF effect.

    Uses dynamic Y-axis scaling for better visual distinction.

    Args:
        df: Full ablation table
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    grouped = df.groupby('stopwords_enabled')['F1_neg'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ['OFF', 'ON']
    x_pos = [0, 1]

    # Collect values for Y-axis calculation
    means = []
    stds = []

    # Map boolean to index
    for _, row in grouped.iterrows():
        idx = 1 if row['stopwords_enabled'] else 0
        ax.bar(idx, row['mean'], yerr=row['std'], capsize=5, color=COLOR_PRIMARY, alpha=0.8)
        means.append(row['mean'])
        stds.append(row['std'])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('Stopwords', fontsize=11)
    ax.set_ylabel('Mean F1_neg (VAL)', fontsize=11)
    ax.set_title('Stopword Removal Effect on F1_neg', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Dynamic Y-axis scaling for better visual distinction
    y_lower = max(0, min(means) - 0.02)
    y_upper = max(m + s for m, s in zip(means, stds)) + 0.01
    ax.set_ylim([y_lower, y_upper])

    return _save_figure(fig, output_path)


# =============================================================================
# Layer 4: Final Model Deep Dive
# =============================================================================


def plot_confusion_matrix_heatmap(
    cm: np.ndarray | list,
    title: str,
    output_path: Path,
    labels: list[str] | None = None,
) -> Path:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix (2x2 array)
        title: Plot title (e.g., "VAL Confusion Matrix")
        output_path: Path to save figure
        labels: Class labels

    Returns:
        Path to saved figure
    """
    cm = np.array(cm)
    labels = labels or ['Negative', 'Positive']

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)

    ax.set_title(title, fontsize=12, fontweight='bold')

    return _save_figure(fig, output_path)


def plot_pr_curve_neg(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    threshold: float,
    output_path: Path,
) -> Path:
    """
    Plot Precision-Recall curve for negative class.

    Marks the operating point at selected threshold.

    Args:
        y_true: True labels (0=negative, 1=positive)
        p_neg: Predicted probability of negative class
        threshold: Selected operating threshold
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    # Binary target for negative class
    y_neg_binary = (y_true == 0).astype(int)

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_neg_binary, p_neg)
    ap = average_precision_score(y_neg_binary, p_neg)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, linewidth=2, color=COLOR_PRIMARY, label=f'PR Curve (AP={ap:.3f})')

    # Find operating point at threshold
    y_pred = (p_neg >= threshold).astype(int)
    op_recall = recall_score(y_neg_binary, y_pred)
    op_precision = precision[np.argmin(np.abs(thresholds - threshold))] if len(thresholds) > 0 else 0

    ax.scatter([op_recall], [op_precision], s=150, c=COLOR_SECONDARY, zorder=5, label=f'Operating Point (t={threshold:.3f})')

    ax.set_xlabel('Recall (Negative Class)', fontsize=11)
    ax.set_ylabel('Precision (Negative Class)', fontsize=11)
    ax.set_title('Precision-Recall Curve (Negative Class)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return _save_figure(fig, output_path)


def plot_threshold_curve(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    selected_threshold: float,
    output_path: Path,
    n_thresholds: int = 100,
) -> Path:
    """
    Plot threshold vs Recall_neg and F1_neg curves.

    Highlights the selected threshold.

    Args:
        y_true: True labels
        p_neg: Predicted probability of negative class
        selected_threshold: Threshold selected on VAL
        output_path: Path to save figure
        n_thresholds: Number of threshold points

    Returns:
        Path to saved figure
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    y_neg_binary = (y_true == 0).astype(int)

    # Pre-allocate arrays for performance
    recalls = np.zeros(n_thresholds)
    f1s = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        y_pred = (p_neg >= t).astype(int)
        recalls[i] = recall_score(y_neg_binary, y_pred, zero_division=0)
        f1s[i] = f1_score(y_neg_binary, y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, recalls, linewidth=2, label='Recall_neg', color=COLOR_PRIMARY)
    ax.plot(thresholds, f1s, linewidth=2, label='F1_neg', color=COLOR_SECONDARY)

    # Mark selected threshold
    ax.axvline(selected_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Selected (t={selected_threshold:.3f})')

    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Threshold vs Metrics Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return _save_figure(fig, output_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> Path:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True labels
        p_neg: Predicted probability of negative class
        output_path: Path to save figure
        n_bins: Number of bins

    Returns:
        Path to saved figure
    """
    y_neg_binary = (y_true == 0).astype(int)

    prob_true, prob_pred = calibration_curve(y_neg_binary, p_neg, n_bins=n_bins, strategy='quantile')

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, color=COLOR_PRIMARY, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    ax.set_xlabel('Predicted Probability (Negative Class)', fontsize=11)
    ax.set_ylabel('Observed Frequency', fontsize=11)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return _save_figure(fig, output_path)


def plot_val_vs_test_comparison(
    val_metrics: dict,
    test_metrics: dict,
    output_path: Path,
) -> Path:
    """
    Plot side-by-side VAL vs TEST comparison bar chart.

    Args:
        val_metrics: Dict with F1_neg, Recall_neg, Precision_neg
        test_metrics: Dict with same keys
        output_path: Path to save figure

    Returns:
        Path to saved figure
    """
    metrics = ['F1_neg', 'Recall_neg', 'Precision_neg']
    val_values = [val_metrics.get(m, 0) for m in metrics]
    test_values = [test_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2, val_values, width, label='VAL', color=COLOR_PRIMARY)
    ax.bar(x + width / 2, test_values, width, label='TEST', color=COLOR_SECONDARY)

    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('VAL vs TEST Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels
    for i, v in enumerate(val_values):
        ax.text(i - width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(test_values):
        ax.text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    return _save_figure(fig, output_path)
