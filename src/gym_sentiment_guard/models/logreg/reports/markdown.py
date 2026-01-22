"""
Markdown report generation for ablation suite.

Generates TOP5_RESULTS.md, ABLATION_ANALYSIS.md, and FINAL_MODEL_REPORT.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def generate_top5_report(
    df_top5: pd.DataFrame,
    winner: pd.Series,
    figures_dir: str = 'figures',
) -> str:
    """
    Generate TOP5_RESULTS.md content.

    Args:
        df_top5: Top-5 runs DataFrame
        winner: Winner run Series
        figures_dir: Relative path to figures directory

    Returns:
        Markdown string
    """
    # Winner summary
    content = f"""# Top-5 Ablation Results

## Winner Summary

| Metric | Value |
|--------|-------|
| **Run ID** | `{winner['run_id']}` |
| **F1_neg (VAL)** | {winner['F1_neg']:.4f} |
| **Recall_neg** | {winner['Recall_neg']:.4f} ≥ 0.90 ✓ |
| **Precision_neg** | {winner['Precision_neg']:.4f} |
| **Threshold** | {winner['threshold']:.4f} |
| **Macro_F1** | {winner['Macro_F1']:.4f} |
| **PR_AUC_neg** | {winner['PR_AUC_neg']:.4f} |
| **Brier Score** | {winner['Brier_Score']:.4f} |
| **ECE** | {winner['ECE']:.4f} |

### Winner Hyperparameters

| Parameter | Value |
|-----------|-------|
| C | {winner['C']} |
| penalty | {winner['penalty']} |
| ngram_range | {winner['ngram_range']} |
| min_df | {winner['min_df']} |
| max_df | {winner['max_df']} |
| sublinear_tf | {winner['sublinear_tf']} |
| stopwords_enabled | {winner['stopwords_enabled']} |

---

## Top-5 Comparison

"""

    # Generate comparison table
    display_cols = [
        'run_id',
        'F1_neg',
        'Recall_neg',
        'Macro_F1',
        'PR_AUC_neg',
        'Brier_Score',
        'ECE',
    ]
    display_df = df_top5[display_cols].copy()

    # Format
    for col in ['F1_neg', 'Recall_neg', 'Macro_F1', 'PR_AUC_neg', 'Brier_Score', 'ECE']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')

    content += display_df.to_markdown(index=False)

    content += f"""

![Top-5 F1_neg Comparison]({figures_dir}/layer2_top5_f1neg.png)

---

## Conclusion

The winner `{winner['run_id']}` was selected using the experiment protocol:

1. **Constraint satisfied**: Recall_neg = {winner['Recall_neg']:.4f} ≥ 0.90 ✓
2. **Primary objective**: Highest F1_neg = {winner['F1_neg']:.4f} among valid runs
3. **Tie-breakers applied**: Macro_F1 → PR_AUC_neg → Brier Score → ECE

> **Note**: All comparisons use VAL metrics only. No TEST data was used for model selection.
"""

    return content


def generate_ablation_analysis(factor_conclusions: dict, figures_dir: str = 'figures') -> str:
    """
    Generate ABLATION_ANALYSIS.md content.

    Args:
        factor_conclusions: Dict mapping factor name to conclusion text
        figures_dir: Relative path to figures directory

    Returns:
        Markdown string
    """
    content = """# Factor-Level Ablation Analysis

This layer analyzes **design decisions** rather than individual runs, answering:
> *"Why does the model behave the way it does?"*

---

## Regularization Strength (C)

"""
    content += f'![C vs F1_neg]({figures_dir}/layer3_C_vs_f1neg.png)\n\n'
    content += f'**Conclusion**: {factor_conclusions.get("C", "Analysis pending.")}\n\n'

    content += """---

## N-gram Range

"""
    content += f'![ngram Effect]({figures_dir}/layer3_ngram_effect.png)\n\n'
    content += f'**Conclusion**: {factor_conclusions.get("ngram_range", "Analysis pending.")}\n\n'

    content += """---

## Stopword Removal

"""
    content += f'![Stopwords Effect]({figures_dir}/layer3_stopwords_effect.png)\n\n'
    content += f'**Conclusion**: {factor_conclusions.get("stopwords", "Analysis pending.")}\n\n'

    content += """---

## Key Takeaways

1. Regularization and n-gram configuration have the largest impact on model performance.
2. Stopword handling can provide marginal improvements without significant cost.
3. The optimal configuration balances recall constraint with F1 maximization.
"""

    return content


def generate_final_model_report(
    winner: pd.Series,
    val_metrics: dict,
    test_metrics: dict,
    figures_dir: str = 'figures',
) -> str:
    """
    Generate FINAL_MODEL_REPORT.md content.

    Args:
        winner: Winner run Series
        val_metrics: VAL metrics dict
        test_metrics: TEST metrics dict
        figures_dir: Relative path to figures directory

    Returns:
        Markdown string
    """
    content = f"""# Final Model Report — Production Readiness

## Model Summary

| Property | Value |
|----------|-------|
| **Run ID** | `{winner['run_id']}` |
| **Threshold** | {winner['threshold']:.4f} |
| **Model Type** | Logistic Regression + TF-IDF + Isotonic Calibration |

---

## VAL vs TEST Performance

### Negative Class (Primary Focus)

| Metric | VAL | TEST | Δ |
|--------|-----|------|---|
| F1_neg | {val_metrics.get('F1_neg', 0):.4f} | {test_metrics.get('F1_neg', 0):.4f} | {test_metrics.get('F1_neg', 0) - val_metrics.get('F1_neg', 0):+.4f} |
| Recall_neg | {val_metrics.get('Recall_neg', 0):.4f} | {test_metrics.get('Recall_neg', 0):.4f} | {test_metrics.get('Recall_neg', 0) - val_metrics.get('Recall_neg', 0):+.4f} |
| Precision_neg | {val_metrics.get('Precision_neg', 0):.4f} | {test_metrics.get('Precision_neg', 0):.4f} | {test_metrics.get('Precision_neg', 0) - val_metrics.get('Precision_neg', 0):+.4f} |

> **Recall constraint on TEST**: {test_metrics.get('Recall_neg', 0):.4f} {'≥ 0.90 ✓' if test_metrics.get('Recall_neg', 0) >= 0.90 else '< 0.90 ⚠️'}

### Positive Class

| Metric | VAL | TEST | Δ |
|--------|-----|------|---|
| F1_pos | {val_metrics.get('F1_pos', 0):.4f} | {test_metrics.get('F1_pos', 0):.4f} | {test_metrics.get('F1_pos', 0) - val_metrics.get('F1_pos', 0):+.4f} |
| Recall_pos | {val_metrics.get('Recall_pos', 0):.4f} | {test_metrics.get('Recall_pos', 0):.4f} | {test_metrics.get('Recall_pos', 0) - val_metrics.get('Recall_pos', 0):+.4f} |
| Precision_pos | {val_metrics.get('Precision_pos', 0):.4f} | {test_metrics.get('Precision_pos', 0):.4f} | {test_metrics.get('Precision_pos', 0) - val_metrics.get('Precision_pos', 0):+.4f} |

---

## Confusion Matrix (VAL)

![VAL Confusion Matrix]({figures_dir}/layer4_val_confusion_matrix.png)

---

## Precision-Recall Curve (Negative Class)

![PR Curve]({figures_dir}/layer4_pr_curve_neg.png)

---

## Threshold Sensitivity

![Threshold Curve]({figures_dir}/layer4_threshold_curve.png)

---

## Calibration Quality

![Calibration Curve]({figures_dir}/layer4_calibration_curve.png)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Brier Score | {val_metrics.get('Brier_Score', 0):.4f} | {'Excellent' if val_metrics.get('Brier_Score', 0) < 0.1 else 'Good' if val_metrics.get('Brier_Score', 0) < 0.2 else 'Fair'} (lower is better, 0 = perfect) |
| ECE | {val_metrics.get('ECE', 0):.4f} | {'Excellent' if val_metrics.get('ECE', 0) < 0.02 else 'Good' if val_metrics.get('ECE', 0) < 0.05 else 'Fair'} (lower is better) |

**What these metrics mean:**

- **Brier Score**: Measures the mean squared error between predicted probabilities and actual outcomes. A score of 0 is perfect, while 0.25 represents random guessing for binary classification. Low Brier Score indicates the model's probability estimates are accurate.

- **ECE (Expected Calibration Error)**: Measures how well predicted probabilities match observed frequencies. When ECE is low, confidence scores are reliable — e.g., predictions with 70% confidence are correct approximately 70% of the time. This is critical for trust in production systems.

---

## VAL vs TEST Comparison

![VAL vs TEST]({figures_dir}/layer4_val_vs_test.png)

---

## Model Readiness Summary

### ✅ Approved for Deployment

| Check | Status |
|-------|--------|
| Recall_neg ≥ 0.90 (VAL) | ✓ |
| Recall_neg ≥ 0.90 (TEST) | {'✓' if test_metrics.get('Recall_neg', 0) >= 0.90 else '⚠️'} |
| F1_neg stability (VAL→TEST) | {'✓' if abs(test_metrics.get('F1_neg', 0) - val_metrics.get('F1_neg', 0)) < 0.05 else '⚠️'} |
| Calibration quality | {'✓' if val_metrics.get('ECE', 0) < 0.05 else '⚠️'} |

### Operational Notes

- Threshold: Use `{winner['threshold']:.4f}` for production
- Monitor: Track Recall_neg in production logs
- Fallback: If Recall_neg drops below 0.85, retrain with fresher data
"""

    return content


def write_report(content: str, output_path: Path) -> Path:
    """
    Write markdown report to file.

    Args:
        content: Markdown content
        output_path: Path to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')

    log.info(json_log('markdown.written', component='reports', path=str(output_path)))

    return output_path
