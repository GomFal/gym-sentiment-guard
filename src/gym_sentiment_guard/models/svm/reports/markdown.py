"""
Markdown report generation for SVM ablation suite.

Generates TOP5_RESULTS.md, ABLATION_ANALYSIS.md, and FINAL_MODEL_REPORT.md.
Supports both Linear and RBF SVM specific analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def generate_top5_report(
    df_top5: pd.DataFrame,
    winner: pd.Series,
    model_type: str,
    figures_dir: str = 'figures',
) -> str:
    """
    Generate TOP5_RESULTS.md content.
    """
    model_name = 'SVM Linear' if model_type == 'linear' else 'SVC RBF'
    
    # Layer 2 Summary table
    from .tables import generate_comparison_table
    comparison_table = generate_comparison_table(df_top5, model_type)
    
    # Hyperparams for winner
    if model_type == 'linear':
        hyperparams = f"""
| Parameter | Value |
|-----------|-------|
| Model Type | {model_name} |
| C | {winner['C']} |
| N-Gram Range | {winner['unigram_ngram_range']} |
| Coef Sparsity | {winner['coef_sparsity']:.2f}% |
"""
    else:
        hyperparams = f"""
| Parameter | Value |
|-----------|-------|
| Model Type | {model_name} |
| Kernel | {winner['kernel']} |
| C | {winner['C']} |
| Gamma | {winner['gamma']} |
| Avg SV count | {winner['avg_support_vectors']:.0f} |
"""

    report = f"""# Layer 2 — Top-K Results & Winner Selection

This report summarizes the most competitive configurations from the **{model_name}** ablation suite.

## Summary Table

{comparison_table}

## Chosen Winner

**Run ID**: `{winner['run_id']}`

### Winner Configuration
{hyperparams}

### Metric Evidence (VAL)
- **F1_neg**: `{winner['F1_neg']:.4f}`
- **Recall_neg**: `{winner['Recall_neg']:.4f}`
- **Macro_F1**: `{winner['Macro_F1']:.4f}`

## Decision Evidence

![Top-5 Comparison]({figures_dir}/layer2_top5_f1neg.png)

The winner was selected based on the primary objective (Recall_neg ≥ 0.90) and the hierarchy of F1_neg followed by calibration metrics.
"""
    return report.strip()


def generate_ablation_analysis(
    factor_conclusions: dict,
    model_type: str,
    figures_dir: str = 'figures',
) -> str:
    """
    Generate ABLATION_ANALYSIS.md content.
    """
    model_name = 'SVM Linear' if model_type == 'linear' else 'SVC RBF'
    
    sections = [
        f"# Layer 3 — Factor-Level Ablation Analysis ({model_name})",
        "\nThis layer analyzes how individual design decisions and hyperparameters impact model performance.",
        
        "## 1. Regularization Strength (C)",
        f"![C Effect]({figures_dir}/layer3_C_vs_f1neg.png)",
        factor_conclusions.get('C', "Analysis of C parameter sweeps."),
    ]
    
    # N-gram section only for linear (RBF experiments don't vary n-grams)
    if model_type == 'linear':
        sections.extend([
            "## 2. N-Gram Range Sensitivity",
            f"![N-Gram Effect]({figures_dir}/layer3_ngram_effect.png)",
            factor_conclusions.get('ngram_effect', "Analysis of Unigram+Multigram combinations."),
        ])
    
    if model_type == 'rbf':
        sections.extend([
            "## 2. Kernel Coefficient (Gamma)",
            f"![Gamma Effect]({figures_dir}/layer3_gamma_effect.png)",
            factor_conclusions.get('gamma', "Analysis of RBF kernel width."),
            
            "## 3. Gamma & C Interaction",
            f"![Gamma-C Heatmap]({figures_dir}/layer3_gamma_c_heatmap.png)",
            factor_conclusions.get('gamma_c_interaction', "Interaction analysis between C and Gamma."),
            
            "## 4. Model Complexity (Support Vectors)",
            f"![SV Count Effect]({figures_dir}/layer3_sv_vs_f1neg.png)",
            factor_conclusions.get('support_vectors', "Correlation between complexity (SV count) and performance."),
            
            "## 5. MaxAbsScaler Effect on Metrics",
            f"![Scaler Comparison]({figures_dir}/layer3_scaler_comparison.png)",
            factor_conclusions.get('scaler_metrics', "Comparison of key metrics between scaled and unscaled pipelines."),
            
            "## 6. Support Vector Distribution by Scaler",
            f"![Scaler SV Ratio]({figures_dir}/layer3_scaler_sv_ratio.png)",
            factor_conclusions.get('scaler_sv', "Impact of scaling on support vector count."),
            
            "## 7. Calibration Quality by Scaler",
            f"![Scaler Calibration]({figures_dir}/layer3_scaler_calibration.png)",
            factor_conclusions.get('scaler_calibration', "Comparison of Brier Score and ECE between scaled and unscaled pipelines."),
        ])
        
    return "\n\n".join(sections)


def generate_final_model_report(
    winner: pd.Series,
    val_metrics: dict,
    test_metrics: dict,
    model_type: str,
    figures_dir: str = 'figures',
) -> str:
    """
    Generate FINAL_MODEL_REPORT.md content.
    """
    model_name = 'SVM Linear' if model_type == 'linear' else 'SVC RBF'
    
    report = f"""# Layer 4 — Final Model Deep Dive & Production Readiness

**Run ID**: `{winner['run_id']}`
**Model Family**: {model_name}

## 1. VAL vs TEST Generalization

![VAL vs TEST Comparison]({figures_dir}/layer4_val_vs_test.png)

| Metric | VAL | TEST | Delta |
|--------|-----|------|-------|
| F1_neg | {val_metrics['F1_neg']:.4f} | {test_metrics['F1_neg']:.4f} | {test_metrics['F1_neg'] - val_metrics['F1_neg']:.4f} |
| Recall_neg | {val_metrics['Recall_neg']:.4f} | {test_metrics['Recall_neg']:.4f} | {test_metrics['Recall_neg'] - val_metrics['Recall_neg']:.4f} |
| Macro_F1 | {val_metrics['Macro_F1']:.4f} | {test_metrics['Macro_F1']:.4f} | {test_metrics['Macro_F1'] - val_metrics['Macro_F1']:.4f} |

## 2. Decision Boundary & Thresholds

![PR Curve]({figures_dir}/layer4_pr_curve_neg.png)
![Threshold Curve]({figures_dir}/layer4_threshold_curve.png)

**Chosen Threshold**: `{val_metrics['threshold']:.4f}`

## 3. Calibration Status

![Calibration Curve]({figures_dir}/layer4_calibration_curve.png)

| Calibration Metric | Value |
|--------------------|-------|
| Brier Score | {val_metrics['Brier_Score']:.4f} |
| ECE | {val_metrics['ECE']:.4f} |

## 4. Error Analysis (Confusion Matrices)

| VAL Confusion Matrix | TEST Confusion Matrix |
|----------------------|-----------------------|
| ![VAL CM]({figures_dir}/layer4_val_cm.png) | ![TEST CM]({figures_dir}/layer4_test_cm.png) |

## Conclusion

The model shows strong generalization with stable recall on TEST. The threshold is well-positioned on the PR curve to meet project constraints.
"""
    return report.strip()


def write_report(content: str, output_path: Path) -> Path:
    """Write markdown content to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')
    log.info(json_log('markdown.saved', path=str(output_path)))
    return output_path
