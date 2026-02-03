"""SVM ablation reports module.

Generates 4-layer analysis reports per REPORTING_STANDARDS.md.
"""

from .ablation_report import generate_ablation_report
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
    plot_support_vectors_vs_f1neg,
    plot_threshold_curve,
    plot_top_k_f1neg_bar,
    plot_val_vs_test_comparison,
)
from .schema import load_all_runs, load_test_predictions, validate_schema
from .tables import (
    export_csv,
    generate_comparison_table,
    get_top_k,
    get_winner,
    sort_ablation_table,
)

__all__ = [
    'generate_ablation_report',
    # Schema
    'load_all_runs',
    'validate_schema',
    'load_test_predictions',
    # Tables
    'sort_ablation_table',
    'get_top_k',
    'get_winner',
    'export_csv',
    'generate_comparison_table',
    # Plots
    'plot_top_k_f1neg_bar',
    'plot_c_vs_f1neg',
    'plot_gamma_vs_f1neg',
    'plot_ngram_effect',
    'plot_gamma_c_heatmap',
    'plot_support_vectors_vs_f1neg',
    'plot_confusion_matrix_heatmap',
    'plot_pr_curve_neg',
    'plot_threshold_curve',
    'plot_calibration_curve',
    'plot_val_vs_test_comparison',
    # Markdown
    'generate_top5_report',
    'generate_ablation_analysis',
    'generate_final_model_report',
    'write_report',
]
