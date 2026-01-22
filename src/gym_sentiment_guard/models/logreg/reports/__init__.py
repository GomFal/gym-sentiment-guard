"""Reports module for Logistic Regression.

Includes error analysis and ablation reporting.
"""

from .ablation_report import generate_ablation_report
from .errors.pipeline import run_error_analysis
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
    generate_comparison_table,
    get_top_k,
    get_winner,
    sort_ablation_table,
)

__all__ = [
    'generate_ablation_report',
    'run_error_analysis',
    # Helper exports (optional, but good for interactive use)
    'load_all_runs',
    'validate_schema',
    'sort_ablation_table',
    'get_top_k',
    'get_winner',
    'generate_comparison_table',
    'generate_top5_report',
    'generate_ablation_analysis',
    'generate_final_model_report',
    'write_report',
    'plot_top_k_f1neg_bar',
    'plot_c_vs_f1neg',
    'plot_ngram_effect',
    'plot_stopwords_effect',
    'plot_confusion_matrix_heatmap',
    'plot_pr_curve_neg',
    'plot_threshold_curve',
    'plot_calibration_curve',
    'plot_val_vs_test_comparison',
    'load_test_predictions',
    'export_csv',
]
