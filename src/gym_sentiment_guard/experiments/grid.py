"""
Parameter grid definitions for LogReg ablation studies.

Defines the ablation surface per EXPERIMENT_PROTOCOL.md §5.
Only parameters listed here may vary during experiments.
"""

from __future__ import annotations

# Import stopwords from common module
from ..common.stopwords import (
    STOPWORD_PRESETS,
    STOPWORDS_NEVER_REMOVE,
    STOPWORDS_SAFE,
    resolve_stop_words,
)

# =============================================================================
# TF-IDF Parameters (§5.1)
# =============================================================================

TFIDF_GRID: dict[str, list] = {
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'min_df': [1, 2, 5],
    'max_df': [0.90, 0.95, 1.0],
    'sublinear_tf': [False, True],
    # norm: default "l2" (treat changes as explicit ablations)
}

# =============================================================================
# Logistic Regression Parameters (§5.2)
# =============================================================================

LOGREG_GRID: dict[str, list] = {
    'penalty': ['l2', 'l1'],
    'C': [0.1, 0.3, 1.0, 3.0, 10.0],
    'class_weight': [None, 'balanced'],
    # solver is determined by penalty:
    # L2 → lbfgs (recommended)
    # L1 → saga (required)
}

# Solver mapping based on penalty (implementation constraint per §5.2)
SOLVER_BY_PENALTY: dict[str, str] = {
    'l2': 'lbfgs',
    'l1': 'saga',
}

# =============================================================================
# Calibration Config (§4.3 - Frozen)
# =============================================================================

CALIBRATION_CONFIG: dict = {
    'method': 'isotonic',
    'cv': 5,
    'random_state': 42,  # Deterministic splitter: StratifiedKFold(shuffle=True)
}

# =============================================================================
# Fixed Parameters
# =============================================================================

FIXED_PARAMS: dict = {
    'random_state': 42,
    'max_iter': 1000,  # Standard max_iter
    'max_iter_retry': 5000,  # Retry value if ConvergenceWarning (§4.2)
    'n_jobs': -1,
}

# Re-export stopwords for backward compatibility
__all__ = [
    'TFIDF_GRID',
    'LOGREG_GRID',
    'SOLVER_BY_PENALTY',
    'CALIBRATION_CONFIG',
    'FIXED_PARAMS',
    # Re-exported from common.stopwords
    'STOPWORDS_SAFE',
    'STOPWORDS_NEVER_REMOVE',
    'STOPWORD_PRESETS',
    'resolve_stop_words',
]
