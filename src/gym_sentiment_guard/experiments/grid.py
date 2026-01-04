"""
Parameter grid definitions for LogReg ablation studies.

Defines the ablation surface per EXPERIMENT_PROTOCOL.md §5.
Only parameters listed here may vary during experiments.
"""

from __future__ import annotations

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

# =============================================================================
# Curated Stopword List (§6.2 - Frozen)
# =============================================================================

# These are non-sentiment-bearing function words safe to remove for TF-IDF.
# Per §6.2: curated, sentiment-safe Spanish stopwords.

STOPWORDS_SAFE: list[str] = [
    'de',
    'el',
    'que',
    'la',
    'en',
    'es',
    'con',
    'las',
    'un',
    'los',
    'me',
    'por',
    'para',
    'lo',
    'una',
    'se',
    'del',
    'al',
    'son',
    'hay',
    'este',
    'te',
    'como',
    'he',
    'mi',
    'ha',
    'tiene',
    'su',
    'todos',
    'cuando',
    'hace',
    'esta',
    'han',
    'desde',
    'parte',
    'día',
    'cada',
    'ir',
    'así',
    'años',
    'llevo',
    'nos',
    'le',
    'vez',
    'todo',
    'muchas',
]

# Tokens that must NOT be removed (§6.3 - diagnostic guardrail)
# These directly affect polarity or polarity modulation.
STOPWORDS_NEVER_REMOVE: list[str] = [
    'no',
    'ni',
    'nada',
    'sin',
    'pero',
    'muy',
    'más',
    'poco',
    'solo',
    'bastante',
    'súper',
    'siempre',
    'además',
]
