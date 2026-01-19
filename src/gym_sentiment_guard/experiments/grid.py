"""
Parameter grid definitions for LogReg ablation studies.

Defines the ablation surface per EXPERIMENT_PROTOCOL.md §5.
Only parameters listed here may vary during experiments.
"""

from __future__ import annotations

from pathlib import Path

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
# Curated Stopword List (§6.2 - Loaded from config)
# =============================================================================

_CONFIG_DIR = Path(__file__).resolve().parents[3] / 'configs'


def _load_stopwords(filename: str) -> list[str]:
    """Load stopwords from a text file (one word per line, # comments)."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return []
    words = []
    with filepath.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                words.append(line)
    return words


# These are non-sentiment-bearing function words safe to remove for TF-IDF.
# Per §6.2: curated, sentiment-safe Spanish stopwords.
STOPWORDS_SAFE: list[str] = _load_stopwords('stopwords_spanish_safe.txt')

# Tokens that must NOT be removed (§6.3 - diagnostic guardrail)
# These directly affect polarity or polarity modulation.
STOPWORDS_NEVER_REMOVE: list[str] = _load_stopwords('stopwords_never_remove_spanish.txt')

# =============================================================================
# Stopword Preset Registry (extensible)
# =============================================================================

STOPWORD_PRESETS: dict[str | None, list[str] | str | None] = {
    'curated_safe': STOPWORDS_SAFE,
    'english': 'english',  # sklearn built-in
    None: None,  # Disabled (legacy behavior)
}


def resolve_stop_words(config_value: str | list | None) -> list[str] | str | None:
    """
    Resolve stop_words config value to TfidfVectorizer-compatible format.

    Args:
        config_value: Value from config file. Can be:
            - 'curated_safe': Uses STOPWORDS_SAFE list
            - 'english': Uses sklearn's built-in English stopwords
            - None/missing: No stopwords (legacy behavior)
            - list: Custom stopwords list

    Returns:
        Value suitable for TfidfVectorizer(stop_words=...)

    Raises:
        ValueError: If config_value is not a valid preset or list
    """
    if config_value in STOPWORD_PRESETS:
        return STOPWORD_PRESETS[config_value]
    if isinstance(config_value, list):
        return config_value
    raise ValueError(
        f'Unknown stop_words preset: {config_value!r}. '
        f'Valid presets: {list(STOPWORD_PRESETS.keys())} or custom list.'
    )
