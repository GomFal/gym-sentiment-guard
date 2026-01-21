"""Stopwords handling utilities.

Provides stopword loading, preset registry, and resolution for TF-IDF vectorizers.
These are language-specific but model-agnostic utilities.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Stopword Loading
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


# =============================================================================
# Curated Stopword Lists (§6.2 - Loaded from config)
# =============================================================================

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
