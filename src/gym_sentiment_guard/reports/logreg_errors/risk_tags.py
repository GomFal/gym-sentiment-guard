"""
Risk tag computation for error analysis.

TASK 4: Assign rule-based risk indicators.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils.logging import get_logger, json_log

log = get_logger(__name__)


def load_contrast_keywords(keywords_path: Path) -> set[str]:
    """
    Load contrast keywords from text file.

    Args:
        keywords_path: Path to contrast_keywords.txt

    Returns:
        Set of lowercase keywords
    """
    keywords_path = Path(keywords_path)
    if not keywords_path.exists():
        log.warning(json_log('risk_tags.keywords_not_found', path=str(keywords_path)))
        return set()

    keywords = set()
    with keywords_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                keywords.add(line.lower())

    log.info(
        json_log('risk_tags.keywords_loaded', component='error_analysis', n_keywords=len(keywords))
    )

    return keywords


def has_contrast_marker(text: str, keywords: set[str]) -> bool:
    """
    Check if text contains any contrast keyword.

    Uses simple lowercase substring matching (no NLP parsing).

    Args:
        text: Input text
        keywords: Set of lowercase contrast keywords

    Returns:
        True if any keyword found in text
    """
    text_lower = text.lower()
    # Use any() with generator for short-circuit evaluation
    return any(kw in text_lower for kw in keywords)


def compute_risk_tags(
    df: pd.DataFrame,
    threshold: float,
    near_threshold_band: float,
    contrast_keywords: set[str],
) -> pd.DataFrame:
    """
    Add risk tag columns to error table.

    Args:
        df: Error table with p_neg, low_coverage columns
        threshold: Decision threshold
        near_threshold_band: Half-width of uncertainty band
        contrast_keywords: Set of contrast markers

    Returns:
        DataFrame with added columns:
        - near_threshold, has_contrast (low_coverage already present)
    """
    result = df.copy()

    # Near-threshold: within uncertainty band
    p_neg = result['p_neg'].values
    lower_bound = threshold - near_threshold_band
    upper_bound = threshold + near_threshold_band
    near_threshold = (p_neg >= lower_bound) & (p_neg <= upper_bound)
    result['near_threshold'] = near_threshold

    # Has contrast markers
    # Use vectorized apply for efficiency
    texts = result['text'].astype(str)
    result['has_contrast'] = texts.apply(lambda t: has_contrast_marker(t, contrast_keywords))

    # Log summary
    n_near = near_threshold.sum()
    n_contrast = result['has_contrast'].sum()
    n_low_coverage = result['low_coverage'].sum()

    log.info(
        json_log(
            'risk_tags.computed',
            component='error_analysis',
            n_near_threshold=int(n_near),
            n_has_contrast=int(n_contrast),
            n_low_coverage=int(n_low_coverage),
        )
    )

    return result
