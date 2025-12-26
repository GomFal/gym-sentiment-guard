"""Data processing utilities for gym_sentiment_guard."""

from .cleaning import (
    deduplicate_reviews,
    drop_columns,
    drop_neutral_ratings,
    enforce_expectations,
    load_structural_punctuation,
    normalize_comments,
)
from .merge import merge_processed_csvs
from .split import split_dataset

# NOTE: Language functions (filter_spanish_comments, configure_language_fallback)
# are NOT exported here to avoid pulling in heavy dependencies (tenacity, fasttext).
# Import them directly when needed:
#   from gym_sentiment_guard.data.language import filter_spanish_comments

__all__ = [
    "deduplicate_reviews",
    "drop_columns",
    "drop_neutral_ratings",
    "enforce_expectations",
    "load_structural_punctuation",
    "normalize_comments",
    "merge_processed_csvs",
    "split_dataset",
]


