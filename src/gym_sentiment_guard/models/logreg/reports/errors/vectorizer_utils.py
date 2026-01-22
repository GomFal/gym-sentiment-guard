"""
Vectorizer utilities for error analysis.

Abstracts vectorizer type detection to support both single TfidfVectorizer
and FeatureUnion configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_vectorizer_from_pipeline(model: Pipeline) -> TfidfVectorizer | FeatureUnion:
    """
    Get the vectorizer step from pipeline.

    Supports both 'tfidf' and 'vectorizer' step names for compatibility.

    Args:
        model: Fitted sklearn Pipeline

    Returns:
        The vectorizer component (TfidfVectorizer or FeatureUnion)

    Raises:
        ValueError: If no vectorizer found in pipeline
    """
    for name in ['tfidf', 'vectorizer']:
        if name in model.named_steps:
            return model.named_steps[name]
    raise ValueError('No vectorizer found in pipeline (expected "tfidf" or "vectorizer" step)')


def is_feature_union(vectorizer: TfidfVectorizer | FeatureUnion) -> bool:
    """Check if vectorizer is a FeatureUnion."""
    return isinstance(vectorizer, FeatureUnion)


def get_feature_names(vectorizer: TfidfVectorizer | FeatureUnion) -> NDArray[np.str_]:
    """
    Get feature names from TfidfVectorizer or FeatureUnion.

    For FeatureUnion, returns prefixed feature names (e.g., 'unigrams__word').

    Args:
        vectorizer: Fitted TfidfVectorizer or FeatureUnion

    Returns:
        Array of feature name strings
    """
    # get_feature_names_out() works for both TfidfVectorizer and FeatureUnion
    return vectorizer.get_feature_names_out()


def get_vocabulary_size(vectorizer: TfidfVectorizer | FeatureUnion) -> int:
    """
    Get total vocabulary size.

    Args:
        vectorizer: Fitted TfidfVectorizer or FeatureUnion

    Returns:
        Total number of features
    """
    if is_feature_union(vectorizer):
        return sum(len(v.vocabulary_) for _, v in vectorizer.transformer_list)
    else:
        return len(vectorizer.vocabulary_)
