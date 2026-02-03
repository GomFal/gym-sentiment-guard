"""
Unified SVM pipeline builders.

Single source of truth for pipeline construction used by both experiments
and training modules. Ensures consistency across the entire SVM workflow.

Architecture:
- build_feature_union(): TF-IDF FeatureUnion (unigrams + multigrams)
- build_linear_pipeline(): LinearSVC + Calibration
- build_rbf_pipeline(): SVC(kernel='rbf') + Calibration
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC, LinearSVC

from gym_sentiment_guard.common.stopwords import resolve_stop_words


def build_feature_union(
    unigram_ngram_range: tuple[int, int] = (1, 1),
    unigram_min_df: int = 10,
    unigram_max_df: float = 0.90,
    unigram_sublinear_tf: bool = True,
    unigram_stop_words: list[str] | str | None = 'curated_safe',
    multigram_ngram_range: tuple[int, int] = (2, 3),
    multigram_min_df: int = 2,
    multigram_max_df: float = 0.90,
    multigram_sublinear_tf: bool = True,
    multigram_stop_words: list[str] | str | None = None,
) -> FeatureUnion:
    """
    Build TF-IDF FeatureUnion combining unigrams and multigrams.

    Args:
        unigram_*: Unigram TF-IDF parameters
        multigram_*: Multigram TF-IDF parameters

    Returns:
        FeatureUnion with two TfidfVectorizers
    """
    unigram_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=unigram_ngram_range,
        min_df=unigram_min_df,
        max_df=unigram_max_df,
        sublinear_tf=unigram_sublinear_tf,
        stop_words=resolve_stop_words(unigram_stop_words),
    )

    multigram_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=multigram_ngram_range,
        min_df=multigram_min_df,
        max_df=multigram_max_df,
        sublinear_tf=multigram_sublinear_tf,
        stop_words=resolve_stop_words(multigram_stop_words),
    )

    return FeatureUnion([
        ('unigrams', unigram_vectorizer),
        ('multigrams', multigram_vectorizer),
    ])


def build_linear_pipeline(
    # Vectorizer parameters
    unigram_ngram_range: tuple[int, int] = (1, 1),
    unigram_min_df: int = 10,
    unigram_max_df: float = 0.90,
    unigram_sublinear_tf: bool = True,
    unigram_stop_words: list[str] | str | None = 'curated_safe',
    multigram_ngram_range: tuple[int, int] = (2, 3),
    multigram_min_df: int = 2,
    multigram_max_df: float = 0.90,
    multigram_sublinear_tf: bool = True,
    multigram_stop_words: list[str] | str | None = None,
    # LinearSVC parameters
    penalty: str = 'l2',
    loss: str = 'squared_hinge',
    C: float = 1.0,
    dual: bool = True,
    fit_intercept: bool = True,
    intercept_scaling: float = 1.0,
    tol: float = 0.0001,
    max_iter: int = 2000,
    random_state: int = 42,
    # Calibration parameters
    calibration_method: str = 'isotonic',
    calibration_cv: int = 5,
    # Extra kwargs (ignored, allows passing full config dicts)
    **kwargs,
) -> Pipeline:
    """
    Build LinearSVC pipeline with FeatureUnion and calibration.

    Architecture (per EXPERIMENT_PROTOCOL.md §4):
    - FeatureUnion[TfidfVectorizer(unigrams), TfidfVectorizer(multigrams)]
    - LinearSVC
    - CalibratedClassifierCV with isotonic, 5-fold

    Args:
        unigram_*: Unigram TF-IDF parameters
        multigram_*: Multigram TF-IDF parameters
        penalty, loss, C, etc.: LinearSVC parameters
        calibration_*: Calibration parameters
        **kwargs: Ignored (allows passing full config dicts)

    Returns:
        Fitted-ready sklearn Pipeline
    """
    feature_union = build_feature_union(
        unigram_ngram_range=unigram_ngram_range,
        unigram_min_df=unigram_min_df,
        unigram_max_df=unigram_max_df,
        unigram_sublinear_tf=unigram_sublinear_tf,
        unigram_stop_words=unigram_stop_words,
        multigram_ngram_range=multigram_ngram_range,
        multigram_min_df=multigram_min_df,
        multigram_max_df=multigram_max_df,
        multigram_sublinear_tf=multigram_sublinear_tf,
        multigram_stop_words=multigram_stop_words,
    )

    base_clf = LinearSVC(
        penalty=penalty,
        loss=loss,
        C=C,
        dual=dual,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )

    cv_splitter = StratifiedKFold(
        n_splits=calibration_cv,
        shuffle=True,
        random_state=random_state,
    )

    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method=calibration_method,
        cv=cv_splitter,
    )

    return Pipeline([
        ('features', feature_union),
        ('classifier', calibrated),
    ])


def build_rbf_pipeline(
    # Vectorizer parameters
    unigram_ngram_range: tuple[int, int] = (1, 1),
    unigram_min_df: int = 10,
    unigram_max_df: float = 0.90,
    unigram_sublinear_tf: bool = True,
    unigram_stop_words: list[str] | str | None = 'curated_safe',
    multigram_ngram_range: tuple[int, int] = (2, 3),
    multigram_min_df: int = 2,
    multigram_max_df: float = 0.90,
    multigram_sublinear_tf: bool = True,
    multigram_stop_words: list[str] | str | None = None,
    # SVC RBF parameters
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str | float = 'scale',
    tol: float = 0.001,
    cache_size: float = 1000,
    max_iter: int = -1,
    random_state: int = 42,
    # Calibration parameters
    calibration_method: str = 'isotonic',
    calibration_cv: int = 5,
    # Scaling parameters
    use_scaler: bool = True,
    # Extra kwargs (ignored, allows passing full config dicts)
    **kwargs,
) -> Pipeline:
    """
    Build SVC RBF pipeline with FeatureUnion, scaling, and calibration.

    Architecture:
    - FeatureUnion[TfidfVectorizer(unigrams), TfidfVectorizer(multigrams)]
    - MaxAbsScaler (preserves sparsity, normalizes features for RBF)
    - SVC(kernel='rbf', probability=False)
    - CalibratedClassifierCV with isotonic, 5-fold

    Args:
        unigram_*: Unigram TF-IDF parameters
        multigram_*: Multigram TF-IDF parameters
        kernel, C, gamma, etc.: SVC parameters
        calibration_*: Calibration parameters
        use_scaler: Whether to apply MaxAbsScaler before classifier (default: True).
                   Recommended for RBF kernels to normalize feature magnitudes.
        **kwargs: Ignored (allows passing full config dicts)

    Returns:
        Fitted-ready sklearn Pipeline
    """
    feature_union = build_feature_union(
        unigram_ngram_range=unigram_ngram_range,
        unigram_min_df=unigram_min_df,
        unigram_max_df=unigram_max_df,
        unigram_sublinear_tf=unigram_sublinear_tf,
        unigram_stop_words=unigram_stop_words,
        multigram_ngram_range=multigram_ngram_range,
        multigram_min_df=multigram_min_df,
        multigram_max_df=multigram_max_df,
        multigram_sublinear_tf=multigram_sublinear_tf,
        multigram_stop_words=multigram_stop_words,
    )

    # probability=False because we use CalibratedClassifierCV for better calibration
    base_clf = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        tol=tol,
        cache_size=cache_size,
        max_iter=max_iter,
        random_state=random_state,
        probability=False,
    )

    cv_splitter = StratifiedKFold(
        n_splits=calibration_cv,
        shuffle=True,
        random_state=random_state,
    )

    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method=calibration_method,
        cv=cv_splitter,
    )

    # Build pipeline steps
    steps = [('features', feature_union)]

    # Add scaler for RBF (preserves sparsity)
    if use_scaler:
        steps.append(('scaler', MaxAbsScaler()))

    steps.append(('classifier', calibrated))

    return Pipeline(steps)
