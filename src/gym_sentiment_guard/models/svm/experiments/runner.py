"""
Single experiment runner for SVM Linear model.

Implements the core experiment execution logic:
1. Load frozen splits
2. Build pipeline (FeatureUnion[TF-IDF_unigram, TF-IDF_multigram] → LinearSVC → Isotonic)
3. Fit on TRAIN only
4. Get probabilities on VAL
5. Select threshold (§3)
6. Compute metrics (§7)
7. Check validity (§8)
8. Persist artifacts (§10)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.common.artifacts import (
    RunConfig,
    RunResult,
    generate_run_id,
    get_environment_info,
    get_git_info,
    save_predictions,
    save_run_artifact,
)
from gym_sentiment_guard.common.metrics import compute_val_metrics
from gym_sentiment_guard.common.threshold import apply_threshold, select_threshold
from gym_sentiment_guard.models.svm.pipelines import build_linear_pipeline
from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


@dataclass
class SVMExperimentConfig:
    """Configuration for a single SVM experiment run."""

    # Data paths
    train_path: str
    val_path: str
    test_path: str

    # Column names
    text_column: str = 'comment'
    label_column: str = 'sentiment'
    label_mapping: dict[str, int] = field(default_factory=lambda: {'negative': 0, 'positive': 1})

    # FeatureUnion: Unigram TF-IDF parameters
    unigram_ngram_range: tuple[int, int] = (1, 1)
    unigram_min_df: int = 10
    unigram_max_df: float = 0.90
    unigram_sublinear_tf: bool = True
    unigram_stop_words: list[str] | str | None = 'curated_safe'

    # FeatureUnion: Multigram TF-IDF parameters
    multigram_ngram_range: tuple[int, int] = (2, 3)
    multigram_min_df: int = 2
    multigram_max_df: float = 0.90
    multigram_sublinear_tf: bool = True
    multigram_stop_words: list[str] | str | None = None

    # SVM Linear parameters (ablatable)
    penalty: str = 'l2'
    loss: str = 'squared_hinge'
    C: float = 1.0
    dual: bool = True
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    tol: float = 1e-4
    max_iter: int = 2000

    # Calibration parameters (from config)
    calibration_method: str = 'isotonic'
    calibration_cv: int = 5

    # Fixed parameters (from config)
    random_state: int = 42
    n_jobs: int = -1

    # Selection constraint (§1.1)
    recall_constraint: float = 0.90

    # Output
    output_dir: str = 'artifacts/experiments/svm_linear'
    save_predictions: bool = True





def _extract_diagnostics(pipeline: Pipeline) -> dict[str, Any]:
    """Extract diagnostic info from fitted pipeline."""
    features = pipeline.named_steps['features']

    # Count features from both vectorizers in FeatureUnion
    unigram_vectorizer = features.transformer_list[0][1]
    multigram_vectorizer = features.transformer_list[1][1]

    n_unigram = len(unigram_vectorizer.vocabulary_)
    n_multigram = len(multigram_vectorizer.vocabulary_)
    n_features = n_unigram + n_multigram

    # SVM coefficient sparsity from calibrated estimators
    calibrated = pipeline.named_steps['classifier']
    sparsity_values = []
    for estimator in calibrated.calibrated_classifiers_:
        base = estimator.estimator
        if hasattr(base, 'coef_'):
            coef = base.coef_.ravel()
            zero_pct = np.mean(coef == 0) * 100
            sparsity_values.append(zero_pct)

    avg_sparsity = float(np.mean(sparsity_values)) if sparsity_values else 0.0

    return {
        'n_features': n_features,
        'n_unigram_features': n_unigram,
        'n_multigram_features': n_multigram,
        'coefficient_sparsity': avg_sparsity,
    }


def run_single_experiment(
    config: SVMExperimentConfig,
) -> RunResult:
    """
    Run a single SVM experiment following the protocol.

    Args:
        config: SVM experiment configuration

    Returns:
        RunResult with metrics, validity status, and artifacts
    """
    start_time = time.perf_counter()

    # Generate run ID
    output_dir = Path(config.output_dir)
    run_id = generate_run_id(output_dir, prefix='run')
    run_dir = output_dir / run_id
    timestamp = datetime.now(UTC).isoformat()

    log.info(
        json_log(
            'experiment.start',
            component='svm_experiments',
            run_id=run_id,
            C=config.C,
            intercept_scaling=config.intercept_scaling,
            tol=config.tol,
            max_iter=config.max_iter,
        )
    )

    # Build run config
    git_info = get_git_info()
    env_info = get_environment_info()

    run_config = RunConfig(
        run_id=run_id,
        timestamp=timestamp,
        git_commit=git_info.get('git_commit'),
        git_branch=git_info.get('git_branch'),
        git_dirty=git_info.get('git_dirty'),
        python_version=env_info['python_version'],
        sklearn_version=env_info['sklearn_version'],
        train_path=config.train_path,
        val_path=config.val_path,
        test_path=config.test_path,
        tfidf_params={
            'unigram_ngram_range': list(config.unigram_ngram_range),
            'unigram_min_df': config.unigram_min_df,
            'unigram_max_df': config.unigram_max_df,
            'unigram_sublinear_tf': config.unigram_sublinear_tf,
            'unigram_stop_words': config.unigram_stop_words,
            'multigram_ngram_range': list(config.multigram_ngram_range),
            'multigram_min_df': config.multigram_min_df,
            'multigram_max_df': config.multigram_max_df,
            'multigram_sublinear_tf': config.multigram_sublinear_tf,
            'multigram_stop_words': config.multigram_stop_words,
        },
        classifier_params={
            'penalty': config.penalty,
            'loss': config.loss,
            'C': config.C,
            'dual': config.dual,
            'fit_intercept': config.fit_intercept,
            'intercept_scaling': config.intercept_scaling,
            'tol': config.tol,
            'max_iter': config.max_iter,
            'random_state': config.random_state,
        },
        classifier_type='svm_linear',
        calibration_config={
            'method': config.calibration_method,
            'cv': config.calibration_cv,
            'random_state': config.random_state,
        },
    )

    # Load data
    try:
        train_df = pd.read_csv(config.train_path)
        val_df = pd.read_csv(config.val_path)
    except Exception as e:
        log.error(json_log('experiment.data_load_failed', run_id=run_id, error=str(e)))
        return RunResult(
            config=run_config,
            validity_status='invalid',
            invalidity_reason=f'Data load failed: {e}',
        )

    X_train = train_df[config.text_column].astype(str)

    # Validate train labels mapping
    y_train_mapped = train_df[config.label_column].map(config.label_mapping)
    unmapped_train = train_df.loc[y_train_mapped.isnull(), config.label_column].unique()
    if len(unmapped_train) > 0:
        raise ValueError(
            f"Unmapped labels found in TRAIN data: {unmapped_train.tolist()}. "
            f"Source: {config.train_path}, column: {config.label_column}"
        )
    y_train = y_train_mapped.astype(int)

    X_val = val_df[config.text_column].astype(str)

    # Validate val labels mapping
    y_val_mapped = val_df[config.label_column].map(config.label_mapping)
    unmapped_val = val_df.loc[y_val_mapped.isnull(), config.label_column].unique()
    if len(unmapped_val) > 0:
        raise ValueError(
            f"Unmapped labels found in VAL data: {unmapped_val.tolist()}. "
            f"Source: {config.val_path}, column: {config.label_column}"
        )
    y_val = y_val_mapped.astype(int)

    # Build and train pipeline
    # Build pipeline using shared builder
    pipeline = build_linear_pipeline(
        unigram_ngram_range=config.unigram_ngram_range,
        unigram_min_df=config.unigram_min_df,
        unigram_max_df=config.unigram_max_df,
        unigram_sublinear_tf=config.unigram_sublinear_tf,
        unigram_stop_words=config.unigram_stop_words,
        multigram_ngram_range=config.multigram_ngram_range,
        multigram_min_df=config.multigram_min_df,
        multigram_max_df=config.multigram_max_df,
        multigram_sublinear_tf=config.multigram_sublinear_tf,
        multigram_stop_words=config.multigram_stop_words,
        penalty=config.penalty,
        loss=config.loss,
        C=config.C,
        dual=config.dual,
        fit_intercept=config.fit_intercept,
        intercept_scaling=config.intercept_scaling,
        tol=config.tol,
        max_iter=config.max_iter,
        random_state=config.random_state,
        calibration_method=config.calibration_method,
        calibration_cv=config.calibration_cv,
    )

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            pipeline.fit(X_train, y_train)

            # Check for convergence warnings
            convergence_warnings = [
                warning for warning in w if issubclass(warning.category, ConvergenceWarning)
            ]

            if convergence_warnings:
                log.warning(
                    json_log(
                        'experiment.convergence_warning',
                        run_id=run_id,
                        message='LinearSVC did not converge',
                    )
                )

    except Exception as e:
        log.error(json_log('experiment.training_failed', run_id=run_id, error=str(e)))
        return RunResult(
            config=run_config,
            validity_status='invalid',
            invalidity_reason=f'Training failed: {e}',
        )

    training_time = time.perf_counter() - start_time

    # Get predictions on VAL
    try:
        proba = pipeline.predict_proba(X_val)
        # Classes are [0, 1] where 0=negative, 1=positive
        class_idx = {c: i for i, c in enumerate(pipeline.classes_)}
        p_neg = proba[:, class_idx[0]]  # P(negative)
        p_pos = proba[:, class_idx[1]]  # P(positive)
    except Exception as e:
        log.error(json_log('experiment.prediction_failed', run_id=run_id, error=str(e)))
        return RunResult(
            config=run_config,
            validity_status='invalid',
            invalidity_reason=f'Prediction failed: {e}',
        )

    # Select threshold (§3)
    threshold_result = select_threshold(
        y_true=y_val.values,
        p_neg=p_neg,
        recall_constraint=config.recall_constraint,
    )

    # Apply threshold to get predictions
    y_pred = apply_threshold(p_neg, threshold_result.threshold)

    # Compute metrics (§7)
    val_metrics = compute_val_metrics(
        y_true=y_val.values,
        p_neg=p_neg,
        y_pred=y_pred,
        threshold=threshold_result.threshold,
        constraint_status=threshold_result.constraint_status,
    )

    # Extract diagnostics
    diagnostics = _extract_diagnostics(pipeline)

    # Determine validity status
    if threshold_result.constraint_status == 'not_met':
        validity_status = 'constraint_not_met'
    else:
        validity_status = 'valid'

    # Build result
    result = RunResult(
        config=run_config,
        val_metrics=val_metrics,
        validity_status=validity_status,
        training_time_seconds=training_time,
        n_features=diagnostics['n_features'],
        coefficient_sparsity=diagnostics['coefficient_sparsity'],
    )

    # Save artifacts (§10)
    save_run_artifact(result, run_dir)

    # Save predictions if requested
    if config.save_predictions:
        save_predictions(
            ids=None,
            y_true=y_val.values.tolist(),
            p_pos=p_pos.tolist(),
            p_neg=p_neg.tolist(),
            output_dir=run_dir,
            filename='val_predictions.csv',
        )

    log.info(
        json_log(
            'experiment.completed',
            component='svm_experiments',
            run_id=run_id,
            f1_neg=val_metrics.f1_neg,
            recall_neg=val_metrics.recall_neg,
            threshold=val_metrics.threshold,
            constraint_status=val_metrics.constraint_status,
            validity_status=validity_status,
            training_time_seconds=round(training_time, 2),
            n_features=diagnostics['n_features'],
        )
    )

    return result
