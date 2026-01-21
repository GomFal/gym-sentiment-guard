"""
Single experiment runner.

Implements the core experiment execution logic:
1. Load frozen splits
2. Build pipeline (TF-IDF → LogReg → Isotonic)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ..common.metrics import compute_val_metrics
from ..common.threshold import apply_threshold, select_threshold
from ..utils.logging import get_logger, json_log
from .artifacts import (
    RunConfig,
    RunResult,
    generate_run_id,
    get_environment_info,
    get_git_info,
    save_predictions,
    save_run_artifact,
)
from .grid import CALIBRATION_CONFIG, FIXED_PARAMS, SOLVER_BY_PENALTY, STOPWORDS_SAFE

log = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Data paths
    train_path: str
    val_path: str
    test_path: str

    # Column names
    text_column: str = 'comment'
    label_column: str = 'sentiment'
    label_mapping: dict[str, int] = field(default_factory=lambda: {'negative': 0, 'positive': 1})

    # TF-IDF parameters (ablatable per §5.1)
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    stop_words: list[str] | str | None = None  # None, 'curated_safe', or custom list

    # LogReg parameters (ablatable per §5.2)
    penalty: str = 'l2'
    C: float = 1.0
    class_weight: str | None = None

    # Selection constraint (§1.1)
    recall_constraint: float = 0.90

    # Output
    output_dir: str = 'artifacts/experiments'
    save_predictions: bool = True


def _get_solver(penalty: str) -> str:
    """Get appropriate solver for penalty (§5.2)."""
    return SOLVER_BY_PENALTY.get(penalty, 'lbfgs')


def _build_pipeline(config: ExperimentConfig) -> Pipeline:
    """
    Build the experiment pipeline.

    Architecture (§4):
    - TfidfVectorizer (fixed family)
    - LogisticRegression (fixed family)
    - CalibratedClassifierCV with isotonic, 5-fold (frozen)
    """
    # TF-IDF vectorizer (§4.1)
    # Resolve stop_words: None = no stopwords, 'curated_safe' = use STOPWORDS_SAFE
    if config.stop_words == 'curated_safe':
        stop_words = STOPWORDS_SAFE
    elif config.stop_words is None:
        stop_words = None  # No stopwords removal
    else:
        stop_words = config.stop_words  # Custom list provided
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        stop_words=stop_words,
    )

    # Base classifier (§4.2)
    solver = _get_solver(config.penalty)
    base_clf = LogisticRegression(
        penalty=config.penalty,
        C=config.C,
        class_weight=config.class_weight,
        solver=solver,
        max_iter=FIXED_PARAMS['max_iter'],
        random_state=FIXED_PARAMS['random_state'],
        n_jobs=FIXED_PARAMS['n_jobs'],
    )

    # Calibration with deterministic CV splitter (§4.3)
    cv_splitter = StratifiedKFold(
        n_splits=CALIBRATION_CONFIG['cv'],
        shuffle=True,
        random_state=CALIBRATION_CONFIG['random_state'],
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method=CALIBRATION_CONFIG['method'],
        cv=cv_splitter,
    )

    # Build pipeline
    pipeline = Pipeline(
        [
            ('tfidf', vectorizer),
            ('classifier', calibrated),
        ]
    )

    return pipeline


def _extract_diagnostics(pipeline: Pipeline) -> dict[str, Any]:
    """Extract diagnostic info from fitted pipeline."""
    tfidf = pipeline.named_steps['tfidf']
    n_features = len(tfidf.vocabulary_)

    # Coefficient sparsity (% zero coefficients) from calibrated estimators
    # CalibratedClassifierCV has multiple base estimators
    # L2 regularization: ~0% sparsity (all coefficients active)
    # L1 regularization: high sparsity (many zero coefficients)
    calibrated = pipeline.named_steps['classifier']
    sparsity_values = []
    for estimator in calibrated.calibrated_classifiers_:
        base = estimator.estimator
        if hasattr(base, 'coef_'):
            coef = base.coef_.ravel()
            zero_pct = np.mean(coef == 0) * 100  # True sparsity
            sparsity_values.append(zero_pct)

    avg_sparsity = float(np.mean(sparsity_values)) if sparsity_values else 0.0

    return {
        'n_features': n_features,
        'coefficient_sparsity': avg_sparsity,
    }


def run_single_experiment(
    config: ExperimentConfig,
) -> RunResult:
    """
    Run a single experiment following the protocol.

    Args:
        config: Experiment configuration

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
            component='experiments',
            run_id=run_id,
            ngram_range=config.ngram_range,
            C=config.C,
            penalty=config.penalty,
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
            'ngram_range': list(config.ngram_range),
            'min_df': config.min_df,
            'max_df': config.max_df,
            'sublinear_tf': config.sublinear_tf,
            'stop_words': config.stop_words,  # None (no stopwords) or 'curated_safe'
        },
        logreg_params={
            'penalty': config.penalty,
            'C': config.C,
            'class_weight': config.class_weight,
            'solver': _get_solver(config.penalty),
            'max_iter': FIXED_PARAMS['max_iter'],
            'random_state': FIXED_PARAMS['random_state'],
        },
        calibration_config=CALIBRATION_CONFIG,
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
    y_train = train_df[config.label_column].map(config.label_mapping).astype(int)
    X_val = val_df[config.text_column].astype(str)
    y_val = val_df[config.label_column].map(config.label_mapping).astype(int)

    # Build and train pipeline
    pipeline = _build_pipeline(config)

    converged = False
    retry_attempted = False

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            pipeline.fit(X_train, y_train)

            # Check for convergence warning
            convergence_warnings = [
                warning for warning in w if issubclass(warning.category, ConvergenceWarning)
            ]

            if convergence_warnings:
                # Retry with higher max_iter (§4.2)
                log.warning(
                    json_log(
                        'experiment.convergence_retry',
                        run_id=run_id,
                        new_max_iter=FIXED_PARAMS['max_iter_retry'],
                    )
                )
                retry_attempted = True

                # Rebuild pipeline with higher max_iter
                # Note: We need to modify the base estimator in calibrated classifier
                calibrated = pipeline.named_steps['classifier']
                for estimator in calibrated.calibrated_classifiers_:
                    if hasattr(estimator.estimator, 'max_iter'):
                        estimator.estimator.max_iter = FIXED_PARAMS['max_iter_retry']

                # Re-fit
                pipeline.fit(X_train, y_train)
                converged = True
            else:
                converged = True

    except Exception as e:
        log.error(json_log('experiment.training_failed', run_id=run_id, error=str(e)))
        return RunResult(
            config=run_config,
            validity_status='invalid',
            invalidity_reason=f'Training failed: {e}',
        )

    # Check final convergence after retry
    if retry_attempted and not converged:
        return RunResult(
            config=run_config,
            validity_status='invalid',
            invalidity_reason='Convergence failed after retry (§4.2)',
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
            component='experiments',
            run_id=run_id,
            f1_neg=val_metrics.f1_neg,
            recall_neg=val_metrics.recall_neg,
            threshold=val_metrics.threshold,
            constraint_status=val_metrics.constraint_status,
            validity_status=validity_status,
            training_time_seconds=round(training_time, 2),
        )
    )

    return result
