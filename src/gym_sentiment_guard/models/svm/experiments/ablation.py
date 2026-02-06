"""
Ablation suite orchestrator for SVM Linear model.

Implements §9 of EXPERIMENT_PROTOCOL.md:
- Compare configurations using VAL only
- Change one conceptual factor at a time
- Select winner based on primary objective + tie-breakers
- Evaluate winner on TEST once
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from gym_sentiment_guard.common.artifacts import RunResult
from gym_sentiment_guard.common.metrics import compute_test_metrics
from gym_sentiment_guard.common.threshold import apply_threshold
from gym_sentiment_guard.utils.logging import get_logger, json_log

from .runner import SVMExperimentConfig, run_single_experiment

log = get_logger(__name__)


def generate_grid_configs(
    base_config: SVMExperimentConfig,
    unigram_grid: dict[str, list],
    multigram_grid: dict[str, list],
    svm_grid: dict[str, list],
) -> list[SVMExperimentConfig]:
    """
    Generate all configurations from parameter grids.

    Args:
        base_config: Base experiment configuration
        unigram_grid: Unigram TF-IDF parameter grid
        multigram_grid: Multigram TF-IDF parameter grid
        svm_grid: SVM Linear parameter grid

    Returns:
        List of SVMExperimentConfig for each combination
    """
    configs = []

    # Build combined grid with prefixed keys
    combined_grid = {}

    # Unigram params (prefix with 'unigram_')
    for key, values in unigram_grid.items():
        combined_grid[f'unigram_{key}'] = values

    # Multigram params (prefix with 'multigram_')
    for key, values in multigram_grid.items():
        combined_grid[f'multigram_{key}'] = values

    # SVM params (direct mapping)
    combined_grid.update(svm_grid)

    # Generate all combinations
    keys = list(combined_grid.keys())
    values = [combined_grid[k] for k in keys]

    for combo in product(*values):
        params = dict(zip(keys, combo, strict=True))

        config = SVMExperimentConfig(
            # Data paths from base config
            train_path=base_config.train_path,
            val_path=base_config.val_path,
            test_path=base_config.test_path,
            text_column=base_config.text_column,
            label_column=base_config.label_column,
            label_mapping=base_config.label_mapping,
            recall_constraint=base_config.recall_constraint,
            output_dir=base_config.output_dir,
            save_predictions=base_config.save_predictions,
            # Calibration params from base
            calibration_method=base_config.calibration_method,
            calibration_cv=base_config.calibration_cv,
            random_state=base_config.random_state,
            n_jobs=base_config.n_jobs,
            # Unigram TF-IDF params
            unigram_ngram_range=params.get('unigram_ngram_range', base_config.unigram_ngram_range),
            unigram_min_df=params.get('unigram_min_df', base_config.unigram_min_df),
            unigram_max_df=params.get('unigram_max_df', base_config.unigram_max_df),
            unigram_sublinear_tf=params.get('unigram_sublinear_tf', base_config.unigram_sublinear_tf),
            unigram_stop_words=params.get('unigram_stop_words', base_config.unigram_stop_words),
            # Multigram TF-IDF params
            multigram_ngram_range=params.get(
                'multigram_ngram_range', base_config.multigram_ngram_range
            ),
            multigram_min_df=params.get('multigram_min_df', base_config.multigram_min_df),
            multigram_max_df=params.get('multigram_max_df', base_config.multigram_max_df),
            multigram_sublinear_tf=params.get(
                'multigram_sublinear_tf', base_config.multigram_sublinear_tf
            ),
            multigram_stop_words=params.get('multigram_stop_words', base_config.multigram_stop_words),
            # SVM params
            penalty=params.get('penalty', base_config.penalty),
            loss=params.get('loss', base_config.loss),
            C=params.get('C', base_config.C),
            dual=params.get('dual', base_config.dual),
            fit_intercept=params.get('fit_intercept', base_config.fit_intercept),
            intercept_scaling=params.get('intercept_scaling', base_config.intercept_scaling),
            tol=params.get('tol', base_config.tol),
            max_iter=params.get('max_iter', base_config.max_iter),
        )
        configs.append(config)

    return configs


def rank_results(results: list[RunResult]) -> list[RunResult]:
    """
    Rank results by primary objective and tie-breakers (§1).

    Primary: F1_neg (maximize) subject to Recall_neg >= 0.90
    Tie-breakers (in order):
    1. Macro F1 (higher is better)
    2. PR AUC (Negative) (higher is better)
    3. Brier Score (lower is better)
    4. ECE (lower is better)

    Valid runs with constraint met > constraint not met > invalid

    Args:
        results: List of run results

    Returns:
        Sorted list (best first)
    """

    def sort_key(r: RunResult) -> tuple:
        """
        Sort key: (validity, constraint, f1_neg, macro_f1, pr_auc, -brier, -ece)
        Higher is better, so we negate brier/ece.
        """
        if r.validity_status == 'invalid':
            return (0, 0, 0, 0, 0, 0, 0)

        if r.val_metrics is None:
            return (0, 0, 0, 0, 0, 0, 0)

        constraint_met = 1 if r.val_metrics.constraint_status == 'met' else 0

        return (
            1,  # valid
            constraint_met,
            r.val_metrics.f1_neg,
            r.val_metrics.macro_f1,
            r.val_metrics.pr_auc_neg,
            -r.val_metrics.brier_score,  # Lower is better
            -r.val_metrics.ece,  # Lower is better
        )

    return sorted(results, key=sort_key, reverse=True)


def run_ablation_suite(
    base_config: SVMExperimentConfig,
    unigram_grid: dict[str, list],
    multigram_grid: dict[str, list],
    svm_grid: dict[str, list],
    max_runs: int | None = None,
) -> dict[str, Any]:
    """
    Run full SVM ablation suite.

    Args:
        base_config: Base experiment configuration
        unigram_grid: Unigram TF-IDF parameter grid
        multigram_grid: Multigram TF-IDF parameter grid
        svm_grid: SVM Linear parameter grid
        max_runs: Maximum number of runs (for testing, None = all)

    Returns:
        Dictionary with:
        - results: All run results
        - winner: Best configuration
        - summary: Summary statistics
    """
    # Generate all configs
    configs = generate_grid_configs(base_config, unigram_grid, multigram_grid, svm_grid)

    if max_runs is not None:
        configs = configs[:max_runs]

    log.info(
        json_log(
            'ablation.start',
            component='svm_experiments',
            n_configs=len(configs),
        )
    )

    # Run all experiments
    results = []
    for i, config in enumerate(configs):
        log.info(
            json_log(
                'ablation.progress',
                component='svm_experiments',
                current=i + 1,
                total=len(configs),
            )
        )
        result = run_single_experiment(config)
        results.append(result)

    # Rank results
    ranked = rank_results(results)
    winner = ranked[0] if ranked else None

    # Compute summary statistics
    valid_count = sum(1 for r in results if r.validity_status == 'valid')
    constraint_not_met_count = sum(
        1
        for r in results
        if r.validity_status == 'constraint_not_met'
    )
    invalid_count = sum(1 for r in results if r.validity_status == 'invalid')

    # Build ranked runs list for summary (matching LogReg format)
    ranked_runs = []
    for r in ranked:
        if r.val_metrics is not None:
            ranked_runs.append({
                'run_id': r.config.run_id,
                'validity_status': r.validity_status,
                'f1_neg': r.val_metrics.f1_neg,
                'recall_neg': r.val_metrics.recall_neg,
                'macro_f1': r.val_metrics.macro_f1,
                'threshold': r.val_metrics.threshold,
                'constraint_status': r.val_metrics.constraint_status,
            })

    # Build comprehensive summary (matching LogReg format)
    summary = {
        'total_runs': len(results),
        'valid_runs': valid_count,
        'constraint_not_met_runs': constraint_not_met_count,
        'invalid_runs': invalid_count,
        'winner_run_id': winner.config.run_id if winner else None,
        'winner_f1_neg': winner.val_metrics.f1_neg if winner and winner.val_metrics else None,
        'winner_threshold': winner.val_metrics.threshold if winner and winner.val_metrics else None,
        'ranked_runs': ranked_runs,
    }

    # Save suite summary
    output_dir = Path(base_config.output_dir)
    suite_dir = output_dir / f"suite.{datetime.now(UTC).strftime('%Y.%m.%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary_path = suite_dir / 'suite_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if winner and winner.val_metrics:
        log.info(
            json_log(
                'ablation.winner',
                component='svm_experiments',
                run_id=winner.config.run_id,
                f1_neg=winner.val_metrics.f1_neg,
                recall_neg=winner.val_metrics.recall_neg,
            )
        )

    log.info(
        json_log(
            'ablation.complete',
            component='svm_experiments',
            total_runs=len(results),
            valid_runs=valid_count,
            constraint_not_met=constraint_not_met_count,
        )
    )

    return {
        'results': results,
        'ranked': ranked,
        'winner': winner,
        'summary': summary,
        'suite_dir': str(suite_dir),
    }


def evaluate_winner_on_test(
    winner: RunResult,
    test_path: str,
    text_column: str = 'comment',
    label_column: str = 'sentiment',
    label_mapping: dict[str, int] | None = None,
) -> Any:
    """
    Evaluate winner configuration on TEST set (§11).

    This should only be called ONCE after winner is locked.

    Args:
        winner: Winning run result
        test_path: Path to test CSV
        text_column: Text column name
        label_column: Label column name
        label_mapping: Label to int mapping

    Returns:
        TEST metrics
    """
    if label_mapping is None:
        label_mapping = {'negative': 0, 'positive': 1}

    if winner.val_metrics is None:
        raise ValueError('Winner has no validation metrics')

    # Load test data
    test_df = pd.read_csv(test_path)
    y_test = test_df[label_column].map(label_mapping).astype(int)

    # Reconstruct pipeline from winner config
    config_params = winner.config.classifier_params
    tfidf_params = winner.config.tfidf_params

    # Build config for pipeline reconstruction
    from gym_sentiment_guard.models.svm.pipelines import build_linear_pipeline

    recon_config = SVMExperimentConfig(
        train_path=winner.config.train_path,
        val_path=winner.config.val_path,
        test_path=test_path,
        text_column=text_column,
        label_column=label_column,
        label_mapping=label_mapping,
        # Unigram params
        unigram_ngram_range=tuple(tfidf_params.get('unigram_ngram_range', [1, 1])),
        unigram_min_df=tfidf_params.get('unigram_min_df', 10),
        unigram_max_df=tfidf_params.get('unigram_max_df', 0.90),
        unigram_sublinear_tf=tfidf_params.get('unigram_sublinear_tf', True),
        unigram_stop_words=tfidf_params.get('unigram_stop_words', 'curated_safe'),
        # Multigram params
        multigram_ngram_range=tuple(tfidf_params.get('multigram_ngram_range', [2, 3])),
        multigram_min_df=tfidf_params.get('multigram_min_df', 2),
        multigram_max_df=tfidf_params.get('multigram_max_df', 0.90),
        multigram_sublinear_tf=tfidf_params.get('multigram_sublinear_tf', True),
        multigram_stop_words=tfidf_params.get('multigram_stop_words'),
        # SVM params
        penalty=config_params.get('penalty', 'l2'),
        loss=config_params.get('loss', 'squared_hinge'),
        C=config_params.get('C', 1.0),
        dual=config_params.get('dual', True),
        fit_intercept=config_params.get('fit_intercept', True),
        intercept_scaling=config_params.get('intercept_scaling', 1.0),
        tol=config_params.get('tol', 1e-4),
        max_iter=config_params.get('max_iter', 2000),
        random_state=config_params.get('random_state', 42),
    )

    # Load train data and retrain
    train_df = pd.read_csv(winner.config.train_path)
    X_train = train_df[text_column].astype(str)
    y_train = train_df[label_column].map(label_mapping).astype(int)

    pipeline = build_linear_pipeline(
        unigram_ngram_range=recon_config.unigram_ngram_range,
        unigram_min_df=recon_config.unigram_min_df,
        unigram_max_df=recon_config.unigram_max_df,
        unigram_sublinear_tf=recon_config.unigram_sublinear_tf,
        unigram_stop_words=recon_config.unigram_stop_words,
        multigram_ngram_range=recon_config.multigram_ngram_range,
        multigram_min_df=recon_config.multigram_min_df,
        multigram_max_df=recon_config.multigram_max_df,
        multigram_sublinear_tf=recon_config.multigram_sublinear_tf,
        multigram_stop_words=recon_config.multigram_stop_words,
        penalty=recon_config.penalty,
        loss=recon_config.loss,
        C=recon_config.C,
        dual=recon_config.dual,
        fit_intercept=recon_config.fit_intercept,
        intercept_scaling=recon_config.intercept_scaling,
        tol=recon_config.tol,
        max_iter=recon_config.max_iter,
        random_state=recon_config.random_state,
        calibration_method=recon_config.calibration_method,
        calibration_cv=recon_config.calibration_cv,
    )
    pipeline.fit(X_train, y_train)

    # Predict on test
    X_test = test_df[text_column].astype(str)
    proba = pipeline.predict_proba(X_test)

    class_idx = {c: i for i, c in enumerate(pipeline.classes_)}
    p_neg = proba[:, class_idx[0]]

    # Use winner's threshold
    threshold = winner.val_metrics.threshold
    y_pred = apply_threshold(p_neg, threshold)

    # Compute test metrics
    test_metrics = compute_test_metrics(
        y_true=y_test.values,
        p_neg=p_neg,
        threshold=threshold,
    )

    log.info(
        json_log(
            'test.evaluation',
            component='svm_experiments',
            run_id=winner.config.run_id,
            test_f1_neg=test_metrics.f1_neg,
            test_recall_neg=test_metrics.recall_neg,
            val_f1_neg=winner.val_metrics.f1_neg,
            val_recall_neg=winner.val_metrics.recall_neg,
        )
    )

    return test_metrics
