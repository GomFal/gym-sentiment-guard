"""
Ablation suite orchestrator for SVC RBF model.

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

from gym_sentiment_guard.models.svm.pipelines import build_rbf_pipeline

from .common import rank_results, validate_label_mapping
from .runner_rbf import SVCRBFExperimentConfig, run_single_rbf_experiment

log = get_logger(__name__)


def generate_rbf_grid_configs(
    base_config: SVCRBFExperimentConfig,
    unigram_grid: dict[str, list],
    multigram_grid: dict[str, list],
    svc_grid: dict[str, list],
) -> list[SVCRBFExperimentConfig]:
    """
    Generate all configurations from parameter grids for SVC RBF.

    Args:
        base_config: Base experiment configuration
        unigram_grid: Unigram TF-IDF parameter grid
        multigram_grid: Multigram TF-IDF parameter grid
        svc_grid: SVC RBF parameter grid (kernel, C, gamma, tol, cache_size, max_iter)

    Returns:
        List of SVCRBFExperimentConfig for each combination
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

    # SVC params (direct mapping)
    combined_grid.update(svc_grid)

    # Generate all combinations
    keys = list(combined_grid.keys())
    values = [combined_grid[k] for k in keys]

    for combo in product(*values):
        params = dict(zip(keys, combo, strict=True))

        config = SVCRBFExperimentConfig(
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
            # SVC RBF params
            kernel=params.get('kernel', base_config.kernel),
            C=params.get('C', base_config.C),
            gamma=params.get('gamma', base_config.gamma),
            tol=params.get('tol', base_config.tol),
            cache_size=params.get('cache_size', base_config.cache_size),
            max_iter=params.get('max_iter', base_config.max_iter),
            use_scaler=params.get('use_scaler', base_config.use_scaler),
        )
        configs.append(config)

    return configs


def run_rbf_ablation_suite(
    base_config: SVCRBFExperimentConfig,
    unigram_grid: dict[str, list],
    multigram_grid: dict[str, list],
    svc_grid: dict[str, list],
    max_runs: int | None = None,
) -> dict[str, Any]:
    """
    Run full SVC RBF ablation suite.

    Args:
        base_config: Base experiment configuration
        unigram_grid: Unigram TF-IDF parameter grid
        multigram_grid: Multigram TF-IDF parameter grid
        svc_grid: SVC RBF parameter grid
        max_runs: Maximum number of runs (for testing, None = all)

    Returns:
        Dictionary with:
        - results: All run results
        - winner: Best configuration
        - summary: Summary statistics
    """
    # Generate all configs
    configs = generate_rbf_grid_configs(base_config, unigram_grid, multigram_grid, svc_grid)

    if max_runs is not None:
        configs = configs[:max_runs]

    log.info(
        json_log(
            'ablation.start',
            component='svm_rbf_experiments',
            n_configs=len(configs),
        )
    )

    # Run all experiments
    results = []
    for i, config in enumerate(configs):
        log.info(
            json_log(
                'ablation.progress',
                component='svm_rbf_experiments',
                current=i + 1,
                total=len(configs),
            )
        )
        result = run_single_rbf_experiment(config)
        results.append(result)

    # Rank results using shared utility
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

    # Build ranked runs list for summary
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

    # Build comprehensive summary
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
    suite_dir = output_dir / f"suite.{datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary_path = suite_dir / 'suite_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if winner and winner.val_metrics:
        log.info(
            json_log(
                'ablation.winner',
                component='svm_rbf_experiments',
                run_id=winner.config.run_id,
                f1_neg=winner.val_metrics.f1_neg,
                recall_neg=winner.val_metrics.recall_neg,
            )
        )

    log.info(
        json_log(
            'ablation.complete',
            component='svm_rbf_experiments',
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


def evaluate_rbf_winner_on_test(
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
    y_test = validate_label_mapping(test_df, label_column, label_mapping, test_path)

    # Reconstruct pipeline from winner config
    config_params = winner.config.classifier_params
    tfidf_params = winner.config.tfidf_params

    # Build config for pipeline reconstruction
    recon_config = SVCRBFExperimentConfig(
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
        # SVC RBF params
        kernel=config_params.get('kernel', 'rbf'),
        C=config_params.get('C', 1.0),
        gamma=config_params.get('gamma', 'scale'),
        tol=config_params.get('tol', 0.001),
        cache_size=config_params.get('cache_size', 1000),
        max_iter=config_params.get('max_iter', -1),
        random_state=config_params.get('random_state', 42),
    )

    # Load train data and retrain
    train_df = pd.read_csv(winner.config.train_path)
    X_train = train_df[text_column].astype(str)
    y_train = validate_label_mapping(train_df, label_column, label_mapping, winner.config.train_path)

    pipeline = build_rbf_pipeline(
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
        kernel=recon_config.kernel,
        C=recon_config.C,
        gamma=recon_config.gamma,
        tol=recon_config.tol,
        cache_size=recon_config.cache_size,
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

    # Compute test metrics
    test_metrics = compute_test_metrics(
        y_true=y_test.values,
        p_neg=p_neg,
        threshold=threshold,
    )

    log.info(
        json_log(
            'test.evaluation',
            component='svm_rbf_experiments',
            run_id=winner.config.run_id,
            test_f1_neg=test_metrics.f1_neg,
            test_recall_neg=test_metrics.recall_neg,
            val_f1_neg=winner.val_metrics.f1_neg,
            val_recall_neg=winner.val_metrics.recall_neg,
        )
    )

    return test_metrics
