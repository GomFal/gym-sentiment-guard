"""
Ablation suite orchestrator.

Implements §9 of EXPERIMENT_PROTOCOL.md:
- Compare configurations using VAL only
- Change one conceptual factor at a time
- Select winner based on primary objective + tie-breakers
- Evaluate winner on TEST once
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.logging import get_logger, json_log
from .artifacts import RunResult
from .grid import LOGREG_GRID, TFIDF_GRID
from .metrics import compute_test_metrics
from .runner import ExperimentConfig, run_single_experiment

log = get_logger(__name__)


def generate_grid_configs(
    base_config: ExperimentConfig,
    tfidf_grid: dict[str, list] | None = None,
    logreg_grid: dict[str, list] | None = None,
) -> list[ExperimentConfig]:
    """
    Generate all configurations from parameter grids.

    Args:
        base_config: Base experiment configuration
        tfidf_grid: TF-IDF parameter grid (defaults to TFIDF_GRID)
        logreg_grid: LogReg parameter grid (defaults to LOGREG_GRID)

    Returns:
        List of ExperimentConfig for each combination
    """
    tfidf = tfidf_grid or TFIDF_GRID
    logreg = logreg_grid or LOGREG_GRID

    configs = []

    # Generate all combinations
    tfidf_keys = list(tfidf.keys())
    logreg_keys = list(logreg.keys())

    tfidf_values = [tfidf[k] for k in tfidf_keys]
    logreg_values = [logreg[k] for k in logreg_keys]

    for tfidf_combo in product(*tfidf_values):
        for logreg_combo in product(*logreg_values):
            tfidf_params = dict(zip(tfidf_keys, tfidf_combo, strict=True))
            logreg_params = dict(zip(logreg_keys, logreg_combo, strict=True))

            config = ExperimentConfig(
                train_path=base_config.train_path,
                val_path=base_config.val_path,
                test_path=base_config.test_path,
                text_column=base_config.text_column,
                label_column=base_config.label_column,
                label_mapping=base_config.label_mapping,
                recall_constraint=base_config.recall_constraint,
                output_dir=base_config.output_dir,
                save_predictions=base_config.save_predictions,
                # TF-IDF params
                ngram_range=tfidf_params.get('ngram_range', (1, 1)),
                min_df=tfidf_params.get('min_df', 2),
                max_df=tfidf_params.get('max_df', 1.0),
                sublinear_tf=tfidf_params.get('sublinear_tf', True),
                stop_words=tfidf_params.get('stop_words'),  # None or 'curated_safe'
                # LogReg params
                penalty=logreg_params.get('penalty', 'l2'),
                C=logreg_params.get('C', 1.0),
                class_weight=logreg_params.get('class_weight'),
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

        # Validity order: valid > constraint_not_met > invalid
        validity_order = 2 if r.validity_status == 'valid' else 1

        # Constraint order: met > not_met
        constraint_order = 1 if r.val_metrics.constraint_status == 'met' else 0

        # Negate brier and ece so lower = better in descending sort
        return (
            validity_order,
            constraint_order,
            r.val_metrics.f1_neg,
            r.val_metrics.macro_f1,
            r.val_metrics.pr_auc_neg,
            -r.val_metrics.brier_score,  # Lower is better
            -r.val_metrics.ece,  # Lower is better
        )

    return sorted(results, key=sort_key, reverse=True)


def run_ablation_suite(
    base_config: ExperimentConfig,
    tfidf_grid: dict[str, list] | None = None,
    logreg_grid: dict[str, list] | None = None,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """
    Run full ablation suite.

    Args:
        base_config: Base experiment configuration
        tfidf_grid: TF-IDF parameter grid (optional)
        logreg_grid: LogReg parameter grid (optional)
        max_runs: Maximum number of runs (for testing, None = all)

    Returns:
        Dictionary with:
        - results: All run results
        - winner: Best configuration
        - summary: Summary statistics
    """
    # Generate all configs
    configs = generate_grid_configs(base_config, tfidf_grid, logreg_grid)

    if max_runs is not None:
        configs = configs[:max_runs]

    log.info(
        json_log(
            'ablation.start',
            component='experiments',
            n_configs=len(configs),
        )
    )

    # Run all experiments
    results = []
    for i, config in enumerate(configs):
        log.info(
            json_log(
                'ablation.run_progress',
                run_number=i + 1,
                total=len(configs),
            )
        )
        result = run_single_experiment(config)
        results.append(result)

    # Rank results
    ranked = rank_results(results)

    # Get winner
    winner = ranked[0] if ranked else None

    # Summary statistics
    valid_count = sum(1 for r in results if r.validity_status == 'valid')
    constraint_not_met_count = sum(1 for r in results if r.validity_status == 'constraint_not_met')
    invalid_count = sum(1 for r in results if r.validity_status == 'invalid')

    summary = {
        'total_runs': len(results),
        'valid_runs': valid_count,
        'constraint_not_met_runs': constraint_not_met_count,
        'invalid_runs': invalid_count,
        'winner_run_id': winner.config.run_id if winner else None,
        'winner_f1_neg': winner.val_metrics.f1_neg if winner and winner.val_metrics else None,
        'winner_threshold': winner.val_metrics.threshold if winner and winner.val_metrics else None,
    }

    log.info(
        json_log(
            'ablation.completed',
            component='experiments',
            **summary,
        )
    )

    # Save suite summary
    output_dir = Path(base_config.output_dir)
    suite_dir = output_dir / f'suite.{datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")}'
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Save summary JSON
    summary_path = suite_dir / 'suite_summary.json'
    summary_data = {
        **summary,
        'ranked_runs': [
            {
                'run_id': r.config.run_id,
                'validity_status': r.validity_status,
                'f1_neg': r.val_metrics.f1_neg if r.val_metrics else None,
                'recall_neg': r.val_metrics.recall_neg if r.val_metrics else None,
                'macro_f1': r.val_metrics.macro_f1 if r.val_metrics else None,
                'threshold': r.val_metrics.threshold if r.val_metrics else None,
                'constraint_status': r.val_metrics.constraint_status if r.val_metrics else None,
            }
            for r in ranked
        ],
    }
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding='utf-8',
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
) -> dict[str, Any]:
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

    log.info(
        json_log(
            'test.evaluation_start',
            component='experiments',
            winner_run_id=winner.config.run_id,
        )
    )

    # Rebuild the winning pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    from .grid import CALIBRATION_CONFIG, FIXED_PARAMS, STOPWORDS_SAFE

    tfidf_params = winner.config.tfidf_params
    logreg_params = winner.config.logreg_params

    # Rebuild vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=tuple(tfidf_params['ngram_range']),
        min_df=tfidf_params['min_df'],
        max_df=tfidf_params['max_df'],
        sublinear_tf=tfidf_params['sublinear_tf'],
        stop_words=STOPWORDS_SAFE,
    )

    # Rebuild classifier
    base_clf = LogisticRegression(
        penalty=logreg_params['penalty'],
        C=logreg_params['C'],
        class_weight=logreg_params['class_weight'],
        solver=logreg_params['solver'],
        max_iter=logreg_params['max_iter'],
        random_state=logreg_params['random_state'],
        n_jobs=FIXED_PARAMS['n_jobs'],
    )

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

    pipeline = Pipeline(
        [
            ('tfidf', vectorizer),
            ('classifier', calibrated),
        ]
    )

    # Load train and test data
    train_df = pd.read_csv(winner.config.train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[text_column].astype(str)
    y_train = train_df[label_column].map(label_mapping).astype(int)
    X_test = test_df[text_column].astype(str)
    y_test = test_df[label_column].map(label_mapping).astype(int)

    # Fit on train
    pipeline.fit(X_train, y_train)

    # Get predictions on test
    proba = pipeline.predict_proba(X_test)
    class_idx = {c: i for i, c in enumerate(pipeline.classes_)}
    p_neg = proba[:, class_idx[0]]

    # Use threshold from VAL (not re-optimized per §7.2)
    threshold = winner.val_metrics.threshold

    # Compute test metrics
    test_metrics = compute_test_metrics(
        y_true=y_test.values,
        p_neg=p_neg,
        threshold=threshold,
    )

    log.info(
        json_log(
            'test.evaluation_completed',
            component='experiments',
            f1_neg=test_metrics.f1_neg,
            recall_neg=test_metrics.recall_neg,
            threshold=threshold,
        )
    )

    return {
        'test_metrics': asdict(test_metrics),
        'winner_run_id': winner.config.run_id,
        'threshold_from_val': threshold,
    }
