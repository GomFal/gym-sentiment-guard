"""SVM Linear training module.

Provides train_from_config for reproducible SVM model training from YAML config.
Mirrors LogReg training module architecture.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import Pipeline

from gym_sentiment_guard.models.svm.pipelines import build_linear_pipeline
from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)





def train_from_config(config_path: str | Path) -> dict:
    """
    Train SVM model from YAML configuration file.

    Args:
        config_path: Path to training config YAML

    Returns:
        Dictionary with 'report' (test metrics) and 'artifact_dir' (output path)
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))

    # Load data configuration
    splits = cfg['data']['splits']
    text_col = cfg['data']['text_column']
    label_col = cfg['data']['label_column']
    mapping = cfg['data']['label_mapping']

    # Load datasets
    train_df = pd.read_csv(splits['train'])
    test_df = pd.read_csv(splits['test'])

    x_train = train_df[text_col].astype(str)
    y_train = train_df[label_col].map(mapping).astype(int)
    x_test = test_df[text_col].astype(str)
    y_test = test_df[label_col].map(mapping).astype(int)

    # Extract vectorizer params from YAML strategies
    vec_cfg = cfg['vectorizer']
    strategies = vec_cfg.get('strategies', {})
    unigram_cfg = strategies.get('unigrams', {})
    multigram_cfg = strategies.get('multigrams', {})

    # Extract classifier params
    clf_cfg = cfg['classifier']
    calib_cfg = cfg.get('calibration', {})

    # Build pipeline using shared builder
    model = build_linear_pipeline(
        # Unigram vectorizer params
        unigram_ngram_range=tuple(unigram_cfg.get('ngram_range', [1, 1])),
        unigram_min_df=unigram_cfg.get('min_df', 10),
        unigram_max_df=vec_cfg.get('max_df', 0.90),
        unigram_sublinear_tf=vec_cfg.get('sublinear_tf', True),
        unigram_stop_words=unigram_cfg.get('stop_words', 'curated_safe'),
        # Multigram vectorizer params
        multigram_ngram_range=tuple(multigram_cfg.get('ngram_range', [2, 3])),
        multigram_min_df=multigram_cfg.get('min_df', 2),
        multigram_max_df=vec_cfg.get('max_df', 0.90),
        multigram_sublinear_tf=vec_cfg.get('sublinear_tf', True),
        multigram_stop_words=multigram_cfg.get('stop_words'),
        # LinearSVC params
        penalty=clf_cfg.get('penalty', 'l2'),
        loss=clf_cfg.get('loss', 'squared_hinge'),
        C=clf_cfg.get('C', 1.0),
        dual=clf_cfg.get('dual', True),
        fit_intercept=clf_cfg.get('fit_intercept', True),
        intercept_scaling=clf_cfg.get('intercept_scaling', 1.0),
        tol=clf_cfg.get('tol', 0.0001),
        max_iter=clf_cfg.get('max_iter', 2000),
        random_state=clf_cfg.get('random_state', 42),
        # Calibration params
        calibration_method=calib_cfg.get('method', 'isotonic'),
        calibration_cv=calib_cfg.get('cv', 5),
    )

    log.info(
        json_log(
            'pipeline.built',
            component='svm_training',
            C=clf_cfg.get('C', 1.0),
            calibration=calib_cfg.get('method', 'isotonic'),
        )
    )

    # Train
    log.info(json_log('train.start', component='svm_training', config=str(config_path)))
    model.fit(x_train, y_train)
    log.info(json_log('train.completed', component='svm_training'))

    # Get class indices for probability access
    classes = model.classes_
    class_index = {cls: idx for idx, cls in enumerate(classes)}

    # Decision configuration
    decision_cfg = cfg['decision']
    target_name = decision_cfg['target_class']
    if target_name not in mapping:
        raise ValueError(f"decision.target_class '{target_name}' not found in label mapping")
    target_label = mapping[target_name]

    # Find the other class
    other_items = [(name, val) for name, val in mapping.items() if name != target_name]
    if not other_items:
        raise ValueError('label mapping must contain at least two classes')
    other_name, other_label = other_items[0]

    # Predict on TEST
    proba_matrix = model.predict_proba(x_test)
    p_target = proba_matrix[:, class_index[target_label]]
    threshold = decision_cfg['threshold']
    y_test_pred = np.where(p_target >= threshold, target_label, other_label)

    # Compute metrics
    y_target = (y_test == target_label).astype(int)
    pr_auc_target = average_precision_score(y_target, p_target)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    test_metrics = {'classification_report': report_test, 'pr_auc_target': pr_auc_target}

    # Create predictions DataFrame for future analysis
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_test_pred,
        'p_neg': proba_matrix[:, class_index[target_label]],
        'p_pos': proba_matrix[:, class_index[other_label]],
    })

    # Save artifacts
    artifacts = cfg['artifacts']
    base_dir = Path(artifacts['output_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    today = datetime.now(UTC).strftime('%Y-%m-%d')
    existing = sorted(
        p.name for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(f'model.{today}_')
    )
    last_idx = int(existing[-1].split('_')[-1]) if existing else 0
    run_id = f'model.{today}_{last_idx + 1:03d}'

    # Create output directory
    out_dir = base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save model
    model_path = out_dir / 'svm.joblib'
    joblib.dump(model, model_path, compress=3)

    # Save metadata
    metadata = {
        'model_name': cfg['model']['name'],
        'model_type': cfg['model']['type'],
        'version': f'{today}_{last_idx + 1:03d}',
        'config_path': str(config_path),
        'python_version': platform.python_version(),
        'threshold': threshold,
        'threshold_target_class': target_name,
        'label_mapping': mapping,
        'classifier_params': cfg['classifier'],
        'calibration_config': cfg.get('calibration', {}),
        'artifacts': {
            'format': artifacts['format'],
            'path': str(model_path),
            'run_id': run_id,
        },
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out_dir / 'metrics_test.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf-8')
    predictions_df.to_csv(out_dir / 'test_predictions.csv', index=False)

    log.info(
        json_log(
            'train.artifacts_saved',
            component='svm_training',
            run_id=run_id,
            output_dir=str(out_dir),
        )
    )

    return {'report': test_metrics, 'artifact_dir': str(out_dir)}
