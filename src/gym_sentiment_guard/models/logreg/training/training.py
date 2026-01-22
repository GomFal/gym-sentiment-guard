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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

from gym_sentiment_guard.common.stopwords import resolve_stop_words
from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def _build_vectorizer(vectorizer_cfg: dict[str, Any]) -> TfidfVectorizer | FeatureUnion:
    """
    Build vectorizer from config.

    Supports two modes:
    - Legacy: Single TfidfVectorizer (when 'strategies' key absent)
    - Hybrid: FeatureUnion of multiple vectorizers (when 'strategies' key present)

    Per-strategy stopwords:
    - If strategy defines 'stop_words', use that value
    - Otherwise fallback to global 'stop_words'
    - Use 'stop_words: null' to disable filtering for a strategy

    Args:
        vectorizer_cfg: Vectorizer configuration dict from YAML

    Returns:
        TfidfVectorizer or FeatureUnion depending on config
    """
    # Common settings inherited by all vectorizers (excluding stop_words)
    common_params = {
        'lowercase': vectorizer_cfg.get('lowercase', True),
        'max_df': vectorizer_cfg.get('max_df', 1.0),
        'sublinear_tf': vectorizer_cfg.get('sublinear_tf', True),
    }
    global_stop_words = vectorizer_cfg.get('stop_words')

    # Branch: FeatureUnion (new path)
    if 'strategies' in vectorizer_cfg:
        transformers = []
        for name, strategy in vectorizer_cfg['strategies'].items():
            # Resolve stopwords: strategy-level > global-level
            if 'stop_words' in strategy:
                stop_words = resolve_stop_words(strategy['stop_words'])
                stopwords_mode = 'custom' if strategy['stop_words'] else 'none'
            else:
                stop_words = resolve_stop_words(global_stop_words)
                stopwords_mode = 'global'

            vec = TfidfVectorizer(
                ngram_range=tuple(strategy['ngram_range']),
                min_df=strategy.get('min_df', 1),
                stop_words=stop_words,
                **common_params,
            )
            transformers.append((name, vec))

            log.info(
                json_log(
                    'vectorizer.strategy',
                    component='training',
                    strategy=name,
                    stopwords_mode=stopwords_mode,
                    ngram_range=strategy['ngram_range'],
                    min_df=strategy.get('min_df', 1),
                )
            )

        log.info(
            json_log(
                'vectorizer.mode',
                component='training',
                mode='feature_union',
                n_strategies=len(transformers),
            )
        )
        return FeatureUnion(transformers)

    # Branch: Single vectorizer (legacy path - unchanged behavior)
    log.info(json_log('vectorizer.mode', component='training', mode='single'))
    return TfidfVectorizer(
        ngram_range=tuple(vectorizer_cfg['ngram_range']),
        min_df=vectorizer_cfg['min_df'],
        **common_params,
    )


def train_from_config(config_path: str | Path) -> dict:
    """Train model from YAML configuration file."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))

    splits = cfg['data']['splits']
    text_col = cfg['data']['text_column']
    label_col = cfg['data']['label_column']
    mapping = cfg['data']['label_mapping']

    train_df = pd.read_csv(splits['train'])
    test_df = pd.read_csv(splits['test'])

    x_train = train_df[text_col].astype(str)
    y_train = train_df[label_col].map(mapping).astype(int)
    x_test = test_df[text_col].astype(str)
    y_test = test_df[label_col].map(mapping).astype(int)

    vectorizer = _build_vectorizer(cfg['vectorizer'])
    base_clf = LogisticRegression(
        max_iter=cfg['classifier']['max_iter'],
        C=cfg['classifier']['C'],
        solver=cfg['classifier']['solver'],
        n_jobs=cfg['classifier']['n_jobs'],
        class_weight=cfg['classifier']['class_weight'],
    )
    calib_cfg = cfg['calibration']
    if calib_cfg['enabled']:
        classifier = CalibratedClassifierCV(
            base_clf,
            method=calib_cfg['method'],
            cv=calib_cfg['cv'],
        )
    else:
        classifier = base_clf

    model = Pipeline(
        [
            ('tfidf', vectorizer),
            ('logreg', classifier),
        ]
    )

    log.info(json_log('train.start', component='training', config=str(config_path)))
    model.fit(x_train, y_train)
    log.info(json_log('train.completed', component='training'))

    classes = model.classes_
    class_index = {cls: idx for idx, cls in enumerate(classes)}
    decision_cfg = cfg['decision']
    target_name = decision_cfg['target_class']
    if target_name not in mapping:
        raise ValueError(f"decision.target_class '{target_name}' not found in label mapping")
    target_label = mapping[target_name]
    other_items = [(name, val) for name, val in mapping.items() if name != target_name]
    if not other_items:
        raise ValueError('label mapping must contain at least two classes')
    other_name, other_label = other_items[0]
    proba_matrix = model.predict_proba(x_test)
    p_target = proba_matrix[:, class_index[target_label]]
    threshold = decision_cfg['threshold']
    y_test_pred = np.where(p_target >= threshold, target_label, other_label)
    y_target = (y_test == target_label).astype(int)
    pr_auc_target = average_precision_score(y_target, p_target)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    test_metrics = {'classification_report': report_test, 'pr_auc_target': pr_auc_target}

    # Create predictions DataFrame for future analysis (PR curves, calibration)
    predictions_df = pd.DataFrame(
        {
            'y_true': y_test.values,
            'y_pred': y_test_pred,
            'p_neg': proba_matrix[:, class_index[target_label]],
            'p_pos': proba_matrix[:, class_index[other_label]],
        }
    )

    artifacts = cfg['artifacts']
    base_dir = Path(artifacts['output_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime('%Y-%m-%d')
    existing = sorted(
        p.name for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(f'model.{today}_')
    )
    last_idx = int(existing[-1].split('_')[-1]) if existing else 0
    run_id = f'model.{today}_{last_idx + 1:03d}'
    out_dir = base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    model_path = out_dir / 'logreg.joblib'
    joblib.dump(model, model_path, compress=3)

    metadata = {
        'model_name': cfg['model']['name'],
        'version': f'{today}_{last_idx + 1:03d}',
        'config_path': str(config_path),
        'python_version': platform.python_version(),
        'threshold': threshold,
        'threshold_target_class': target_name,
        'label_mapping': mapping,
        'artifacts': {
            'format': artifacts['format'],
            'path': str(model_path),
            'run_id': run_id,
        },
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out_dir / 'metrics_test.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf-8')
    predictions_df.to_csv(out_dir / 'test_predictions.csv', index=False)

    return {'report': test_metrics, 'artifact_dir': str(out_dir)}
