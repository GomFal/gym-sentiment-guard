from __future__ import annotations

import pandas as pd
import pytest

from gym_sentiment_guard.config import (
    CleaningConfig,
    DedupConfig,
    ExpectationsConfig,
    LanguageConfig,
    PathConfig,
    PreprocessConfig,
)
from gym_sentiment_guard.data import language
from gym_sentiment_guard.pipeline import preprocess_reviews


class DummyFastTextModel:
    def predict(self, texts, k=1):
        labels = []
        scores = []
        for text in texts:
            if 'hola' in text.lower():
                label_seq = ['__label__es', '__label__en']
                score_seq = [0.9, 0.1]
            else:
                label_seq = ['__label__en', '__label__es']
                score_seq = [0.85, 0.15]
            if len(label_seq) < k:
                label_seq.extend(['__label__en'] * (k - len(label_seq)))
                score_seq.extend([0.0] * (k - len(score_seq)))
            labels.append(label_seq[:k])
            scores.append(score_seq[:k])
        return labels, scores


def test_preprocess_reviews_creates_clean_file(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'data' / 'raw'
    interim_dir = tmp_path / 'data' / 'interim'
    processed_dir = tmp_path / 'data' / 'processed'
    for path in (raw_dir, interim_dir, processed_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / 'reviews.csv'
    pd.DataFrame(
        {
            'comment': ['Hola Mundo', 'This is English', 'hola   otra vez   '],
            'rating': [5, 4, 5],
        },
    ).to_csv(raw_csv, index=False)

    model_path = tmp_path / 'artifacts' / 'lid.176.bin'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text('stub')

    cfg = PreprocessConfig(
        paths=PathConfig(
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            processed_dir=processed_dir,
        ),
        language=LanguageConfig(
            model_path=model_path,
            text_column='comment',
            batch_size=2,
        ),
        cleaning=CleaningConfig(text_column='comment'),
        dedup=DedupConfig(subset=('comment',)),
        expectations=ExpectationsConfig(
            required_columns=('comment', 'rating'),
            min_text_length=3,
            drop_null_comments=True,
        ),
    )

    monkeypatch.setattr(language, '_load_fasttext_model', lambda _: DummyFastTextModel())

    result = preprocess_reviews(raw_csv, cfg)

    assert result.exists()
    assert result.name == 'reviews.clean.csv'

    df = pd.read_csv(result)
    assert len(df) == 2
    assert all(df['comment'].str.contains('hola'))
    assert (processed_dir / 'reviews.clean.csv').exists()

    non_spanish_path = interim_dir / 'reviews.non_spanish.csv'
    assert non_spanish_path.exists()
    rejected = pd.read_csv(non_spanish_path)
    assert len(rejected) == 1
    assert rejected['comment'].iloc[0] == 'this is english'
    assert 'es_confidence' in rejected.columns
    assert rejected['es_confidence'].iloc[0] == pytest.approx(0.15)
