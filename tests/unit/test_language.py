from __future__ import annotations

import pandas as pd
import pytest

from gym_sentiment_guard.data import language


class DummyFastTextModel:
    """Simple stand-in for the fastText predictor."""

    def predict(self, texts, k=1):
        labels = []
        probabilities = []
        for text in texts:
            if 'hola' in text.lower():
                label_seq = ['__label__es', '__label__en']
                prob_seq = [0.9, 0.1]
            else:
                label_seq = ['__label__en', '__label__es']
                prob_seq = [0.85, 0.15]
            if len(label_seq) < k:
                label_seq.extend(['__label__en'] * (k - len(label_seq)))
                prob_seq.extend([0.0] * (k - len(prob_seq)))
            labels.append(label_seq[:k])
            probabilities.append(prob_seq[:k])
        return labels, probabilities


@pytest.mark.parametrize('column', ['comment', 'review'])
def test_filter_spanish_comments_only_keeps_spanish_rows(tmp_path, column, monkeypatch):
    input_csv = tmp_path / 'reviews.csv'
    df = pd.DataFrame(
        {
            column: ['Hola mundo', 'This is English', 'Hola\ncon salto'],
            'rating': [5, 4, 3],
        },
    )
    df.to_csv(input_csv, index=False)

    dummy_model = DummyFastTextModel()

    # Patch the loader to avoid pulling the real binary during tests.
    language._load_fasttext_model.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setattr(language, '_load_fasttext_model', lambda _: dummy_model)

    output_path = language.filter_spanish_comments(
        input_csv=input_csv,
        output_dir=tmp_path / 'processed',
        model_path=tmp_path / 'lid.176.bin',
        text_column=column,
        batch_size=2,
    )

    assert output_path.exists()

    filtered = pd.read_csv(output_path)
    assert len(filtered) == 2
    assert filtered.iloc[0][column] == 'Hola mundo'
    assert filtered.iloc[1][column].startswith('Hola')

    non_spanish_path = output_path.parent / 'reviews.non_spanish.csv'
    assert non_spanish_path.exists()
    rejected = pd.read_csv(non_spanish_path)
    assert len(rejected) == 1
    assert rejected[column].iloc[0].startswith('This is English')
    assert 'es_confidence' in rejected.columns
    assert rejected['es_confidence'].iloc[0] == pytest.approx(0.0)
