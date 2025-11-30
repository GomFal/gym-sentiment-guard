from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.eval_fasttext_lid import evaluate_fasttext_lid


class DummyFastTextModel:
    def predict(self, texts, k=1):
        mapping = {
            'hola': '__label__es',
            'hello': '__label__en',
            'ciao': '__label__it',
        }
        labels = []
        probs = []
        for text in texts:
            label = mapping.get(text.split()[0].lower(), '__label__en')
            labels.append([label])
            probs.append([0.9])
        return labels, probs


def test_evaluate_fasttext_lid(monkeypatch, tmp_path):
    data = pd.DataFrame(
        {
            'comment': ['hola', 'hello', 'ciao'],
            'language': ['es', 'en', 'it'],
        },
    )
    data_path = tmp_path / 'eval.csv'
    data.to_csv(data_path, index=False)

    model_path = tmp_path / 'lid.176.bin'
    model_path.write_text('stub', encoding='utf-8')

    monkeypatch.setattr(
        'scripts.eval_fasttext_lid.fasttext.load_model',
        lambda path: DummyFastTextModel(),
    )

    output_json = tmp_path / 'report.json'
    confusion_png = tmp_path / 'cm.png'
    results = evaluate_fasttext_lid(
        data_path=data_path,
        model_path=model_path,
        output_json=output_json,
        confusion_png=confusion_png,
    )

    assert pytest.approx(results['accuracy']) == 1.0
    assert output_json.exists()
    assert confusion_png.exists()
    payload = json.loads(output_json.read_text(encoding='utf-8'))
    assert payload['num_samples'] == 3
    assert payload['accuracy'] == results['accuracy']
    thresholds = payload['confidence_summary']['thresholds']
    assert thresholds['0.95']['count'] == 0
    assert thresholds['0.95']['metrics']['accuracy'] is None
    assert thresholds['0.90']['count'] == 3
    metrics = thresholds['0.90']['metrics']
    assert pytest.approx(metrics['accuracy']) == 1.0
    assert metrics['precision'] == pytest.approx(1.0)
