from __future__ import annotations

import pandas as pd

from scripts.sample_language_reviews import sample_language_reviews


def test_sample_language_reviews(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    es_clean = root / 'es.clean.csv'
    en_clean = root / 'en.clean.csv'
    pd.DataFrame({'comment': [f'hola {i}' for i in range(10)]}).to_csv(es_clean, index=False)
    pd.DataFrame({'comment': [f'hello {i}' for i in range(5)]}).to_csv(en_clean, index=False)

    outputs = sample_language_reviews(
        root, pattern='*.clean.csv', n_samples=3, output_suffix='.sampled.csv'
    )

    es_sampled = es_clean.with_name(es_clean.stem + '.sampled.csv')
    en_sampled = en_clean.with_name(en_clean.stem + '.sampled.csv')
    assert es_sampled in outputs
    assert en_sampled in outputs
    assert pd.read_csv(es_sampled).shape[0] == 3
    assert pd.read_csv(en_sampled).shape[0] == 3
