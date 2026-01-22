from __future__ import annotations

import pandas as pd

from scripts.merge_sampled_languages import merge_sampled_files


def test_merge_sampled_files(tmp_path):
    root = tmp_path / 'lid_eval'
    root.mkdir()
    es_sampled = root / 'es.sampled.csv'
    en_sampled = root / 'en.sampled.csv'
    pd.DataFrame({'comment': ['hola'], 'language': ['es']}).to_csv(es_sampled, index=False)
    pd.DataFrame({'comment': ['hello'], 'language': ['en']}).to_csv(en_sampled, index=False)

    output = merge_sampled_files(root, pattern='*.sampled.csv', output='merged.csv')

    merged = pd.read_csv(output)
    assert output == root / 'merged.csv'
    assert len(merged) == 2
    assert set(merged['language']) == {'es', 'en'}
