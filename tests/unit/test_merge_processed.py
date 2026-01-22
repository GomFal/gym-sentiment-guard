from __future__ import annotations

import pandas as pd

from gym_sentiment_guard.data.merge import merge_processed_csvs


def test_merge_processed_csvs_reindexes_ids(tmp_path):
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()

    df1 = pd.DataFrame({'id': [10, 11], 'comment': ['a', 'b'], 'rating': [5, 4]})
    df2 = pd.DataFrame({'id': [20], 'comment': ['c'], 'rating': [1]})
    df1.to_csv(processed_dir / 'file1.clean.csv', index=False)
    df2.to_csv(processed_dir / 'file2.clean.csv', index=False)

    output = tmp_path / 'merged.csv'
    merge_processed_csvs(processed_dir, output_path=output)

    merged = pd.read_csv(output)
    assert merged['id'].tolist() == [1, 2, 3]
