from __future__ import annotations

import pandas as pd

from gym_sentiment_guard.data.split import split_dataset


def test_split_dataset_stratified(tmp_path):
    input_csv = tmp_path / "merged.csv"
    df = pd.DataFrame(
        {
            "sentiment": ["positive"] * 70 + ["negative"] * 30,
            "rating": [5] * 70 + [1] * 30,
        }
    )
    df.to_csv(input_csv, index=False)

    output_dir = tmp_path / "splits"
    train_path, val_path, test_path = split_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        stratify_column="sentiment",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=0,
    )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["sentiment"]) == {"positive", "negative"}
    assert set(val["sentiment"]) == {"positive", "negative"}
    assert set(test["sentiment"]) == {"positive", "negative"}
