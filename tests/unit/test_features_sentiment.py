from __future__ import annotations

import pandas as pd

from gym_sentiment_guard.features import add_rating_sentiment


def test_add_rating_sentiment_maps_pos_neg(tmp_path):
    input_csv = tmp_path / "input.csv"
    df = pd.DataFrame({"rating": [5, 4, 2, 1]})
    df.to_csv(input_csv, index=False)

    output_csv = tmp_path / "output.csv"
    add_rating_sentiment(input_csv=input_csv, output_csv=output_csv)

    result = pd.read_csv(output_csv)
    assert result["sentiment"].tolist() == ["positive", "positive", "negative", "negative"]


def test_add_rating_sentiment_marks_unknown(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"rating": [3]}).to_csv(input_csv, index=False)

    output_csv = tmp_path / "output.csv"
    add_rating_sentiment(input_csv=input_csv, output_csv=output_csv)

    result = pd.read_csv(output_csv)
    assert result["sentiment"].tolist() == ["unknown"]
