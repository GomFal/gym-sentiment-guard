"""Sentiment features derived from numeric ratings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gym_sentiment_guard.utils import get_logger, json_log

log = get_logger(__name__)


def add_rating_sentiment(
    input_csv: str | Path,
    output_csv: str | Path,
    rating_column: str = "rating",
    sentiment_column: str = "sentiment",
) -> Path:
    """
    Map ratings to sentiment labels and write a new CSV.

    Args:
        input_csv: Source CSV path.
        output_csv: Destination CSV path (with sentiment column added).
        rating_column: Column containing numeric ratings.
        sentiment_column: Name of sentiment column to add.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if rating_column not in df.columns:
        raise ValueError(f"Column '{rating_column}' not found for sentiment creation")

    rating_series = df[rating_column]
    sentiment = pd.Series(index=rating_series.index, dtype="object")
    sentiment[rating_series.isin([4, 5])] = "positive"
    sentiment[rating_series.isin([1, 2])] = "negative"

    unknown_mask = sentiment.isna()
    if unknown_mask.any():
        log.warning(
            json_log(
                "features.sentiment.unknown_rating",
                component="features.sentiment",
                count=int(unknown_mask.sum()),
            )
        )
        sentiment[unknown_mask] = "unknown"

    df[sentiment_column] = sentiment
    df.to_csv(output_path, index=False)

    log.info(
        json_log(
            "features.sentiment.completed",
            component="features.sentiment",
            input=str(input_path),
            output=str(output_path),
            rows=len(df),
            sentiment_column=sentiment_column,
        )
    )
    return output_path
