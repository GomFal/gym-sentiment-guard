from __future__ import annotations

import pandas as pd

from gym_sentiment_guard.data.cleaning import (
    drop_neutral_ratings,
    enforce_expectations,
)


def test_enforce_expectations_strips_emojis(tmp_path):
    input_csv = tmp_path / 'raw.csv'
    df = pd.DataFrame(
        {
            'comment': ['Hola 😊', 'Solo emoji 😭', 'Texto limpio'],
            'rating': [5, 4, 3],
        },
    )
    df.to_csv(input_csv, index=False)

    output_csv = tmp_path / 'validated.csv'
    result = enforce_expectations(
        input_csv=input_csv,
        output_path=output_csv,
        required_columns=('comment', 'rating'),
        text_column='comment',
        min_text_length=3,
        drop_null_comments=True,
    )

    assert result == output_csv
    cleaned = pd.read_csv(result)
    assert list(cleaned['comment']) == ['Hola ', 'Solo emoji ', 'Texto limpio']


def test_drop_neutral_ratings_removes_rating_three_and_saves_neutral(tmp_path):
    input_csv = tmp_path / 'raw.csv'
    df = pd.DataFrame(
        {
            'comment': ['pos', 'neutral', 'neg'],
            'rating': [5, 3, 1],
        },
    )
    df.to_csv(input_csv, index=False)

    output_csv = tmp_path / 'filtered.csv'
    neutral_csv = tmp_path / 'neutral.csv'
    drop_neutral_ratings(
        input_csv=input_csv,
        output_path=output_csv,
        rating_column='rating',
        neutral_value=3,
        neutral_output_path=neutral_csv,
    )

    filtered = pd.read_csv(output_csv)
    assert list(filtered['rating']) == [5, 1]
    neutral = pd.read_csv(neutral_csv)
    assert neutral['comment'].tolist() == ['neutral']
