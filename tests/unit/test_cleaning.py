from __future__ import annotations

import pandas as pd

from gym_sentiment_guard.data.cleaning import enforce_expectations


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
