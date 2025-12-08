"""Dataset splitting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from gym_sentiment_guard.utils import get_logger, json_log

log = get_logger(__name__)


def split_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    stratify_column: str = "sentiment",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    """Split a dataset into train/val/test sets and save them."""
    ratios_sum = train_ratio + val_ratio + test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    input_path = Path(input_csv)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if stratify_column not in df.columns:
        raise ValueError(f"Column '{stratify_column}' not found for stratified split")

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df[stratify_column],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / temp_ratio) if temp_ratio else 0.0,
        stratify=temp_df[stratify_column],
        random_state=random_state,
    )

    train_path = output_base / "train" / "train.csv"
    val_path = output_base / "val" / "val.csv"
    test_path = output_base / "test" / "test.csv"

    for path in (train_path, val_path, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info(
        json_log(
            "split.completed",
            component="data.split",
            input=str(input_path),
            output_dir=str(output_base),
            stratify=stratify_column,
            rows=int(len(df)),
            train_rows=int(len(train_df)),
            val_rows=int(len(val_df)),
            test_rows=int(len(test_df)),
        )
    )
    return train_path, val_path, test_path
