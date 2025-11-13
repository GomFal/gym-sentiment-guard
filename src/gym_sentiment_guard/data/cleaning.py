"""Text cleaning and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import get_logger, json_log

log = get_logger(__name__)


def normalize_comments(
    input_csv: str | Path,
    output_path: str | Path,
    text_column: str = "comment",
) -> Path:
    """Normalize comment text (lowercase, strip, collapse whitespace)."""
    input_path = Path(input_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found for normalization")

    series = df[text_column].fillna("").astype(str)
    normalized = (
        series.str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[text_column] = normalized

    df.to_csv(output_path, index=False)

    log.info(
        json_log(
            "normalize.completed",
            component="data.cleaning",
            input=str(input_path),
            output=str(output_path),
            rows=len(df),
            text_column=text_column,
        ),
    )
    return output_path


def deduplicate_reviews(
    input_csv: str | Path,
    output_path: str | Path,
    subset: Iterable[str] | None = None,
) -> Path:
    """Drop duplicate rows using the provided subset columns."""
    input_path = Path(input_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    original_rows = len(df)
    deduped = df.drop_duplicates(subset=list(subset) if subset else None)
    deduped.to_csv(output_path, index=False)

    log.info(
        json_log(
            "dedup.completed",
            component="data.cleaning",
            input=str(input_path),
            output=str(output_path),
            rows_in=original_rows,
            rows_out=len(deduped),
            subset=list(subset) if subset else None,
        ),
    )
    return output_path


def enforce_expectations(
    input_csv: str | Path,
    output_path: str | Path,
    required_columns: Iterable[str],
    text_column: str,
    min_text_length: int = 1,
    drop_null_comments: bool = True,
) -> Path:
    """Apply lightweight expectations to the dataset."""
    input_path = Path(input_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    series = df[text_column].fillna("").astype(str)
    mask = series.str.len() >= int(min_text_length)
    if drop_null_comments:
        mask &= series.str.strip() != ""

    filtered = df.loc[mask].copy()
    filtered.to_csv(output_path, index=False)

    log.info(
        json_log(
            "expectations.completed",
            component="data.cleaning",
            input=str(input_path),
            output=str(output_path),
            rows_in=len(df),
            rows_out=len(filtered),
            min_text_length=min_text_length,
        ),
    )
    return output_path
