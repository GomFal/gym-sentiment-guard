"""Text cleaning and validation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..utils import get_logger, json_log

log = get_logger(__name__)

EMOJI_PATTERN = re.compile(
    '['
    '\U0001F300-\U0001F5FF'  # symbols & pictographs
    '\U0001F600-\U0001F64F'  # emoticons
    '\U0001F680-\U0001F6FF'  # transport & map
    '\U0001F700-\U0001F77F'  # alchemical symbols
    '\U0001F780-\U0001F7FF'  # geometric shapes extended
    '\U0001F800-\U0001F8FF'  # supplemental arrows-c
    '\U0001F900-\U0001F9FF'  # supplemental symbols & pictographs
    '\U0001FA00-\U0001FAFF'  # symbols and pictographs extended-a
    '\U00002702-\U000027B0'  # dingbats
    '\U000024C2-\U0001F251'  # enclosed characters
    '\U0001F1E6-\U0001F1FF'  # flags
    '\U0001F018-\U0001F270'
    '\U0001F600-\U0001F636'
    '\U0001F681-\U0001F6C5'
    '\U0001F30D-\U0001F567'
    ']+',
    flags=re.UNICODE,
)

DEFAULT_STRUCTURAL_PUNCTUATION = r"[.,;:\(\)\[\]\{\}<>\-–—_/\\|\'\"«»“”‘’`~^@#$%&*=+©®]"

def normalize_comments(
    input_csv: str | Path,
    output_path: str | Path,
    text_column: str = "comment",
    structural_punctuation: str | None = None,
) -> Path:
    """Normalize comment text (lowercase, strip, collapse whitespace)."""
    input_path = Path(input_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found for normalization")

    series = df[text_column].fillna("").astype(str)
    pattern = structural_punctuation or DEFAULT_STRUCTURAL_PUNCTUATION
    normalized = (
        series.str.lower()
        .str.replace(pattern, " ", regex=True)
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


def drop_columns(
    input_csv: str | Path,
    output_path: str | Path,
    columns: Iterable[str],
) -> Path:
    """Drop specified columns from a CSV."""
    input_path = Path(input_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        log.warning(
            json_log(
                "drop_columns.missing",
                component="data.cleaning",
                input=str(input_path),
                missing=missing,
            )
        )
    filtered = df.drop(columns=[col for col in columns if col in df.columns])
    filtered.to_csv(output_path, index=False)

    log.info(
        json_log(
            "drop_columns.completed",
            component="data.cleaning",
            input=str(input_path),
            output=str(output_path),
            columns_dropped=[col for col in columns if col in df.columns],
            rows=len(filtered),
        )
    )
    return output_path


def drop_neutral_ratings(
    input_csv: str | Path,
    output_path: str | Path,
    rating_column: str = "rating",
    neutral_value: int | float = 3,
    neutral_output_path: str | Path | None = None,
) -> Path:
    """Remove rows whose rating equals the neutral value."""
    input_path = Path(input_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if rating_column not in df.columns:
        raise ValueError(f"Column '{rating_column}' not found for neutral drop")

    original_rows = len(df)
    neutral_mask = df[rating_column] == neutral_value
    filtered = df.loc[~neutral_mask]
    filtered.to_csv(output_path, index=False)

    if neutral_output_path is not None:
        neutral_path = Path(neutral_output_path)
        neutral_path.parent.mkdir(parents=True, exist_ok=True)
        df.loc[neutral_mask].to_csv(neutral_path, index=False)
    else:
        neutral_path = None

    log.info(
        json_log(
            "neutral_drop.completed",
            component="data.cleaning",
            input=str(input_path),
            output=str(output_path),
            rows_in=original_rows,
            rows_out=len(filtered),
            rating_column=rating_column,
            neutral_value=neutral_value,
            neutral_output=str(neutral_path) if neutral_path else None,
            neutral_rows=int(neutral_mask.sum()),
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
    cleaned_series = series.str.replace(EMOJI_PATTERN, '', regex=True)
    removed_emojis = int(((series != cleaned_series) & series.str.strip().astype(bool)).sum())
    mask = cleaned_series.str.len() >= int(min_text_length)
    if drop_null_comments:
        mask &= cleaned_series.str.strip() != ""

    filtered = df.loc[mask].copy()
    filtered[text_column] = cleaned_series[mask]
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
            emoji_stripped=removed_emojis,
        ),
    )
    return output_path


@lru_cache(maxsize=8)
def load_structural_punctuation(path: str | Path | None) -> str | None:
    """Return the punctuation pattern defined in the provided file."""
    if path is None:
        return None
    pattern_path = Path(path)
    if not pattern_path.exists():
        raise FileNotFoundError(f"Structural punctuation file not found: {pattern_path}")
    return pattern_path.read_text(encoding="utf-8").strip()
