"""Preprocessing orchestration."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

from ..config import PreprocessConfig
from ..data import (
    deduplicate_reviews,
    enforce_expectations,
    filter_spanish_comments,
    normalize_comments,
)
from ..utils import get_logger, json_log

log = get_logger(__name__)


def preprocess_reviews(
    input_path: str | Path,
    config: PreprocessConfig,
    output_path: str | Path | None = None,
) -> Path:
    """Preprocess a raw CSV through language filtering and cleaning steps."""
    source_path = Path(input_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f'Input file not found: {source_path}')

    start = time.perf_counter()
    log.info(
        json_log(
            'preprocess.start',
            component='pipeline.preprocess',
            input=str(source_path),
        ),
    )

    paths = config.paths
    interim_dir = paths.interim_dir
    interim_dir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix or '.csv'
    stem = source_path.stem

    validated_path = interim_dir / f'{stem}.validated{suffix}'
    normalized_path = interim_dir / f'{stem}.normalized{suffix}'
    dedup_path = interim_dir / f'{stem}.dedup{suffix}'
    spanish_path = interim_dir / f'{stem}.spanish{suffix}'
    non_spanish_path = interim_dir / f'{stem}.non_spanish{suffix}'

    enforce_expectations(
        input_csv=source_path,
        output_path=validated_path,
        required_columns=config.expectations.required_columns,
        text_column=config.cleaning.text_column,
        min_text_length=config.expectations.min_text_length,
        drop_null_comments=config.expectations.drop_null_comments,
    )

    normalize_comments(
        input_csv=validated_path,
        output_path=normalized_path,
        text_column=config.cleaning.text_column,
    )

    deduplicate_reviews(
        input_csv=normalized_path,
        output_path=dedup_path,
        subset=config.dedup.subset,
    )

    language_enabled = config.language.enabled
    if language_enabled:
        filter_spanish_comments(
            input_csv=dedup_path,
            output_dir=interim_dir,
            output_path=spanish_path,
            rejected_output_path=non_spanish_path,
            model_path=config.language.model_path,
            text_column=config.language.text_column,
            batch_size=config.language.batch_size,
        )

    final_output = (
        Path(output_path).resolve()
        if output_path
        else paths.processed_dir / f'{stem}.clean{suffix}'
    )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    source_final = spanish_path if language_enabled else dedup_path
    shutil.copy2(source_final, final_output)

    row_count = _count_rows(final_output)
    duration = time.perf_counter() - start
    log.info(
        json_log(
            'preprocess.completed',
            component='pipeline.preprocess',
            input=str(source_path),
            output=str(final_output),
            rows=row_count,
            duration_seconds=duration,
        ),
    )
    return final_output


def _count_rows(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    return len(df)
