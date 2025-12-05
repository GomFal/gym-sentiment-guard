"""Preprocessing orchestration."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..config import PreprocessConfig
from ..data import (
    configure_language_fallback,
    deduplicate_reviews,
    drop_neutral_ratings,
    enforce_expectations,
    filter_spanish_comments,
    merge_processed_csvs,
    normalize_comments,
)
from ..features import add_rating_sentiment
from ..io import count_csv_rows, list_pending_raw_files
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
    neutral_path = interim_dir / f'{stem}.non_neutral{suffix}'
    neutral_dump_path = interim_dir / f'{stem}.neutral{suffix}'
    features_path = interim_dir / f'{stem}.features{suffix}'
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

    configure_language_fallback(
        enabled=config.language.fallback_enabled,
        threshold=config.language.confidence_threshold,
        endpoint=config.language.fallback_endpoint,
        api_key_env=config.language.fallback_api_key_env,
    )

    drop_neutral_ratings(
        input_csv=dedup_path,
        output_path=neutral_path,
        rating_column='rating',
        neutral_output_path=neutral_dump_path,
    )

    add_rating_sentiment(
        input_csv=neutral_path,
        output_csv=features_path,
        rating_column='rating',
        sentiment_column='sentiment',
    )

    language_enabled = config.language.enabled
    if language_enabled:
        filter_spanish_comments(
            input_csv=features_path,
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
    source_final = spanish_path if language_enabled else features_path
    shutil.copy2(source_final, final_output)

    row_count = count_csv_rows(final_output)
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


def preprocess_pending_reviews(
    config: PreprocessConfig,
    pattern: str = '*.csv',
) -> list[Path]:
    """Process every raw CSV missing a processed counterpart."""
    raw_dir = config.paths.raw_dir
    processed_dir = config.paths.processed_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pending = list_pending_raw_files(raw_dir, processed_dir, pattern)
    if not pending:
        log.info(
            json_log(
                'preprocess_batch.up_to_date',
                component='pipeline.preprocess',
                raw_dir=str(raw_dir),
                processed_dir=str(processed_dir),
                pattern=pattern,
            ),
        )
        return []

    log.info(
        json_log(
            'preprocess_batch.start',
            component='pipeline.preprocess',
            count=len(pending),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            pattern=pattern,
        ),
    )

    outputs: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for raw_csv in pending:
        try:
            result = preprocess_reviews(raw_csv, config=config)
            outputs.append(result)
        except Exception as exc:  # pragma: no cover - propagate summary below
            failures.append((raw_csv, exc))
            log.error(
                json_log(
                    'preprocess_batch.failure',
                    component='pipeline.preprocess',
                    file=str(raw_csv),
                    error=str(exc),
                ),
            )

    if failures:
        failed_files = ', '.join(str(item[0]) for item in failures)
        raise RuntimeError(
            f'Batch preprocessing failed for: {failed_files}',
        ) from failures[0][1]

    log.info(
        json_log(
            'preprocess_batch.completed',
            component='pipeline.preprocess',
            processed=len(outputs),
            raw_dir=str(raw_dir),
        ),
    )
    return outputs


def run_full_pipeline(
    config: PreprocessConfig,
    raw_pattern: str = '*.csv',
    merge_pattern: str = '*.clean.csv',
    merge_output: Path | None = None,
) -> Path:
    """Run batch preprocessing followed by dataset merge."""
    preprocess_pending_reviews(config=config, pattern=raw_pattern)
    output_path = (
        merge_output
        if merge_output is not None
        else config.paths.processed_dir /'merged'/'merged_dataset.csv'
    )
    merged = merge_processed_csvs(
        processed_dir=config.paths.processed_dir,
        output_path=output_path,
        pattern=merge_pattern,
    )
    log.info(
        json_log(
            'full_pipeline.completed',
            component='pipeline.preprocess',
            merge_output=str(merged),
        ),
    )
    return merged
