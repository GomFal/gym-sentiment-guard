"""Utilities to combine processed CSV files into a single dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gym_sentiment_guard.io import list_csv_files
from gym_sentiment_guard.utils.logging import get_logger, json_log

log = get_logger(__name__)


def merge_processed_csvs(
    processed_dir: str | Path,
    output_path: str | Path,
    pattern: str = "*.clean.csv",
) -> Path:
    """
    Merge processed CSVs into a single dataset.

    Args:
        processed_dir: Directory containing processed CSVs.
        output_path: Destination path for merged dataset.
        pattern: Glob pattern to match processed files.

    Returns:
        Path to the merged dataset.
    """
    processed = Path(processed_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_files(processed, pattern)
    if not csv_files:
        log.error(
            json_log(
                "merge.no_files_found",
                component="data.merge",
                processed_dir=str(processed),
                pattern=pattern,
            )
        )
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {processed_dir}."
        )

    frames: list[pd.DataFrame] = []
    total_rows = 0
    reference_columns: tuple[str, ...] | None = None
    for file in csv_files:
        df = pd.read_csv(file)
        columns = tuple(df.columns)
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            log.error(
                json_log(
                    "merge.schema_mismatch",
                    component="data.merge",
                    file=str(file),
                    expected=list(reference_columns),
                    actual=list(columns),
                )
            )
            raise ValueError(
                f"Schema mismatch in {file}. Expected columns {reference_columns}, "
                f"got {columns}."
            )
        frames.append(df)
        total_rows += len(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output, index=False)

    log.info(
        json_log(
            "merge.completed",
            component="data.merge",
            processed_dir=str(processed),
            output=str(output),
            files=len(csv_files),
            rows=int(total_rows),
        )
    )
    return output
