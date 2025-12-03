"""File utilities for listing and inspecting CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_csv_files(directory: Path, pattern: str = '*.csv') -> list[Path]:
    """Return sorted CSV files matching pattern inside directory."""
    directory.mkdir(parents=True, exist_ok=True)
    files = sorted(directory.glob(pattern))
    return [file for file in files if file.is_file()]


def list_pending_raw_files(
    raw_dir: Path,
    processed_dir: Path,
    pattern: str = '*.csv',
) -> list[Path]:
    """Return raw CSVs that lack a processed .clean counterpart."""
    raw_files = list_csv_files(raw_dir, pattern=pattern)
    pending: list[Path] = []
    for raw_csv in raw_files:
        suffix = raw_csv.suffix or '.csv'
        clean_candidate = processed_dir / f'{raw_csv.stem}.clean{suffix}'
        if not clean_candidate.exists():
            pending.append(raw_csv)
    return pending


def count_csv_rows(csv_path: Path) -> int:
    """Return the number of rows in a CSV file."""
    df = pd.read_csv(csv_path)
    return int(len(df))
