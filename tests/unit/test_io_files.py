"""Tests for IO helper utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gym_sentiment_guard.io import (
    count_csv_rows,
    list_csv_files,
    list_pending_raw_files,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_list_csv_files_returns_sorted_matches(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (_ := data_dir / "b.clean.csv").write_text("id\n1\n", encoding="utf-8")
    (_ := data_dir / "a.clean.csv").write_text("id\n1\n", encoding="utf-8")
    (data_dir / "not_csv.txt").write_text("ignore", encoding="utf-8")

    result = list_csv_files(data_dir, pattern="*.clean.csv")
    assert [p.name for p in result] == ["a.clean.csv", "b.clean.csv"]


def test_list_pending_raw_files_detects_missing_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    raw_a = raw_dir / "a.csv"
    raw_b = raw_dir / "b.csv"
    raw_a.write_text("id\n1\n", encoding="utf-8")
    raw_b.write_text("id\n1\n", encoding="utf-8")
    # Only a.csv has processed counterpart.
    (processed_dir / "a.clean.csv").write_text("id\n1\n", encoding="utf-8")

    pending = list_pending_raw_files(raw_dir, processed_dir)
    assert pending == [raw_b]


def test_count_csv_rows_returns_number_of_records(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path, [{"id": 1}, {"id": 2}, {"id": 3}])

    assert count_csv_rows(csv_path) == 3
