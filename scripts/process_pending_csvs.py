"""Process all raw CSVs that do not yet have a .clean version."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_pending_files(raw_dir: Path, processed_dir: Path) -> list[Path]:
    """Return CSVs in raw_dir that do not have a .clean counterpart in processed_dir."""
    pending: list[Path] = []
    for raw_csv in sorted(raw_dir.glob("*.csv")):
        clean_csv = processed_dir / f"{raw_csv.stem}.clean.csv"
        if not clean_csv.exists():
            pending.append(raw_csv)
    return pending


def run_preprocess(raw_csv: Path, config: Path, python_executable: str) -> int:
    """Invoke the existing preprocess CLI for the given raw CSV."""
    cmd = [
        python_executable,
        "-m",
        "gym_sentiment_guard.cli.main",
        "main",
        "preprocess",
        "--input",
        str(raw_csv),
        "--config",
        str(config),
    ]
    result = subprocess.run(cmd, check=False)  # noqa: S603
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw CSVs that do not yet have a .clean file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSVs (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed .clean.csv files (default: data/processed).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocess.yaml"),
        help="Path to preprocess config (default: configs/preprocess.yaml).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    config_path = args.config.resolve()
    python_executable = args.python

    pending_files = find_pending_files(raw_dir, processed_dir)
    if not pending_files:
        print("No pending CSVs found; everything is up-to-date.")
        return

    print(f"Found {len(pending_files)} pending CSV(s):")
    for raw_csv in pending_files:
        print(f"  - {raw_csv}")

    failures: list[Path] = []
    for raw_csv in pending_files:
        print(f"Processing {raw_csv}")
        code = run_preprocess(raw_csv, config_path, python_executable)
        if code != 0:
            failures.append(raw_csv)
            print(f"  ❌ Failed with exit code {code}")
        else:
            print("  ✅ Completed")

    if failures:
        print("Finished with failures:")
        for raw_csv in failures:
            print(f"  - {raw_csv}")
        sys.exit(1)
    else:
        print("All pending CSVs processed successfully.")


if __name__ == "__main__":
    main()
