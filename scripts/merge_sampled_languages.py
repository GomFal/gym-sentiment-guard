"""Merge all sampled language CSVs into a single evaluation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_sampled_files(
    root_dir: str | Path,
    pattern: str = "*.sampled.csv",
    output: str = "merged_sampled_ground_truth.csv",
) -> Path:
    """Merge all sampled CSVs under root into a single CSV."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    frames = []
    for csv_path in root.rglob(pattern):
        if csv_path.is_file():
            frames.append(pd.read_csv(csv_path))

    if not frames:
        raise ValueError(f"No files matching pattern '{pattern}' found under {root}")

    merged = pd.concat(frames, ignore_index=True)
    output_path = root / output
    merged.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all sampled language files into a single CSV.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing language sampled files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.sampled.csv",
        help="Glob pattern for sampled files (default: *.sampled.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_sampled_ground_truth.csv",
        help="Output filename for the merged CSV.",
    )
    args = parser.parse_args()

    output_path = merge_sampled_files(args.root, pattern=args.pattern, output=args.output)
    print(f"Merged sampled dataset written to: {output_path}")


if __name__ == "__main__":
    main()
