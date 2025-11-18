"""Sample a fixed number of reviews per language file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_language_file(
    csv_path: Path,
    n_samples: int,
    output_suffix: str = ".sampled.csv",
    random_state: int = 42,
) -> Path:
    """Sample up to ``n_samples`` rows from ``csv_path`` and save to a new file."""
    df = pd.read_csv(csv_path)
    sample_size = min(n_samples, len(df))
    sampled = df.sample(n=sample_size, random_state=random_state)
    output_path = csv_path.with_name(csv_path.stem + output_suffix)
    sampled.to_csv(output_path, index=False)
    return output_path


def sample_language_reviews(
    root_dir: str | Path,
    pattern: str = "*.clean.csv",
    n_samples: int = 250,
    output_suffix: str = ".sampled.csv",
) -> list[Path]:
    """Sample language review files under ``root_dir`` and return output paths."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    outputs: list[Path] = []
    for csv_path in root.rglob(pattern):
        if csv_path.is_file():
            output_path = sample_language_file(
                csv_path,
                n_samples=n_samples,
                output_suffix=output_suffix,
            )
            outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample reviews from each language's clean CSV.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing language .clean.csv files.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=250,
        help="Number of reviews to sample per file (default: 250).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.clean.csv",
        help="Glob pattern to match files (default: *.clean.csv).",
    )
    args = parser.parse_args()

    outputs = sample_language_reviews(
        args.root,
        pattern=args.pattern,
        n_samples=args.samples,
    )
    print(f"Sampled {len(outputs)} file(s).")


if __name__ == "__main__":
    main()
