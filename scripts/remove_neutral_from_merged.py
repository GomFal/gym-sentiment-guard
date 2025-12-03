"""Helper to drop neutral (rating==3) rows from a merged dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from gym_sentiment_guard.data.cleaning import drop_neutral_ratings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove rating=3 rows from a merged dataset.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/merged/merged_dataset.csv"),
        help="Path to the merged dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train_dataset.non_neutral.csv"),
        help="Destination for the filtered dataset.",
    )
    parser.add_argument(
        "--neutral-output",
        type=Path,
        default=Path("data/processed/train_dataset.neutral.csv"),
        help="Where to store the dropped neutral rows.",
    )
    parser.add_argument(
        "--rating-column",
        default="rating",
        help="Column containing rating values (default: rating).",
    )
    parser.add_argument(
        "--neutral-value",
        type=float,
        default=3,
        help="Neutral rating value to drop (default: 3).",
    )
    args = parser.parse_args()

    drop_neutral_ratings(
        input_csv=args.input,
        output_path=args.output,
        rating_column=args.rating_column,
        neutral_value=args.neutral_value,
        neutral_output_path=args.neutral_output,
    )
    print(f"Filtered dataset written to {args.output}")
    print(f"Neutral rows written to {args.neutral_output}")


if __name__ == "__main__":
    main()
