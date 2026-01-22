"""Annotate evaluation CSVs with a ground-truth language column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LANGUAGE_DIRS = {'es', 'pt', 'it', 'en'}


def annotate_language_columns(root_dir: str | Path) -> list[Path]:
    """
    Walk subdirectories under ``root_dir`` and inject a ``language`` column.
    Only files ending in the suffix *_non_processed.csv will be annotated.

    Args:
        root_dir: Base directory containing language-specific subfolders.

    Returns:
        List of CSV paths that were updated.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'Root directory not found: {root}')

    updated_files: list[Path] = []

    for language_dir in root.iterdir():
        if not language_dir.is_dir():
            continue
        language_code = language_dir.name.lower()
        if language_code not in LANGUAGE_DIRS:
            continue
        for csv_path in language_dir.glob('*_non_processed.csv'):
            df = pd.read_csv(csv_path)
            df['language'] = language_code
            csv_path = csv_path.with_name(csv_path.name.replace('_non_processed', ''))
            df.to_csv(csv_path, index=False)
            updated_files.append(csv_path)
    return updated_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Annotate CSVs with ground-truth language codes.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        required=True,
        help='Root directory containing language-specific folders (es, pt, etc.).',
    )
    args = parser.parse_args()
    updated = annotate_language_columns(args.root)
    print(f'Annotated {len(updated)} file(s).')


if __name__ == '__main__':
    main()
