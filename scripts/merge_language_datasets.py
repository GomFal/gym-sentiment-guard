"""Merge language-specific CSVs into one file per language."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Change to merge only certain folders. E.g: LANGUAGE_DIRS = {"it"}
LANGUAGE_DIRS = {'es', 'pt', 'it', 'en'}


def merge_language_folder(folder: Path, output_name: str = 'merged.csv') -> Path | None:
    """Merge CSVs inside a language folder, keeping only rows matching that language."""
    if folder.name.lower() not in LANGUAGE_DIRS:
        return None

    csv_files = list(folder.glob('*.csv'))
    if not csv_files:
        return None

    lang_code = folder.name.lower()
    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if 'language' not in df.columns:
            raise ValueError(f"'language' column missing in {csv_path}")
        filtered = df[df['language'].str.lower() == lang_code]
        frames.append(filtered)

    merged = pd.concat(frames, ignore_index=True)
    output_path = folder / output_name
    merged.to_csv(output_path, index=False)
    return output_path


def merge_language_folders(root_dir: str | Path, output_name: str = 'merged.csv') -> list[Path]:
    """Iterate through language folders under root and merge each one."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'Root directory not found: {root}')

    merged_paths: list[Path] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        merged_path = merge_language_folder(folder, output_name=output_name)
        if merged_path:
            merged_paths.append(merged_path)
    return merged_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge per-language CSVs into one file per language folder.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        required=True,
        help='Root directory containing language folders (es, en, etc.).',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='merged.csv',
        help='File name to use for merged output inside each language folder.',
    )
    args = parser.parse_args()
    merged = merge_language_folders(args.root, output_name=args.output_name)
    print(f'Merged {len(merged)} language folder(s).')


if __name__ == '__main__':
    main()
