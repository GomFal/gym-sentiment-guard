"""Thin wrapper that delegates dataset merging to the Typer CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge processed CSVs via the official CLI command.',
    )
    parser.add_argument(
        '--processed-dir',
        type=Path,
        help='Optional override for processed directory (defaults to config value).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for the merged dataset.',
    )
    parser.add_argument(
        '--pattern',
        default='*.clean.csv',
        help='Glob pattern for processed CSVs (default: *.clean.csv).',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/preprocess.yaml'),
        help='Path to preprocess config (default: configs/preprocess.yaml).',
    )
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable to use (default: current interpreter).',
    )
    args = parser.parse_args()

    cmd = [
        str(args.python),
        '-m',
        'gym_sentiment_guard.cli.main',
        'main',
        'merge-processed',
        '--config',
        str(args.config),
        '--pattern',
        args.pattern,
    ]
    if args.processed_dir:
        cmd += ['--processed-dir', str(args.processed_dir)]
    if args.output:
        cmd += ['--output', str(args.output)]

    result = subprocess.run(cmd, check=False)  # noqa: S603
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
