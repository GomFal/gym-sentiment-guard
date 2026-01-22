"""Thin wrapper that invokes the Typer CLI batch preprocess command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Process raw CSVs via the official CLI batch command.',
    )
    parser.add_argument(
        '--raw-dir',
        type=Path,
        help='Optional override for raw directory (defaults to config value).',
    )
    parser.add_argument(
        '--processed-dir',
        type=Path,
        help='Optional override for processed directory (defaults to config value).',
    )
    parser.add_argument(
        '--pattern',
        default='*.csv',
        help='Glob pattern for raw CSVs (default: *.csv).',
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
        'preprocess-batch',
        '--config',
        str(args.config),
        '--pattern',
        args.pattern,
    ]
    if args.raw_dir:
        cmd += ['--raw-dir', str(args.raw_dir)]
    if args.processed_dir:
        cmd += ['--processed-dir', str(args.processed_dir)]

    raise_on_error = subprocess.run(cmd, check=False)  # noqa: S603
    sys.exit(raise_on_error.returncode)


if __name__ == '__main__':
    main()
