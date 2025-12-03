"""Command-line interface for gym_sentiment_guard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from ..config import PreprocessConfig, load_preprocess_config
from ..data import merge_processed_csvs
from ..pipeline import (
    preprocess_pending_reviews,
    preprocess_reviews,
)
from ..pipeline import (
    run_full_pipeline as pipeline_run_full_pipeline,
)
from ..utils import get_logger, json_log

app = typer.Typer(help='Gym Sentiment Guard CLI', no_args_is_help=True)
pipeline_app = typer.Typer(help='Pipeline commands', no_args_is_help=True)
app.add_typer(pipeline_app, name='main')

log = get_logger(__name__)


def _resolve_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _apply_path_overrides(
    config: PreprocessConfig,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> PreprocessConfig:
    overrides: dict[str, Path] = {}
    if raw_dir is not None:
        overrides['raw_dir'] = _resolve_path(raw_dir)  # type: ignore[assignment]
    if processed_dir is not None:
        overrides['processed_dir'] = _resolve_path(processed_dir)  # type: ignore[assignment]
    if not overrides:
        return config
    new_paths = replace(config.paths, **overrides)
    return replace(config, paths=new_paths)


@pipeline_app.command('preprocess')
def preprocess(
    input_csv: Annotated[
        Path,
        typer.Option(
            '--input',
            '-i',
            exists=True,
            readable=True,
            help='Path to raw CSV file.',
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to preprocess configuration YAML.',
        ),
    ] = Path('configs/preprocess.yaml'),
    output: Annotated[
        Path | None,
        typer.Option(
            '--output',
            '-o',
            help='Optional explicit output CSV path. Defaults to configs.processed_dir/<name>.clean.csv',
        ),
    ] = None,
) -> None:
    """Run the preprocessing pipeline on the given CSV."""
    log.info(
        json_log(
            'cli.preprocess.start',
            component='cli',
            input=str(input_csv),
            config=str(config),
            output_override=str(output) if output else None,
        ),
    )
    cfg = load_preprocess_config(config)
    result_path = preprocess_reviews(
        input_path=input_csv,
        config=cfg,
        output_path=output,
    )

    log.info(
        json_log(
            'cli.preprocess.completed',
            component='cli',
            output=str(result_path),
        ),
    )
    typer.echo(f'Processed data written to: {result_path}')


@pipeline_app.command('preprocess-batch')
def preprocess_batch(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to preprocess configuration YAML.',
        ),
    ] = Path('configs/preprocess.yaml'),
    pattern: Annotated[
        str,
        typer.Option(
            '--pattern',
            '-p',
            help='Glob pattern for raw CSVs (default: *.csv).',
        ),
    ] = '*.csv',
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            '--raw-dir',
            help='Optional override for raw directory.',
        ),
    ] = None,
    processed_dir: Annotated[
        Path | None,
        typer.Option(
            '--processed-dir',
            help='Optional override for processed directory.',
        ),
    ] = None,
) -> None:
    """Process all pending raw CSV files."""
    log.info(
        json_log(
            'cli.preprocess_batch.start',
            component='cli',
            config=str(config),
            pattern=pattern,
        ),
    )
    cfg = load_preprocess_config(config)
    cfg = _apply_path_overrides(cfg, raw_dir=raw_dir, processed_dir=processed_dir)
    outputs = preprocess_pending_reviews(cfg, pattern=pattern)
    if not outputs:
        typer.echo('No pending CSVs found; everything is up-to-date.')
    else:
        typer.echo(f'Processed {len(outputs)} file(s).')
    log.info(
        json_log(
            'cli.preprocess_batch.completed',
            component='cli',
            processed=len(outputs),
        ),
    )


@pipeline_app.command('merge-processed')
def merge_processed(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to preprocess configuration YAML.',
        ),
    ] = Path('configs/preprocess.yaml'),
    processed_dir: Annotated[
        Path | None,
        typer.Option('--processed-dir', help='Optional override for processed dir.'),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option('--output', '-o', help='Destination merged CSV path.'),
    ] = None,
    pattern: Annotated[
        str,
        typer.Option('--pattern', help='Glob pattern for processed CSVs.'),
    ] = '*.clean.csv',
) -> None:
    """Merge processed CSVs into a training dataset."""
    cfg = load_preprocess_config(config)
    cfg = _apply_path_overrides(cfg, processed_dir=processed_dir)
    target_processed = cfg.paths.processed_dir
    target_output = (
        _resolve_path(output)
        if output is not None
        else target_processed / 'train_dataset.csv'
    )
    merged = merge_processed_csvs(
        processed_dir=target_processed,
        output_path=target_output,
        pattern=pattern,
    )
    log.info(
        json_log(
            'cli.merge.completed',
            component='cli',
            output=str(merged),
        ),
    )
    typer.echo(f'Merged dataset written to: {merged}')


@pipeline_app.command('run-full-pipeline')
def run_full_pipeline(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to preprocess configuration YAML.',
        ),
    ] = Path('configs/preprocess.yaml'),
    raw_pattern: Annotated[
        str,
        typer.Option('--raw-pattern', help='Glob for raw CSVs (default: *.csv).'),
    ] = '*.csv',
    merge_pattern: Annotated[
        str,
        typer.Option(
            '--merge-pattern',
            help='Glob for processed CSVs (default: *.clean.csv).',
        ),
    ] = '*.clean.csv',
    merge_output: Annotated[
        Path | None,
        typer.Option('--merge-output', help='Merged dataset path override.'),
    ] = None,
    raw_dir: Annotated[
        Path | None,
        typer.Option('--raw-dir', help='Optional override for raw directory.'),
    ] = None,
    processed_dir: Annotated[
        Path | None,
        typer.Option('--processed-dir', help='Optional override for processed dir.'),
    ] = None,
) -> None:
    """Run batch preprocessing and merge into a single dataset."""
    cfg = load_preprocess_config(config)
    cfg = _apply_path_overrides(cfg, raw_dir=raw_dir, processed_dir=processed_dir)
    target_output = (
        _resolve_path(merge_output)
        if merge_output is not None
        else cfg.paths.processed_dir / 'train_dataset.csv'
    )
    merged = pipeline_run_full_pipeline(
        config=cfg,
        raw_pattern=raw_pattern,
        merge_pattern=merge_pattern,
        merge_output=target_output,
    )
    typer.echo(f'Full pipeline completed. Merged dataset: {merged}')
if __name__ == '__main__':
    app()
