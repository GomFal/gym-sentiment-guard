"""Pipeline commands for data curation and preprocessing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

# NOTE: Heavy imports (config, data, pipeline, utils) are lazy-loaded inside
# command handlers to speed up CLI startup time for --help and metadata ops.

app = typer.Typer(help='Data pipeline commands', no_args_is_help=True)


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


@app.command('preprocess')
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
    # Lazy imports for faster CLI startup
    from ..config import load_preprocess_config
    from ..pipeline import preprocess_reviews
    from ..utils import get_logger, json_log

    log = get_logger(__name__)
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


@app.command('batch')
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
    # Lazy imports for faster CLI startup
    from ..config import load_preprocess_config
    from ..pipeline import preprocess_pending_reviews
    from ..utils import get_logger, json_log

    log = get_logger(__name__)
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


@app.command('merge')
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
    # Lazy imports for faster CLI startup
    from ..config import load_preprocess_config
    from ..data import merge_processed_csvs
    from ..utils import get_logger, json_log

    log = get_logger(__name__)
    cfg = load_preprocess_config(config)
    cfg = _apply_path_overrides(cfg, processed_dir=processed_dir)
    target_processed = cfg.paths.processed_dir
    target_output = (
        _resolve_path(output)
        if output is not None
        else target_processed / 'merged' / 'merged_dataset.csv'
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


@app.command('split')
def split_data(
    input_csv: Annotated[
        Path,
        typer.Option(
            '--input',
            '-i',
            exists=True,
            readable=True,
            help='Path to merged dataset CSV.',
        ),
    ] = Path('data/processed/merged/merged_dataset.csv'),
    output_dir: Annotated[
        Path,
        typer.Option('--output-dir', help='Directory to store splits.'),
    ] = Path('data/processed/splits'),
    column: Annotated[
        str,
        typer.Option('--column', help='Column to stratify on.'),
    ] = 'sentiment',
    train_ratio: Annotated[
        float,
        typer.Option('--train-ratio', help='Train split ratio (default 0.7).'),
    ] = 0.7,
    val_ratio: Annotated[
        float,
        typer.Option('--val-ratio', help='Validation split ratio (default 0.15).'),
    ] = 0.15,
    test_ratio: Annotated[
        float,
        typer.Option('--test-ratio', help='Test split ratio (default 0.15).'),
    ] = 0.15,
    random_state: Annotated[
        int,
        typer.Option('--random-state', help='Random seed for splitting.'),
    ] = 42,
) -> None:
    """Split a dataset into train/val/test CSVs."""
    # Lazy imports for faster CLI startup
    from ..data import split_dataset

    split_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        stratify_column=column,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    typer.echo(f'Splits written to: {output_dir}')


@app.command('run')
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
    split_output: Annotated[
        Path | None,
        typer.Option('--split-output', help='Directory to store dataset splits.'),
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
    # Lazy imports for faster CLI startup
    from ..config import load_preprocess_config
    from ..pipeline import run_full_pipeline as pipeline_run_full_pipeline

    cfg = load_preprocess_config(config)
    cfg = _apply_path_overrides(cfg, raw_dir=raw_dir, processed_dir=processed_dir)
    target_output = (
        _resolve_path(merge_output)
        if merge_output is not None
        else cfg.paths.processed_dir / 'merged' / 'merged_dataset.csv'
    )
    split_dir = (
        _resolve_path(split_output)
        if split_output is not None
        else cfg.paths.processed_dir / 'splits'
    )
    merged = pipeline_run_full_pipeline(
        config=cfg,
        raw_pattern=raw_pattern,
        merge_pattern=merge_pattern,
        merge_output=target_output,
        split_output_dir=split_dir,
    )
    typer.echo(f'Full pipeline completed. Merged dataset: {merged}')


@app.command('normalize')
def normalize_dataset(
    input_csv: Annotated[
        Path,
        typer.Option(
            '--input',
            '-i',
            exists=True,
            readable=True,
            help='CSV file to normalize.',
        ),
    ],
    output_csv: Annotated[
        Path | None,
        typer.Option(
            '--output',
            '-o',
            help='Output path (default: <input>.normalized.csv).',
        ),
    ] = None,
    column: Annotated[
        str,
        typer.Option('--column', help='Text column to normalize (default: comment).'),
    ] = 'comment',
    punctuation_file: Annotated[
        Path | None,
        typer.Option('--punctuation-file', help='Optional structural punctuation file.'),
    ] = None,
) -> None:
    """Normalize a dataset without running the entire pipeline."""
    # Lazy imports for faster CLI startup
    from ..data import load_structural_punctuation, normalize_comments

    target_output = (
        _resolve_path(output_csv)
        if output_csv is not None
        else Path(input_csv).with_suffix('.normalized.csv')
    )
    normalize_comments(
        input_csv=input_csv,
        output_path=target_output,
        text_column=column,
        structural_punctuation=load_structural_punctuation(punctuation_file),
    )
    typer.echo(f'Normalized dataset written to: {target_output}')
