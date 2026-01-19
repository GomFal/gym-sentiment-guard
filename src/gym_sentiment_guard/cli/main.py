"""Command-line interface for gym_sentiment_guard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from ..config import PreprocessConfig, load_preprocess_config
from ..data import (
    load_structural_punctuation,
    merge_processed_csvs,
    normalize_comments,
    split_dataset,
)
from ..pipeline import (
    preprocess_pending_reviews,
    preprocess_reviews,
)
from ..pipeline import (
    run_full_pipeline as pipeline_run_full_pipeline,
)
from ..training import train_from_config
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


@pipeline_app.command('split-data')
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


@pipeline_app.command('normalize-dataset')
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


@pipeline_app.command('train-model')
def train_model(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to model config YAML.',
        ),
    ] = Path('configs/logreg_v1.yaml'),
) -> None:
    """Train a sentiment model from a config file."""
    result = train_from_config(config)
    typer.echo(f'Model trained. Artifacts in {result["artifact_dir"]}')


@pipeline_app.command('run-experiment')
def run_experiment(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to experiment config YAML.',
        ),
    ] = Path('configs/experiment.yaml'),
    ngram_min: Annotated[
        int,
        typer.Option('--ngram-min', help='Minimum n-gram size (default: 1).'),
    ] = 1,
    ngram_max: Annotated[
        int,
        typer.Option('--ngram-max', help='Maximum n-gram size (default: 2).'),
    ] = 2,
    min_df: Annotated[
        int,
        typer.Option('--min-df', help='Minimum document frequency (default: 2).'),
    ] = 2,
    max_df: Annotated[
        float,
        typer.Option('--max-df', help='Maximum document frequency (default: 1.0).'),
    ] = 1.0,
    sublinear_tf: Annotated[
        bool,
        typer.Option('--sublinear-tf/--no-sublinear-tf', help='Use sublinear TF scaling.'),
    ] = True,
    C: Annotated[
        float,
        typer.Option('--C', help='Regularization strength (default: 1.0).'),
    ] = 1.0,
    penalty: Annotated[
        str,
        typer.Option('--penalty', help='Regularization type: l1 or l2 (default: l2).'),
    ] = 'l2',
    class_weight: Annotated[
        str | None,
        typer.Option('--class-weight', help='Class weight: balanced or None.'),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option('--output-dir', help='Output directory for artifacts.'),
    ] = Path('artifacts/experiments'),
) -> None:
    """Run a single experiment with specified hyperparameters."""
    import yaml

    from ..experiments import ExperimentConfig, run_single_experiment

    # Load config for data paths
    with open(config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    splits = data.get('splits', {})

    exp_config = ExperimentConfig(
        train_path=splits.get('train', ''),
        val_path=splits.get('val', ''),
        test_path=splits.get('test', ''),
        text_column=data.get('text_column', 'comment'),
        label_column=data.get('label_column', 'sentiment'),
        label_mapping=data.get('label_mapping', {'negative': 0, 'positive': 1}),
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        C=C,
        penalty=penalty,
        class_weight=class_weight if class_weight != 'None' else None,
        recall_constraint=cfg.get('selection', {}).get('recall_constraint', 0.90),
        output_dir=str(output_dir),
    )

    log.info(
        json_log(
            'cli.run_experiment.start',
            component='cli',
            ngram_range=(ngram_min, ngram_max),
            C=C,
            penalty=penalty,
        ),
    )

    result = run_single_experiment(exp_config)

    if result.val_metrics:
        typer.echo(f'Run ID: {result.config.run_id}')
        typer.echo(f'F1_neg: {result.val_metrics.f1_neg:.4f}')
        typer.echo(f'Recall_neg: {result.val_metrics.recall_neg:.4f}')
        typer.echo(f'Threshold: {result.val_metrics.threshold:.4f}')
        typer.echo(f'Constraint: {result.val_metrics.constraint_status}')
        typer.echo(f'Validity: {result.validity_status}')
    else:
        typer.echo(f'Run failed: {result.invalidity_reason}')

    log.info(
        json_log(
            'cli.run_experiment.completed',
            component='cli',
            run_id=result.config.run_id,
            validity_status=result.validity_status,
        ),
    )


@pipeline_app.command('run-ablation')
def run_ablation(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to experiment config YAML.',
        ),
    ] = Path('configs/experiment.yaml'),
    max_runs: Annotated[
        int | None,
        typer.Option('--max-runs', help='Maximum number of runs (for testing).'),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option('--output-dir', help='Output directory for artifacts.'),
    ] = Path('artifacts/experiments'),
) -> None:
    """Run full ablation suite from config grids."""
    import yaml

    from ..experiments import ExperimentConfig, run_ablation_suite

    # Load config
    with open(config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    splits = data.get('splits', {})
    ablation = cfg.get('ablation', {})
    tfidf_cfg = ablation.get('tfidf', {})
    logreg_cfg = ablation.get('logreg', {})

    # Build grids from config
    tfidf_grid = {
        'ngram_range': [tuple(x) for x in tfidf_cfg.get('ngram_range', [[1, 1]])],
        'min_df': tfidf_cfg.get('min_df', [2]),
        'max_df': tfidf_cfg.get('max_df', [1.0]),
        'sublinear_tf': tfidf_cfg.get('sublinear_tf', [True]),
        'stop_words': tfidf_cfg.get('stop_words', [None]),  # None or 'curated_safe'
    }
    logreg_grid = {
        'penalty': logreg_cfg.get('penalty', ['l2']),
        'C': logreg_cfg.get('C', [1.0]),
        'class_weight': logreg_cfg.get('class_weight', [None]),
    }

    base_config = ExperimentConfig(
        train_path=splits.get('train', ''),
        val_path=splits.get('val', ''),
        test_path=splits.get('test', ''),
        text_column=data.get('text_column', 'comment'),
        label_column=data.get('label_column', 'sentiment'),
        label_mapping=data.get('label_mapping', {'negative': 0, 'positive': 1}),
        recall_constraint=cfg.get('selection', {}).get('recall_constraint', 0.90),
        output_dir=str(output_dir),
    )

    log.info(
        json_log(
            'cli.run_ablation.start',
            component='cli',
            tfidf_grid_size=len(list(tfidf_grid.values())),
            logreg_grid_size=len(list(logreg_grid.values())),
        ),
    )

    suite = run_ablation_suite(
        base_config=base_config,
        tfidf_grid=tfidf_grid,
        logreg_grid=logreg_grid,
        max_runs=max_runs,
    )

    winner = suite.get('winner')
    summary = suite.get('summary', {})

    typer.echo(f'Ablation complete!')
    typer.echo(f'Total runs: {summary.get("total_runs", 0)}')
    typer.echo(f'Valid runs: {summary.get("valid_runs", 0)}')

    if winner and winner.val_metrics:
        typer.echo(f'\nWinner: {winner.config.run_id}')
        typer.echo(f'  F1_neg: {winner.val_metrics.f1_neg:.4f}')
        typer.echo(f'  Recall_neg: {winner.val_metrics.recall_neg:.4f}')
        typer.echo(f'  Threshold: {winner.val_metrics.threshold:.4f}')

    typer.echo(f'\nSuite artifacts: {suite.get("suite_dir", "")}')

    log.info(
        json_log(
            'cli.run_ablation.completed',
            component='cli',
            total_runs=summary.get('total_runs', 0),
            winner_run_id=summary.get('winner_run_id'),
        ),
    )


@pipeline_app.command('ablation-report')
def ablation_report(
    experiments_dir: Annotated[
        Path,
        typer.Option(
            '--experiments-dir',
            '-e',
            exists=True,
            readable=True,
            help='Path to experiments directory containing run.* folders.',
        ),
    ] = Path('artifacts/experiments'),
    output_dir: Annotated[
        Path,
        typer.Option(
            '--output',
            '-o',
            help='Output directory for reports.',
        ),
    ] = Path('reports/logreg_ablations'),
    test_predictions: Annotated[
        Path | None,
        typer.Option(
            '--test-predictions',
            '-t',
            exists=True,
            readable=True,
            help='Path to test_predictions.csv for Layer 4 PR curves.',
        ),
    ] = None,
    winner_run_id: Annotated[
        str | None,
        typer.Option(
            '--winner',
            '-w',
            help='Explicit winner run_id (auto-detect if omitted).',
        ),
    ] = None,
) -> None:
    """Generate ablation suite report with visualizations."""
    from ..reports import generate_ablation_report as gen_report

    log.info(
        json_log(
            'cli.ablation_report.start',
            component='cli',
            experiments_dir=str(experiments_dir),
            output_dir=str(output_dir),
            test_predictions=str(test_predictions) if test_predictions else None,
        ),
    )

    artifacts = gen_report(
        experiments_dir=experiments_dir,
        output_dir=output_dir,
        test_predictions_path=test_predictions,
        winner_run_id=winner_run_id,
    )

    typer.echo(f'Ablation report generated!')
    typer.echo(f'Output directory: {output_dir}')
    typer.echo(f'Artifacts generated: {len(artifacts)}')

    for name, path in artifacts.items():
        typer.echo(f'  - {name}: {path}')

    log.info(
        json_log(
            'cli.ablation_report.completed',
            component='cli',
            n_artifacts=len(artifacts),
        ),
    )


@pipeline_app.command('error-analysis')
def error_analysis(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to error_analysis.yaml config.',
        ),
    ] = Path('configs/error_analysis.yaml'),
    output_dir: Annotated[
        Path | None,
        typer.Option(
            '--output',
            '-o',
            help='Output directory. Auto-derived from model path if not specified.',
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option(
            '--model',
            '-m',
            exists=True,
            readable=True,
            help='Override model path from config.',
        ),
    ] = None,
    predictions_path: Annotated[
        Path | None,
        typer.Option(
            '--predictions',
            '-p',
            exists=True,
            readable=True,
            help='Override predictions CSV path from config.',
        ),
    ] = None,
    test_csv_path: Annotated[
        Path | None,
        typer.Option(
            '--test-csv',
            '-t',
            exists=True,
            readable=True,
            help='Override test CSV path from config.',
        ),
    ] = None,
) -> None:
    """Run post-training error analysis on model predictions."""
    import yaml

    from ..reports.logreg_errors import run_error_analysis as run_analysis

    # Load config to get model path if not specified
    config_data = yaml.safe_load(config.read_text(encoding='utf-8'))
    model_path_resolved = Path(model_path or config_data['model_path'])

    # Auto-derive output directory from model path if not specified
    if output_dir is None:
        # Extract model_id from parent directory (e.g., "model.2026-01-10_002")
        model_id = model_path_resolved.parent.name
        output_dir = Path('reports/error_analysis') / model_id

    log.info(
        json_log(
            'cli.error_analysis.start',
            component='cli',
            config=str(config),
            output_dir=str(output_dir),
        ),
    )

    artifacts = run_analysis(
        config_path=config,
        output_dir=output_dir,
        model_path=model_path,
        predictions_path=predictions_path,
        test_csv_path=test_csv_path,
    )

    typer.echo('Error analysis completed!')
    typer.echo(f'Output directory: {output_dir}')
    typer.echo(f'Artifacts generated: {len(artifacts)}')

    for name, path in artifacts.items():
        typer.echo(f'  - {name}: {path}')

    log.info(
        json_log(
            'cli.error_analysis.completed',
            component='cli',
            n_artifacts=len(artifacts),
        ),
    )


if __name__ == '__main__':
    app()

