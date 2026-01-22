"""Logistic Regression specific commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

# NOTE: Heavy imports (training, experiments, reports, utils) are lazy-loaded
# inside command handlers to speed up CLI startup time for --help and metadata ops.

app = typer.Typer(help='LogReg model commands', no_args_is_help=True)


@app.command('train')
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
    ] = Path('configs/logreg/training_v1.yaml'),
) -> None:
    """Train a sentiment model from a config file."""
    # Lazy imports for faster CLI startup
    from ..models.logreg.training import train_from_config

    result = train_from_config(config)
    typer.echo(f'Model trained. Artifacts in {result["artifact_dir"]}')


@app.command('experiment')
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
    ] = Path('configs/logreg/experiment.yaml'),
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
    ] = Path('artifacts/experiments/logreg'),
) -> None:
    """Run a single experiment with specified hyperparameters."""
    # Lazy imports for faster CLI startup
    from ..models.logreg.experiments import ExperimentConfig, run_single_experiment
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

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


@app.command('ablation')
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
    ] = Path('configs/logreg/experiment.yaml'),
    max_runs: Annotated[
        int | None,
        typer.Option('--max-runs', help='Maximum number of runs (for testing).'),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option('--output-dir', help='Output directory for artifacts.'),
    ] = Path('artifacts/experiments/logreg'),
) -> None:
    """Run full ablation suite from config grids."""
    # Lazy imports for faster CLI startup
    from ..models.logreg.experiments import ExperimentConfig, run_ablation_suite
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

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

    typer.echo('Ablation complete!')
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


@app.command('ablation-report')
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
    ] = Path('artifacts/experiments/logreg'),
    output_dir: Annotated[
        Path,
        typer.Option(
            '--output',
            '-o',
            help='Output directory for reports.',
        ),
    ] = Path('reports/logreg/ablations'),
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
    # Lazy imports for faster CLI startup
    from ..models.logreg.reports import generate_ablation_report
    from ..utils import get_logger, json_log

    log = get_logger(__name__)
    log.info(
        json_log(
            'cli.ablation_report.start',
            component='cli',
            experiments_dir=str(experiments_dir),
            output_dir=str(output_dir),
            test_predictions=str(test_predictions) if test_predictions else None,
        ),
    )

    artifacts = generate_ablation_report(
        experiments_dir=experiments_dir,
        output_dir=output_dir,
        test_predictions_path=test_predictions,
        winner_run_id=winner_run_id,
    )

    typer.echo('Ablation report generated!')
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


@app.command('error-analysis')
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
    ] = Path('configs/logreg/error_analysis.yaml'),
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
    # Lazy imports for faster CLI startup
    from ..models.logreg.reports import run_error_analysis as run_analysis
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

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
