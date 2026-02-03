"""SVM model CLI commands.

Provides commands for SVM Linear training, experiments, and ablation studies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

app = typer.Typer(name='svm', help='SVM Linear model commands')


@app.command('train')
def train_model(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to training config YAML.',
        ),
    ] = Path('configs/svm/training.yaml'),
) -> None:
    """Train an SVM model from a config file."""
    # Lazy imports for faster CLI startup
    from ..models.svm.training import train_from_config

    result = train_from_config(config)
    typer.echo(f'Model trained. Artifacts in {result["artifact_dir"]}')


@app.command('train-rbf')
def train_rbf_model(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            exists=True,
            readable=True,
            help='Path to RBF training config YAML.',
        ),
    ] = Path('configs/svm/training_rbf.yaml'),
) -> None:
    """Train an SVM RBF model from a config file."""
    # Lazy imports for faster CLI startup
    from ..models.svm.training import train_rbf_from_config

    result = train_rbf_from_config(config)
    typer.echo(f'RBF Model trained. Artifacts in {result["artifact_dir"]}')


@app.command()
def experiment(
    config: Path = typer.Option(
        Path('configs/svm/experiment.yaml'),
        '--config',
        '-c',
        help='Path to experiment config file',
    ),
    C: float = typer.Option(1.0, '--C', help='Regularization strength'),
    intercept_scaling: float = typer.Option(1.0, '--intercept-scaling', help='Intercept scaling'),
    tol: float = typer.Option(1e-4, '--tol', help='Convergence tolerance'),
    max_iter: int = typer.Option(2000, '--max-iter', help='Maximum iterations'),
    output_dir: Path = typer.Option(
        Path('artifacts/experiments/svm_linear'),
        '--output-dir',
        '-o',
        help='Output directory for experiment artifacts',
    ),
) -> None:
    """Run a single SVM experiment with specified hyperparameters."""
    # Lazy imports for faster CLI startup
    from ..models.svm.experiments import SVMExperimentConfig, run_single_experiment
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

    # Load config for data paths
    with open(config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    splits = data.get('splits', {})
    ablation = cfg.get('ablation', {})
    calibration_cfg = ablation.get('calibration', {})
    feature_union = ablation.get('feature_union', {})
    unigrams = feature_union.get('unigrams', {})
    multigrams = feature_union.get('multigrams', {})

    exp_config = SVMExperimentConfig(
        train_path=splits.get('train', ''),
        val_path=splits.get('val', ''),
        test_path=splits.get('test', ''),
        text_column=data.get('text_column', 'comment'),
        label_column=data.get('label_column', 'sentiment'),
        label_mapping=data.get('label_mapping', {'negative': 0, 'positive': 1}),
        # Unigram params
        unigram_ngram_range=tuple(unigrams.get('ngram_range', [[1, 1]])[0]),
        unigram_min_df=unigrams.get('min_df', [10])[0],
        unigram_max_df=unigrams.get('max_df', [0.90])[0],
        unigram_sublinear_tf=unigrams.get('sublinear_tf', [True])[0],
        unigram_stop_words=unigrams.get('stop_words', ['curated_safe'])[0],
        # Multigram params
        multigram_ngram_range=tuple(multigrams.get('ngram_range', [[2, 3]])[0]),
        multigram_min_df=multigrams.get('min_df', [2])[0],
        multigram_max_df=multigrams.get('max_df', [0.90])[0],
        multigram_sublinear_tf=multigrams.get('sublinear_tf', [True])[0],
        multigram_stop_words=multigrams.get('stop_words', [None])[0],
        # Experiment params
        C=C,
        intercept_scaling=intercept_scaling,
        tol=tol,
        max_iter=max_iter,
        recall_constraint=cfg.get('selection', {}).get('recall_constraint', 0.90),
        output_dir=str(output_dir),
        # Calibration params
        calibration_method=calibration_cfg.get('method', ['isotonic'])[0],
        calibration_cv=calibration_cfg.get('cv', [5])[0],
    )

    log.info(
        json_log(
            'cli.run_experiment.start',
            component='cli',
            C=C,
            intercept_scaling=intercept_scaling,
            tol=tol,
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


@app.command()
def ablation(
    config: Path = typer.Option(
        Path('configs/svm/experiment.yaml'),
        '--config',
        '-c',
        help='Path to ablation config file',
    ),
    max_runs: int | None = typer.Option(
        None,
        '--max-runs',
        help='Maximum runs (for testing)',
    ),
    output_dir: Path = typer.Option(
        Path('artifacts/experiments/svm_linear'),
        '--output-dir',
        '-o',
        help='Output directory for ablation artifacts',
    ),
) -> None:
    """Run full SVM ablation suite from config grids."""
    # Lazy imports for faster CLI startup
    from ..models.svm.experiments import SVMExperimentConfig, run_ablation_suite
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

    # Load config
    with open(config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    splits = data.get('splits', {})
    ablation_cfg = cfg.get('ablation', {})
    feature_union = ablation_cfg.get('feature_union', {})
    unigrams_cfg = feature_union.get('unigrams', {})
    multigrams_cfg = feature_union.get('multigrams', {})
    svm_cfg = ablation_cfg.get('svm_linear', {})
    calibration_cfg = ablation_cfg.get('calibration', {})

    # Build grids from config
    unigram_grid = {
        'ngram_range': [tuple(x) for x in unigrams_cfg.get('ngram_range', [[1, 1]])],
        'min_df': [int(x) for x in unigrams_cfg.get('min_df', [10])],
        'max_df': [float(x) for x in unigrams_cfg.get('max_df', [0.90])],
        'sublinear_tf': [bool(x) for x in unigrams_cfg.get('sublinear_tf', [True])],
        'stop_words': unigrams_cfg.get('stop_words', ['curated_safe']),
    }

    multigram_grid = {
        'ngram_range': [tuple(x) for x in multigrams_cfg.get('ngram_range', [[2, 3]])],
        'min_df': [int(x) for x in multigrams_cfg.get('min_df', [2])],
        'max_df': [float(x) for x in multigrams_cfg.get('max_df', [0.90])],
        'sublinear_tf': [bool(x) for x in multigrams_cfg.get('sublinear_tf', [True])],
        'stop_words': multigrams_cfg.get('stop_words', [None]),
    }

    svm_grid = {
        'penalty': svm_cfg.get('penalty', ['l2']),
        'loss': svm_cfg.get('loss', ['squared_hinge']),
        'C': [float(x) for x in svm_cfg.get('C', [1.0])],
        'dual': [bool(x) for x in svm_cfg.get('dual', [True])],
        'fit_intercept': [bool(x) for x in svm_cfg.get('fit_intercept', [True])],
        'intercept_scaling': [float(x) for x in svm_cfg.get('intercept_scaling', [1.0])],
        'tol': [float(x) for x in svm_cfg.get('tol', [0.0001])],
        'max_iter': [int(x) for x in svm_cfg.get('max_iter', [2000])],
    }

    # Get first value from each FeatureUnion grid for base config
    base_config = SVMExperimentConfig(
        train_path=splits.get('train', ''),
        val_path=splits.get('val', ''),
        test_path=splits.get('test', ''),
        text_column=data.get('text_column', 'comment'),
        label_column=data.get('label_column', 'sentiment'),
        label_mapping=data.get('label_mapping', {'negative': 0, 'positive': 1}),
        recall_constraint=cfg.get('selection', {}).get('recall_constraint', 0.90),
        output_dir=str(output_dir),
        # Calibration params
        calibration_method=calibration_cfg.get('method', ['isotonic'])[0],
        calibration_cv=calibration_cfg.get('cv', [5])[0],
    )

    log.info(
        json_log(
            'cli.run_ablation.start',
            component='cli',
            unigram_grid_size=sum(len(v) for v in unigram_grid.values()),
            multigram_grid_size=sum(len(v) for v in multigram_grid.values()),
            svm_grid_size=sum(len(v) for v in svm_grid.values()),
        ),
    )

    suite = run_ablation_suite(
        base_config=base_config,
        unigram_grid=unigram_grid,
        multigram_grid=multigram_grid,
        svm_grid=svm_grid,
        max_runs=max_runs,
    )

    winner = suite.get('winner')
    summary = suite.get('summary', {})

    typer.echo('Ablation complete!')
    typer.echo(f'Total runs: {summary.get("total_runs", 0)}')
    typer.echo(f'Valid runs: {summary.get("valid_runs", 0)}')
    typer.echo(f'Constraint met: {summary.get("constraint_met_runs", 0)}')

    if winner and winner.val_metrics:
        typer.echo(f'\nWinner: {winner.config.run_id}')
        typer.echo(f'  F1_neg: {winner.val_metrics.f1_neg:.4f}')
        typer.echo(f'  Recall_neg: {winner.val_metrics.recall_neg:.4f}')
        typer.echo(f'  Threshold: {winner.val_metrics.threshold:.4f}')


@app.command('ablation-rbf')
def ablation_rbf(
    config: Path = typer.Option(
        Path('configs/svm/experiment.yaml'),
        '--config',
        '-c',
        help='Path to ablation config file',
    ),
    max_runs: int | None = typer.Option(
        None,
        '--max-runs',
        help='Maximum runs (for testing)',
    ),
    output_dir: Path = typer.Option(
        Path('artifacts/experiments/svm_rbf'),
        '--output-dir',
        '-o',
        help='Output directory for RBF ablation artifacts',
    ),
) -> None:
    """Run full SVC RBF ablation suite from config grids."""
    # Lazy imports for faster CLI startup
    from ..models.svm.experiments import SVCRBFExperimentConfig, run_rbf_ablation_suite
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

    # Load config
    with open(config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    splits = data.get('splits', {})
    ablation_cfg = cfg.get('ablation', {})
    feature_union = ablation_cfg.get('feature_union', {})
    unigrams_cfg = feature_union.get('unigrams', {})
    multigrams_cfg = feature_union.get('multigrams', {})
    svm_rbf_cfg = ablation_cfg.get('svm_rbf', {})
    calibration_cfg = ablation_cfg.get('calibration', {})

    # Build grids from config
    unigram_grid = {
        'ngram_range': [tuple(x) for x in unigrams_cfg.get('ngram_range', [[1, 1]])],
        'min_df': [int(x) for x in unigrams_cfg.get('min_df', [10])],
        'max_df': [float(x) for x in unigrams_cfg.get('max_df', [0.90])],
        'sublinear_tf': [bool(x) for x in unigrams_cfg.get('sublinear_tf', [True])],
        'stop_words': unigrams_cfg.get('stop_words', ['curated_safe']),
    }

    multigram_grid = {
        'ngram_range': [tuple(x) for x in multigrams_cfg.get('ngram_range', [[2, 3]])],
        'min_df': [int(x) for x in multigrams_cfg.get('min_df', [2])],
        'max_df': [float(x) for x in multigrams_cfg.get('max_df', [0.90])],
        'sublinear_tf': [bool(x) for x in multigrams_cfg.get('sublinear_tf', [True])],
        'stop_words': multigrams_cfg.get('stop_words', [None]),
    }

    # SVC RBF grid - handle gamma which can be string or float
    gamma_values = svm_rbf_cfg.get('gamma', ['scale'])
    processed_gamma = []
    for g in gamma_values:
        if isinstance(g, str):
            processed_gamma.append(g)
        else:
            processed_gamma.append(float(g))

    svc_grid = {
        'kernel': svm_rbf_cfg.get('kernel', ['rbf']),
        'C': [float(x) for x in svm_rbf_cfg.get('C', [1.0])],
        'gamma': processed_gamma,
        'tol': [float(x) for x in svm_rbf_cfg.get('tol', [0.001])],
        'cache_size': [float(x) for x in svm_rbf_cfg.get('cache_size', [1000])],
        'max_iter': [int(x) for x in svm_rbf_cfg.get('max_iter', [-1])],
        'use_scaler': [bool(x) for x in svm_rbf_cfg.get('use_scaler', [True])],
    }

    # Get first value from each FeatureUnion grid for base config
    base_config = SVCRBFExperimentConfig(
        train_path=splits.get('train', ''),
        val_path=splits.get('val', ''),
        test_path=splits.get('test', ''),
        text_column=data.get('text_column', 'comment'),
        label_column=data.get('label_column', 'sentiment'),
        label_mapping=data.get('label_mapping', {'negative': 0, 'positive': 1}),
        recall_constraint=cfg.get('selection', {}).get('recall_constraint', 0.90),
        output_dir=str(output_dir),
        # Calibration params
        calibration_method=calibration_cfg.get('method', ['isotonic'])[0],
        calibration_cv=calibration_cfg.get('cv', [5])[0],
    )

    log.info(
        json_log(
            'cli.run_ablation_rbf.start',
            component='cli',
            unigram_grid_size=sum(len(v) for v in unigram_grid.values()),
            multigram_grid_size=sum(len(v) for v in multigram_grid.values()),
            svc_grid_size=sum(len(v) for v in svc_grid.values()),
        ),
    )

    suite = run_rbf_ablation_suite(
        base_config=base_config,
        unigram_grid=unigram_grid,
        multigram_grid=multigram_grid,
        svc_grid=svc_grid,
        max_runs=max_runs,
    )

    winner = suite.get('winner')
    summary = suite.get('summary', {})

    typer.echo('RBF Ablation complete!')
    typer.echo(f'Total runs: {summary.get("total_runs", 0)}')
    typer.echo(f'Valid runs: {summary.get("valid_runs", 0)}')
    typer.echo(f'Constraint not met: {summary.get("constraint_not_met_runs", 0)}')

    if winner and winner.val_metrics:
        typer.echo(f'\nWinner: {winner.config.run_id}')
        typer.echo(f'  F1_neg: {winner.val_metrics.f1_neg:.4f}')
        typer.echo(f'  Recall_neg: {winner.val_metrics.recall_neg:.4f}')
        typer.echo(f'  Threshold: {winner.val_metrics.threshold:.4f}')


@app.command('ablation-report')
def ablation_report(
    model_type: Annotated[
        str,
        typer.Option(
            '--model-type',
            '-m',
            help='Model type: linear or rbf (REQUIRED)',
        ),
    ] = ...,  # Required, no default
    experiments_dir: Annotated[
        Path,
        typer.Option(
            '--experiments-dir',
            '-e',
            exists=True,
            help='Path to experiments directory containing run.* folders.',
        ),
    ] = Path('artifacts/experiments/svm_linear'),
    output_dir: Annotated[
        Path,
        typer.Option(
            '--output-dir',
            '-o',
            help='Output directory for reports.',
        ),
    ] = Path('reports/svm_ablations'),
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
    """Generate ablation analysis report for SVM experiments."""
    from ..models.svm.reports import generate_ablation_report
    from ..utils import get_logger, json_log

    log = get_logger(__name__)

    # Validate model_type
    if model_type not in ('linear', 'rbf'):
        typer.echo(f"Error: --model-type must be 'linear' or 'rbf', got: {model_type}", err=True)
        raise typer.Exit(1)

    log.info(
        json_log(
            'cli.ablation_report.start',
            component='cli',
            model_type=model_type,
            experiments_dir=str(experiments_dir),
            output_dir=str(output_dir),
            test_predictions=str(test_predictions) if test_predictions else None,
        ),
    )

    artifacts = generate_ablation_report(
        experiments_dir=experiments_dir,
        output_dir=output_dir,
        model_type=model_type,
        test_predictions_path=test_predictions,
        winner_run_id=winner_run_id,
    )

    typer.echo('SVM Ablation report generated!')
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
