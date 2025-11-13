"""Command-line interface for gym_sentiment_guard."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import load_preprocess_config
from ..pipeline import preprocess_reviews
from ..utils import get_logger, json_log

app = typer.Typer(help='Gym Sentiment Guard CLI', no_args_is_help=True)
pipeline_app = typer.Typer(help='Pipeline commands', no_args_is_help=True)
app.add_typer(pipeline_app, name='main')

log = get_logger(__name__)


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


if __name__ == '__main__':
    app()
