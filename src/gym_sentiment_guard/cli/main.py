"""Command-line interface dispatcher for gym_sentiment_guard."""

from __future__ import annotations

import typer

from .logreg import app as logreg_app
from .pipeline import app as pipeline_app

app = typer.Typer(help='Gym Sentiment Guard CLI', no_args_is_help=True)

# Register sub-apps for clean hierarchy
app.add_typer(pipeline_app, name='pipeline')
app.add_typer(logreg_app, name='logreg')

if __name__ == '__main__':
    app()
