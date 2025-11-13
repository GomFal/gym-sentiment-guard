from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gym_sentiment_guard.cli.main import app


def test_cli_preprocess_invokes_pipeline(monkeypatch, tmp_path):
    runner = CliRunner()
    input_csv = tmp_path / 'raw.csv'
    input_csv.write_text('comment,rating\nHola,5\n')
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('stub: true')
    output_csv = tmp_path / 'out.csv'

    captured = {}

    def fake_load_config(path: Path):
        captured['config_path'] = Path(path)
        return 'CONFIG'

    def fake_preprocess(input_path, config, output_path=None):  # noqa: ARG001
        captured['input_path'] = Path(input_path)
        captured['config'] = config
        captured['output_path'] = Path(output_path) if output_path else None
        return output_csv

    monkeypatch.setattr('gym_sentiment_guard.cli.main.load_preprocess_config', fake_load_config)
    monkeypatch.setattr('gym_sentiment_guard.cli.main.preprocess_reviews', fake_preprocess)

    result = runner.invoke(
        app,
        [
            'main',
            'preprocess',
            '--input',
            str(input_csv),
            '--config',
            str(config_file),
            '--output',
            str(output_csv),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert 'Processed data written to' in result.stdout
    assert captured['config_path'] == config_file
    assert captured['input_path'] == input_csv
    assert captured['output_path'] == output_csv
