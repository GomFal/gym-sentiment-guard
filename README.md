# gym-sentiment-guard
Sentiment classifier for gym reviews using TF-IDF, linear models, and reproducible preprocessing.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Preprocess CLI

Place the fastText `lid.176.bin` model under `artifacts/models/lid.176.bin` (or update the config), then run:

```bash
python -m gym_sentiment_guard.cli.main main preprocess \
  --input data/raw/Fitness_Park_Madrid_La_Vaguada.csv \
  --config configs/preprocess.yaml
```

This command:
- Filters reviews to Spanish only via fastText.
- Normalizes text, deduplicates rows, and enforces lightweight expectations.
- Writes interim artifacts to `data/interim/` (including `<name>.non_spanish.csv` with an `es_confidence` column for rejected rows) and the final cleaned file to `data/processed/<name>.clean.csv`.

Override the final output path with `--output path/to/custom.csv` if needed.
