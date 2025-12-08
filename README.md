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

### Single CSV

```bash
python -m gym_sentiment_guard.cli.main main preprocess \
  --input data/raw/Fitness_Park_Madrid_La_Vaguada.csv \
  --config configs/preprocess.yaml
```

This command (with `language.enabled: true` in `configs/preprocess.yaml`):
- Filters reviews to Spanish only via fastText.
- Normalizes text, deduplicates rows, and enforces lightweight expectations.
- Drops the `name` column and neutral (3-star) reviews while writing a `<name>.neutral.csv` audit file under `data/interim/` for later analysis.
- Adds a `sentiment` column (positive/negative) derived from ratings (`language.sentiment_column` in the config) so training scripts share a consistent label.
- Drops neutral (3-star) reviews while writing a `<name>.neutral.csv` audit file under `data/interim/` for later analysis.
- Writes interim artifacts to `data/interim/` (including `<name>.non_spanish.csv` with an `es_confidence` column for rejected rows) and the final cleaned file to `data/processed/<name>.clean.csv`.

Override the final output path with `--output path/to/custom.csv` if needed.

### Batch preprocessing (all pending CSVs)

```bash
python -m gym_sentiment_guard.cli.main main preprocess-batch \
  --config configs/preprocess.yaml \
  --pattern "*.csv"
```

The CLI scans `paths.raw_dir` for CSVs that do not yet have a `.clean.csv` in `paths.processed_dir`, runs the single-file pipeline for each one, and logs successes/failures. Use `--raw-dir` / `--processed-dir` if you need to override the paths from the config. (The legacy `scripts/process_pending_csvs.py` now simply wraps this command.)

### Merge processed datasets

```bash
python -m gym_sentiment_guard.cli.main main merge-processed \
  --config configs/preprocess.yaml \
  --pattern "*.clean.csv" \
  --output data/processed/train_dataset.csv
```

The command gathers every matching file inside `paths.processed_dir`, enforces consistent columns, and writes a merged dataset. (The helper `scripts/merge_processed_datasets.py` delegates to this command for compatibility.)

### Full pipeline (batch preprocess + merge)

```bash
python -m gym_sentiment_guard.cli.main main run-full-pipeline \
  --config configs/preprocess.yaml
```

This orchestrates `preprocess-batch` followed by `merge-processed`, producing the merged dataset in one shot. Override `--raw-pattern`, `--merge-pattern`, or `--merge-output` as needed.


### Skipping language filtering

Set `language.enabled: false` in `configs/preprocess.yaml` when you want to reuse the cleaning steps (expectations → normalize → dedup) without dropping non-Spanish rows—for example when preparing the multi-language evaluation datasets. In that mode, the pipeline copies the deduplicated file directly to `data/processed/<name>.clean.csv` and skips generating `.non_spanish.csv`.

> Schema guard: if any processed CSV has columns that differ from the others, the merge helper logs `merge.schema_mismatch` and raises a `ValueError` so we do not silently mix incompatible datasets.

### Train/Val/Test splits

Splitting the merged dataset uses the `split-data` command (also run automatically by `run-full-pipeline`):

```bash
python -m gym_sentiment_guard.cli.main main split-data \
  --input data/processed/merged/merged_dataset.csv \
  --output-dir data/processed/splits \
  --column sentiment
```

The helper stratifies by the chosen column (default `language.sentiment_column`) and writes `train.csv`, `val.csv`, and `test.csv` (70/15/15).

### Normalize an arbitrary dataset

To apply the latest text normalization rules (emoji stripping, punctuation cleanup) to an existing CSV such as `merged_dataset.csv` without re-running the full pipeline:

```bash
python -m gym_sentiment_guard.cli.main main normalize-dataset \
  --input data/processed/merged/merged_dataset.csv \
  --output data/processed/merged/merged_dataset.normalized.csv \
  --column comment \
  --punctuation-file configs/structural_punctuation.txt
```

This reuses the core `normalize_comments` helper so normalization stays consistent with the preprocessing pipeline.

## Language Evaluation Utilities

The following scripts in `scripts/` help build LID ground truth for the Google Maps evaluation dataset:

- `python scripts/tag_ground_truth_languages.py --root data/raw`
  - Walks language folders (es/pt/it/en) and stamps a `language` column in every CSV.
- `python scripts/merge_language_datasets.py --root data/raw --output-name merged.csv`
  - Concatenates each folder’s CSVs into `<folder>/merged.csv`, discarding rows whose `language` tag doesn’t match the folder name. (The language tag of each review was reviewed by Gemini 2.5 pro to assert that it matches the real language
  of the review).
- `python scripts/sample_language_reviews.py --root data/raw --samples 250`
  - Randomly samples up to 250 reviews from each `.clean.csv`, writing `<name>.sampled.csv` for eval subsets.
- `python scripts/merge_sampled_languages.py --root data/raw`
  - Combines all `.sampled.csv` files into a single `merged_sampled_ground_truth.csv` ready for LID evaluation.
- `python scripts/eval_fasttext_lid.py --data data/lid_eval/eval_dataset/merged_sampled_ground_truth.csv --model artifacts/external/models/lid.176.bin`
  - Loads the merged dataset, runs fastText predictions, prints overall accuracy plus per-language precision/recall/F1, writes `eval_results.json`, saves a confusion-matrix image (`confusion_matrix.png` by default), and reports coverage + accuracy/precision/recall/F1 for each confidence threshold (≥0.95/0.90/0.85/0.80/0.75) to design fallback policies.


### LLM Fallback Service (Gemini)

To use Gemini as the fallback language detector:

1. Export your API key (e.g., PowerShell):  
   ```powershell
   $env:GEMINI_API_KEY = "your-key"
   ```

2. Start the FastAPI service:  
   ```bash
   uvicorn scripts.llm_service:app --reload --port 8000
   ```

3. In `configs/preprocess.yaml`, enable fallback:  
   ```yaml
   language:
     ...
     confidence_threshold: 0.75
     fallback_enabled: true
     fallback_endpoint: http://localhost:8000/detect_language
     fallback_api_key_env: GEMINI_API_KEY
   ```

When fastText confidence drops below the threshold, the pipeline will POST the review to the running FastAPI service, which in turn calls Gemini and returns the ISO 639-1 language code.

**Logging & Monitoring**

- The FastAPI service logs every Gemini call (`Gemini status ... response ...`). Errors (4xx/5xx) are logged at ERROR level so you can spot upstream issues.
- The preprocess pipeline logs fallback usage. `language_filter.llm_response` entries appear at DEBUG level for successful calls; `language_filter.llm_response_error` warnings show when the Gemini call fails, making it easy to diagnose problems directly in the CLI output.
- The fallback is triggered both when fastText’s top prediction is low-confidence *and* when Spanish ranks within the top-3 with high probability but isn’t the top label. This lets Gemini double-check mixed-language reviews.

### Optional: Mistral helper

`scripts/llm_service.py` now includes `_call_mistral`, a drop-in helper that hits `https://api.mistral.ai/v1/chat/completions` using the OpenAI-compatible schema. To use it:

1. Export credentials:  
   ```powershell
   $env:MISTRAL_API_KEY = "your-mistral-token"
   $env:MISTRAL_MODEL = "mistral-small-latest"  # optional override
   ```
2. Point the FastAPI route to `_call_mistral` (or add a second endpoint) if you prefer Mistral over Gemini. The helper reuses the same prompt/temperature settings and structured logging, so swapping providers only requires changing the callable.

## Design Decisions (LLM Chain-of-Thought)

### Language Identification Confidence Threshold

**Summary**  
FastText LID (`lid.176.bin`) was evaluated on 1000 labeled gym reviews (250 per language: en, es, pt, it). Accuracy was measured for subsets of predictions at different confidence levels.

| Confidence ≥ τ | Coverage | Accuracy |
|----------------|----------|----------|
| ≥0.95 | 78.2% | 0.997 |
| ≥0.90 | 88.2% | 0.998 |
| ≥0.85 | 91.7% | 0.998 |
| ≥0.80 | 93.4% | 0.996 |
| **≥0.75** | **94.5%** | **0.996** |
| ≥0.50 | 97.2% | 0.994 |
| ≥0.25 | 98.8% | 0.990 |

**Interpretation**
- Accuracy remains ≈0.996–0.998 for τ ≥ 0.75.  
- Most misclassifications occur below 0.75 confidence.  
- Increasing τ above 0.75 yields little accuracy gain but reduces coverage.  
- Differences between 0.996 and 0.998 accuracy are not statistically meaningful at n=1000.

**Decision**
- Use **τ = 0.75** as the confidence cutoff.  
  - If `prob ≥ 0.75`: accept fastText prediction.  
  - If `prob < 0.75`: fallback to Gemini 2.5 Flash for language verification.

This threshold balances high accuracy with minimal fallback usage.


### Data Splitting

**Decision**
The dataset is divided into 70/15/15 for train, val and test:
  - I consider that a dataset of ~10k reviews is enough to reduce variation in the eval and test set. Using stratified sampling will reduce possible variation because both of the classes are proportionally represented, and the minority class has 1500+ revies, making it sufficient for a first linear model. 
  If low performance is observed some changes which would be considered would be: 
  - augmenting the training set 
  - changing the data split 
  - adding K fold cross validation to the training process.
  - Use temporal splitting: train model with older data and test it with recent data (E.g: train on 2020-2022, test on 2024-2025). The provblem with this is that, even though gym situations can change (New monitors, new machines, management changes), the way of expressing satisfaction or discomfort with services should not in a such small period of time (Slang change, vocabulary change, new generations entering the gym). 
