# gym-sentiment-guard
Sentiment classifier for gym reviews using TF-IDF, linear models, and reproducible preprocessing.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## CLI Reference

The CLI is organized into two command groups:
- **`gym pipeline`**: Data curation and preprocessing commands
- **`gym logreg`**: Logistic Regression model training, experiments, and analysis

---

### `gym pipeline preprocess`

Run the preprocessing pipeline on a single CSV file.

```bash
gym pipeline preprocess \
  --input data/raw/Fitness_Park_Madrid_La_Vaguada.csv \
  --config configs/preprocess.yaml \
  --output data/processed/custom_output.csv
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--input` | `-i` | **Yes** | — | Path to raw CSV file |
| `--config` | `-c` | No | `configs/preprocess.yaml` | Path to preprocess configuration YAML |
| `--output` | `-o` | No | Auto-derived | Output CSV path (defaults to `<processed_dir>/<name>.clean.csv`) |

**Behavior**: Filters reviews to Spanish (if enabled), normalizes text, deduplicates rows, adds `sentiment` column, writes interim artifacts to `data/interim/`, and final output to `data/processed/`.

---

### `gym pipeline batch`

Process all pending raw CSV files in a directory.

```bash
gym pipeline batch \
  --config configs/preprocess.yaml \
  --pattern "*.csv" \
  --raw-dir data/raw \
  --processed-dir data/processed
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/preprocess.yaml` | Path to preprocess configuration YAML |
| `--pattern` | `-p` | No | `*.csv` | Glob pattern for raw CSVs |
| `--raw-dir` | — | No | From config | Override raw directory path |
| `--processed-dir` | — | No | From config | Override processed directory path |

---

### `gym pipeline merge`

Merge multiple processed CSVs into a single training dataset.

```bash
gym pipeline merge \
  --config configs/preprocess.yaml \
  --pattern "*.clean.csv" \
  --output data/processed/train_dataset.csv \
  --processed-dir data/processed
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/preprocess.yaml` | Path to preprocess configuration YAML |
| `--pattern` | — | No | `*.clean.csv` | Glob pattern for processed CSVs |
| `--output` | `-o` | No | `<processed_dir>/merged/merged_dataset.csv` | Destination merged CSV path |
| `--processed-dir` | — | No | From config | Override processed directory path |

---

### `gym pipeline split`

Split a dataset into train/val/test CSVs with stratification.

```bash
gym pipeline split \
  --input data/processed/merged/merged_dataset.csv \
  --output-dir data/processed/splits \
  --column sentiment \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --random-state 42
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--input` | `-i` | No | `data/processed/merged/merged_dataset.csv` | Path to merged dataset CSV |
| `--output-dir` | — | No | `data/processed/splits` | Directory to store splits |
| `--column` | — | No | `sentiment` | Column to stratify on |
| `--train-ratio` | — | No | `0.7` | Train split ratio |
| `--val-ratio` | — | No | `0.15` | Validation split ratio |
| `--test-ratio` | — | No | `0.15` | Test split ratio |
| `--random-state` | — | No | `42` | Random seed for reproducibility |

---

### `gym pipeline run`

Run the full pipeline: batch preprocess + merge + split.

```bash
gym pipeline run \
  --config configs/preprocess.yaml \
  --raw-pattern "*.csv" \
  --merge-pattern "*.clean.csv" \
  --merge-output data/processed/merged/merged_dataset.csv \
  --split-output data/processed/splits \
  --raw-dir data/raw \
  --processed-dir data/processed
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/preprocess.yaml` | Path to preprocess configuration YAML |
| `--raw-pattern` | — | No | `*.csv` | Glob for raw CSVs |
| `--merge-pattern` | — | No | `*.clean.csv` | Glob for processed CSVs |
| `--merge-output` | — | No | Auto-derived | Merged dataset path override |
| `--split-output` | — | No | Auto-derived | Directory to store dataset splits |
| `--raw-dir` | — | No | From config | Override raw directory |
| `--processed-dir` | — | No | From config | Override processed directory |

---

### `gym pipeline normalize`

Apply text normalization rules to an existing CSV without full preprocessing.

```bash
gym pipeline normalize \
  --input data/processed/merged/merged_dataset.csv \
  --output data/processed/merged/merged_dataset.normalized.csv \
  --column comment \
  --punctuation-file configs/structural_punctuation.txt
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--input` | `-i` | **Yes** | — | CSV file to normalize |
| `--output` | `-o` | No | `<input>.normalized.csv` | Output path |
| `--column` | — | No | `comment` | Text column to normalize |
| `--punctuation-file` | — | No | None | Structural punctuation file |

---

### `gym logreg train`

Train a sentiment model from a configuration file.

```bash
gym logreg train \
  --config configs/logreg/training_v4.yaml
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/logreg_v1.yaml` | Path to model config YAML |

**Output**: Model artifacts saved to `artifacts/models/sentiment_logreg/model.<date>_<seq>/`.

---

### `gym logreg experiment`

Run a single experiment with specified hyperparameters.

```bash
gym logreg experiment \
  --config configs/logreg/experiment.yaml \
  --ngram-min 1 \
  --ngram-max 2 \
  --min-df 2 \
  --max-df 1.0 \
  --sublinear-tf \
  --C 1.0 \
  --penalty l2 \
  --class-weight balanced \
  --output-dir artifacts/experiments
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/logreg/experiment.yaml` | Path to experiment config YAML |
| `--ngram-min` | — | No | `1` | Minimum n-gram size |
| `--ngram-max` | — | No | `2` | Maximum n-gram size |
| `--min-df` | — | No | `2` | Minimum document frequency |
| `--max-df` | — | No | `1.0` | Maximum document frequency |
| `--sublinear-tf` | — | No | `true` | Use sublinear TF scaling (use `--no-sublinear-tf` to disable) |
| `--C` | — | No | `1.0` | Regularization strength |
| `--penalty` | — | No | `l2` | Regularization type: `l1` or `l2` |
| `--class-weight` | — | No | None | Class weight: `balanced` or None |
| `--output-dir` | — | No | `artifacts/experiments` | Output directory for artifacts |

---

### `gym logreg ablation`

Run a full ablation suite from config grids.

```bash
gym logreg ablation \
  --config configs/logreg/experiment.yaml \
  --max-runs 10 \
  --output-dir artifacts/experiments
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/logreg/experiment.yaml` | Path to experiment config YAML |
| `--max-runs` | — | No | None (all) | Maximum number of runs (for testing) |
| `--output-dir` | — | No | `artifacts/experiments` | Output directory for artifacts |

Grid parameters are defined in `configs/logreg/experiment.yaml` under `ablation.tfidf` and `ablation.logreg`.

---

### `gym logreg ablation-report`

Generate ablation suite reports with visualizations.

```bash
gym logreg ablation-report \
  --experiments-dir artifacts/experiments \
  --output reports/logreg_ablations \
  --test-predictions artifacts/models/sentiment_logreg/model.2026-01-10_002/test_predictions.csv \
  --winner run.2026-01-10_001
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--experiments-dir` | `-e` | No | `artifacts/experiments` | Path to experiments directory containing `run.*` folders |
| `--output` | `-o` | No | `reports/logreg_ablations` | Output directory for reports |
| `--test-predictions` | `-t` | No | None | Path to `test_predictions.csv` for Layer 4 PR curves |
| `--winner` | `-w` | No | Auto-detect | Explicit winner run_id |

---

### `gym logreg error-analysis`

Run post-training error analysis on model predictions.

```bash
gym logreg error-analysis \
  --config configs/logreg/error_analysis.yaml \
  --output reports/error_analysis/model.2026-01-21_001 \
  --model artifacts/models/sentiment_logreg/model.2026-01-21_001/model.joblib \
  --predictions artifacts/models/sentiment_logreg/model.2026-01-21_001/test_predictions.csv \
  --test-csv data/frozen/sentiment_logreg/2025.12.15_01/test/test.csv
```

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/logreg/error_analysis.yaml` | Path to error_analysis.yaml config |
| `--output` | `-o` | No | Auto-derived from model path | Output directory |
| `--model` | `-m` | No | From config | Override model path |
| `--predictions` | `-p` | No | From config | Override predictions CSV path |
| `--test-csv` | `-t` | No | From config | Override test CSV path |

**Output**: Artifacts saved to `reports/error_analysis/<model_id>/`.

---

## Training Configuration

### Hybrid Vectorization (FeatureUnion)

For advanced use cases, configure separate TF-IDF settings per n-gram length using `FeatureUnion`. This enables **strategy-specific** `min_df` and `stop_words` filtering:

```yaml
vectorizer:
  type: feature_union
  lowercase: true
  max_df: 0.95
  sublinear_tf: true
  stop_words: curated_safe  # Global fallback
  strategies:
    unigrams:
      ngram_range: [1, 1]
      min_df: 10
      stop_words: curated_safe  # Heavy filtering for single words
    multigrams:
      ngram_range: [2, 3]
      min_df: 2
      stop_words: null  # No filtering to preserve context anchors
```

**Why per-strategy filtering?**

- **Unigrams** benefit from aggressive stopword removal (e.g., `"de"`, `"el"`, `"que"`) and higher `min_df` to reduce domain bias and noise.
- **Multigrams** should preserve all words to maintain meaningful 3-gram anchors like `"respuesta del director"` or `"devuelvan mi dinero"` — removing stopwords would break these context-dependent phrases.

**Result:** This differentiation enables the model to learn semantically meaningful n-gram coefficients such as:
- `"muy bien"` (+1.91)
- `"me encanta"` (+1.79)
- `"no está mal"` (+1.72) 
- `"tiene de todo"` (+1.51)

Critically, words like `"no"` that are typically negative in isolation (−8.75) become **positive indicators** when combined in phrases like `"no está mal"` (not bad = good).

> See full coefficients: [`reports/error_analysis/model.2026-01-16_003/model_coefficients.json`](reports/error_analysis/model.2026-01-16_003/model_coefficients.json)

Use `stop_words: null` explicitly to disable filtering for a strategy. Legacy configs (without `strategies` key) continue to work unchanged.

### Ablation Report Layers

| Layer | Purpose | Artifacts |
|-------|---------|-----------|
| **1** | Ablation Summary Table | `ablation_table_sorted.csv` |
| **2** | Top-K Results | `TOP5_RESULTS.md`, bar chart |
| **3** | Factor-Level Analysis | `ABLATION_ANALYSIS.md`, C/ngram/stopwords plots |
| **4** | Final Model Deep Dive | `FINAL_MODEL_REPORT.md`, confusion matrix, PR curve, calibration |

### Error Analysis Artifacts

| Artifact | Description |
|----------|-------------|
| `error_table.parquet` | Full error table with risk tags |
| `ranked_errors/high_confidence_wrong.csv` | Confident but incorrect predictions |
| `ranked_errors/top_loss_wrong.csv` | Highest cross-entropy loss errors |
| `ranked_errors/near_threshold_wrong.csv` | Uncertain predictions |
| `slice_metrics.json` | Metrics per data slice |
| `model_coefficients.json` | Top positive/negative model coefficients |
| `KNOWN_LIMITATIONS.md` | Auto-generated deployment knowledge |
| `run_manifest.json` | Reproducibility audit trail |

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
- The fallback is triggered both when fastText’s top prediction is low-confidence *and* when Spanish ranks within the top-k (k=5) with high probability but isn’t the top label. This lets Gemini double-check mixed-language reviews.

### Optional: Mistral helper

`scripts/llm_service.py` now includes `_call_mistral`, a drop-in helper that hits `https://api.mistral.ai/v1/chat/completions` using the OpenAI-compatible schema. To use it:

1. Export credentials:  
   ```powershell
   $env:MISTRAL_API_KEY = "your-mistral-token"
   $env:MISTRAL_MODEL = "mistral-small-latest"  # optional override
   ```
2. Point the FastAPI route to `_call_mistral` (or add a second endpoint) if you prefer Mistral over Gemini. The helper reuses the same prompt/temperature settings and structured logging, so swapping providers only requires changing the callable.

## Design Decisions

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

---

## Serving API

Serve the trained sentiment model via a REST API for real-time predictions.

### Start the Server

```bash
uvicorn gym_sentiment_guard.serving.app:app --reload --port 8001
```

Or with a custom config:

```bash
GSG_SERVING_CONFIG=configs/serving.yaml uvicorn gym_sentiment_guard.serving.app:app --port 8001
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check (model loaded) |
| GET | `/model/info` | Model metadata (version, threshold) |
| POST | `/predict` | Predict sentiment for 1-100 texts |
| POST | `/predict/explain` | Predict with feature importance explanations |

### Example Requests

**Single prediction:**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Excelente gimnasio, muy limpio y buen ambiente"]}'
```

Response:
```json
[
  {
    "sentiment": "positive",
    "confidence": 0.87,
    "probability_positive": 0.87,
    "probability_negative": 0.13,
    "model_version": "2025.12.16_002"
  }
]
```

**Multiple predictions:**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Muy buen gym", "Pésimo servicio"]}'
```

**Prediction with explanation:**

```bash
curl -X POST http://localhost:8001/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Excelente gimnasio, muy limpio"]}'
```

Response:
```json
[
  {
    "sentiment": "positive",
    "confidence": 0.91,
    "probability_positive": 0.91,
    "probability_negative": 0.09,
    "model_version": "2025.12.16_002",
    "explanation": [
      {"feature": "excelente", "importance": 2.45},
      {"feature": "limpio", "importance": 1.82},
      {"feature": "gimnasio", "importance": 0.34}
    ]
  }
]
```

The `explanation` field contains the top contributing features sorted by absolute importance. Positive values push towards "positive" sentiment, negative values push towards "negative" sentiment.

### Configuration

Configure via `configs/serving.yaml`:

```yaml
model:
  path: artifacts/models/sentiment_logreg/model.2025.12.16_002

preprocessing:
  enabled: true

validation:
  max_text_bytes: 51200  # 50KB max

batch:
  max_items: 100
  max_text_bytes_per_item: 5120  # 5KB

logging:
  mode: minimal  # or "requests" for per-request logging
```

### Swagger Documentation

Visit `http://localhost:8001/docs` for interactive API documentation.

---

## Docker Deployment

### Build the Image

```bash
docker build -t gym-sentiment-guard:local .
```

### Run with Docker Compose (Recommended)

```bash
# Start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Run with Docker Directly

```bash
docker run -d \
  --name gsg-api \
  -p 8001:8080 \
  -e GSG_SERVING_CONFIG=/app/configs/serving.yaml \
  gym-sentiment-guard:local
```

### Test the Container

```bash
# Health check
curl http://localhost:8001/health

# Prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Excelente gimnasio"]}'

# Explain prediction
curl -X POST http://localhost:8001/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Excelente gimnasio"]}'
```

### Verify Security

```bash
# Confirm non-root user
docker exec gsg-api whoami
# Expected output: appuser
```

### Image Details

| Property | Value |
|----------|-------|
| Base Image | `python:3.12-slim-bookworm` |
| Port | 8080 (mapped to 8001 locally) |
| User | `appuser` (non-root) |
| Health Check | `/health` endpoint |

