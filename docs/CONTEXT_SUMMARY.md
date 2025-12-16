## General Context and Summary of the project:

Gym Sentiment Guard ingests Google Maps–style gym reviews and produces clean, Spanish-only datasets so a classic ML model (LogReg/SVM) can be trained and run on schedule.  
The repo centers on a Typer CLI (`python -m gym_sentiment_guard.cli.main main …`) that orchestrates a deterministic pipeline:

- Expectations checks (schema, min text length, emoji stripping)  
- Normalization (lowercase, structural punctuation removal, whitespace collapse)  
- Dedup on `(comment, rating, name)`  
- fastText language filtering with LLM fallback (Gemini or Mistral)  
- Neutral rating removal (3-star reviews dropped, `.neutral.csv` audit file written)  
- Sentiment label derivation (`positive`/`negative` from ratings 4-5 / 1-2)  
- Name column removal for privacy  
- Emission of `.clean` processed files and `.non_spanish` audit files  

Paths, thresholds, and toggles live in `configs/preprocess.yaml` to keep runs reproducible without code edits.

---

### Architecture

- `src/gym_sentiment_guard/cli/main.py` → Typer CLI with 7 pipeline commands  
- `src/gym_sentiment_guard/data/` → cleaners (`cleaning.py`), language logic (`language.py`), merge (`merge.py`), split (`split.py`)  
- `src/gym_sentiment_guard/pipeline/preprocess.py` → orchestrates pipeline stages  
- `src/gym_sentiment_guard/training/model.py` → config-driven model training  
- `src/gym_sentiment_guard/features/` → sentiment label derivation from ratings  
- `src/gym_sentiment_guard/config/` → pydantic config schemas  
- `scripts/` → auxiliary tooling (language-eval prep, fastText eval, FastAPI LLM proxy)  
- Logging is JSON-only so CLI, pipeline stages, and FastAPI responses share a uniform structure.

fastText runs with `k=5` predictions per review.  
Fallback triggers LLM when:
- top-1 confidence < threshold (0.75), **or**
- Spanish appears in top-k above threshold but isn't top-1.

`_call_llm_language_detector` handles one-review-per-request, enforced by:
- a leaky-bucket governor (60 RPM default for Mistral)  
- Tenacity retries/backoff to stay within API rate limits  

`scripts/llm_service.py` wraps both Gemini and Mistral endpoints, logs status + response, and surfaces errors.

---

### CLI Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Process single CSV through full pipeline |
| `preprocess-batch` | Process all pending CSVs in `raw/` directory without `.clean` counterparts |
| `merge-processed` | Merge multiple `.clean.csv` into single dataset with schema validation |
| `split-data` | Stratified train/val/test split (70/15/15 default) |
| `run-full-pipeline` | Batch preprocess → merge → split in one orchestrated command |
| `normalize-dataset` | Apply normalization rules to existing CSV without full pipeline |
| `train-model` | Train sentiment model from YAML config (TF-IDF + calibrated LogReg) |

---

### Supporting Services / Pipeline Processes

- **Preprocess CLI** (core pipeline; set `language.enabled: false` to skip language filtering)  
- **preprocess-batch** CLI command (replaces legacy `process_pending_csvs.py` which now wraps this)  
- **eval_fasttext_lid.py** → accuracy/precision/recall/F1, threshold coverage, confusion-matrix PNG, JSON summaries  
- Scripts for:
  - annotating ground-truth languages (`tag_ground_truth_languages.py`)  
  - merging per-language folders (`merge_language_datasets.py`)  
  - balanced sampling (`sample_language_reviews.py`)  
  - final merges for evaluation (`merge_sampled_languages.py`)  
- **FastAPI** proxy (`scripts/llm_service.py`) supports both Gemini and Mistral, streams structured logs and backs off on 429s

---

### Model Training

Training uses `train-model` CLI with config from `configs/logreg_v1.yaml`:
- **Vectorizer**: TF-IDF with unigrams, min_df=2, sublinear_tf enabled  
- **Classifier**: Logistic Regression (C=1.0, lbfgs solver, max_iter=1000)  
- **Calibration**: Isotonic regression with 5-fold CV for probability calibration  
- **Decision threshold**: 0.44 targeting the negative class probability  
- **Artifacts**: `logreg.joblib` model, `metadata.json`, `metrics_test.json` per run  
- Versioned output directories: `artifacts/models/sentiment_logreg/model.YYYY-MM-DD_NNN/`

---

### Design Decisions

**Language ID Confidence Threshold (τ=0.75)**  
FastText LID was evaluated on 1000 labeled gym reviews (250 per language: en, es, pt, it):
- At τ=0.75: 94.5% coverage with 0.996 accuracy  
- Below τ=0.75: most misclassifications occur  
- Above τ=0.75: minimal accuracy gain but reduced coverage  
Decision: Accept fastText prediction if confidence ≥ 0.75; otherwise fallback to LLM.

**Data Splitting (70/15/15)**  
Stratified sampling on sentiment column ensures proportional class representation. With ~10k reviews and 1500+ minority class samples, splitting is sufficient for initial linear models. Future considerations: augmentation, K-fold CV, or temporal splits.

---

### Status

Preprocessing pipeline is production-ready: deterministic Spanish filtering, LLM fallback, strong observability.  
Model training pipeline is implemented with calibrated LogReg and config-driven reproducibility.  
**Serving API** is implemented with FastAPI for real-time predictions.  
Evaluation data + metrics exist for both LID and sentiment classification.  
Batch-run helpers, dataset merging, and splitting are implemented as CLI commands.  
Test suite covers 16 unit/integration test files across cleaning, language, pipeline, training, and serving modules.

Remaining work: Docker containerization (Phase 2), Cloud Run deployment (Phase 3), CI/CD pipeline (Phase 4).


## **Specific Features and Functions:**

### Preprocessing Pipeline
CLI pipeline stages raw CSVs through expectations → normalization → dedup → language filter → neutral drop → sentiment labeling, emitting `.clean` + `.non_spanish` + `.neutral` artifacts per run.

All file paths resolve via `configs/preprocess.yaml` (`paths.raw_dir|interim_dir|processed_dir|expectations_dir`), keeping repo-relative defaults for reproducibility.

Expectations step enforces required columns (`comment`, `rating`), minimum text length (3 chars), null-drop, and strips emojis (via regex pattern in `cleaning.py`) before any downstream logic.

Normalization lowercases text, replaces structural punctuation (configurable via `configs/structural_punctuation.txt`) with spaces, and collapses whitespace; dedup checks `(comment, rating, name)` to separate duplicate user reviews.

### Language Filtering
`filter_spanish_comments` uses fastText LID (`k=5`) and logs both kept and rejected row counts plus per-review Spanish confidence.

FastText fallback rules: invoke LLM when top-1 confidence < 0.75 or when Spanish appears in top-k above threshold but isn't top-1.

`_predict_languages` returns top label, per-review confidence, and explicit Spanish probability so fallback decisions don't need extra calls.

LLM fallback requests run through `_call_llm_language_detector`, which batches nothing (one review/request) and preserves the public API signature.

LLM FastAPI proxy (`scripts/llm_service.py`) wraps both Google Gemini and Mistral endpoints, logs every request/response, and handles retries/backoff before surfacing HTTP errors. Switch providers by pointing the endpoint to `_call_gemini` or `_call_mistral`.

Rate limiting uses a leaky-bucket governor (configurable RPM, default 60 for Mistral) plus Tenacity retry policy (up to 10 attempts, exponential backoff capped at 60s) to reduce 429 spam.

Non-Spanish exports (`*.non_spanish.csv`) include `es_confidence` so analysts can audit borderline cases.

### Post-Processing
Neutral rating removal (`drop_neutral_ratings` in `cleaning.py`) filters out 3-star reviews and writes them to `.neutral.csv` for analysis.

Sentiment derivation (`add_rating_sentiment` in `features/`) maps ratings to binary labels: 1-2 → `negative`, 4-5 → `positive`.

Name column removal ensures privacy by dropping reviewer names from final processed files.

### Dataset Operations
`merge_processed` CLI command gathers `.clean.csv` files, enforces canonical schema (drops legacy columns like `name`, adds missing ones like `sentiment`, reorders columns), and writes merged dataset. Schema mismatch raises `ValueError` with `merge.schema_mismatch` log.

`split_data` CLI command performs stratified sampling on the sentiment column, writing `train.csv`, `val.csv`, `test.csv` at 70/15/15 ratio with configurable random state.

### Training
`train_from_config` in `training/model.py` reads YAML config, fits TF-IDF + CalibratedClassifierCV(LogisticRegression), evaluates on test set, and writes versioned artifacts with metadata and metrics.

### Serving API
FastAPI application in `src/gym_sentiment_guard/serving/` provides REST API for real-time predictions:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness check (model loaded) |
| `/model/info` | GET | Model metadata (version, threshold) |
| `/predict` | POST | Single prediction with full probabilities |
| `/predict/batch` | POST | Batch predictions (max 100 texts) |

Key components:
- `serving/loader.py`: Model loading with `ModelArtifact` dataclass
- `serving/schemas.py`: Pydantic request/response validation
- `serving/predict.py`: Preprocessing + threshold-based classification
- `serving/app.py`: FastAPI application with startup model loading

Configuration via `configs/serving.yaml`: model path, preprocessing toggle, validation limits (50KB max text), batch limits (100 items max), logging mode (minimal/requests).

### Evaluation
Evaluation tooling (`scripts/eval_fasttext_lid.py`) builds accuracy/precision/recall/F1 metrics, generates a confusion-matrix PNG, and produces JSON summaries of confidence thresholds for fallback policy design.

Language-eval data prep lives in `scripts/` (annotate language column, merge folders, sample 250 per language, merge all languages) to keep main pipeline untouched.

### Logging
Logging everywhere uses structured JSON (`json_log` helper in `utils/logging.py`) so CLI, pipeline, and FastAPI/LLM traces share a consistent format for debugging and automation.

