## General Context and Summary of the project:

Gym Sentiment Guard ingests Google Maps–style gym reviews and produces clean, Spanish-only datasets so a classic ML model (LogReg/SVM) can be trained and run on schedule.  
The repo centers on a Typer CLI (`python -m gym_sentiment_guard.cli main …`) that orchestrates a deterministic pipeline:

- Expectations checks (schema, min text length, emoji stripping)  
- Normalization  
- Dedup on `(comment, rating, name)`  
- fastText language filtering with Gemini fallback  
- Emission of `.clean` processed files and `.non_spanish` audit files  

Paths, thresholds, and toggles live in `configs/preprocess.yaml` to keep runs reproducible without code edits.

---

### Architecture

- `src/gym_sentiment_guard/data/` → cleaners and language logic  
- `pipeline/preprocess.py` → orchestrates pipeline stages  
- `config/` → pydantic config schemas  
- `scripts/` → auxiliary tooling (language-eval prep, fastText eval, FastAPI Gemini proxy, batch CSV processor)  
- Logging is JSON-only so CLI, pipeline stages, and FastAPI responses share a uniform structure.

fastText runs with `k=5`.  
Fallback triggers Gemini when:
- top-1 confidence < threshold, **or**
- Spanish appears in top-k above threshold but isn’t top-1.

`_call_llm_language_detector` handles one-review-per-request, enforced by:
- a leaky-bucket governor  
- Tenacity retries/backoff to stay within Gemini rate limits  

`scripts/llm_service.py` wraps Google’s endpoint, logs status + response, and surfaces errors.

---

### Supporting Services / Pipeline Processes

- **Preprocess CLI** (core pipeline; optional `--skip-language`)  
- **process_pending_csvs.py** → scans `data/raw/` and processes missing files  
- **eval_fasttext_lid.py** → accuracy/precision/recall/F1, threshold coverage, confusion-matrix PNG, JSON summaries  
- Scripts for:
  - annotating ground-truth languages  
  - merging per-language folders  
  - balanced sampling  
  - final merges for evaluation  
- **FastAPI** proxy (`scripts/llm_service.py`) streams structured logs and backs off on 429s

---

### Status

Preprocessing pipeline is production-ready: deterministic Spanish filtering, LLM fallback, strong observability.  
Evaluation data + metrics exist.  
Batch-run helpers and LID assessment tooling are implemented.  
Remaining work: expanded monitoring for Gemini limits and downstream model-training integration.


## **Specific Features and Functions:**

CLI pipeline stages raw CSVs through expectations → normalization → dedup → language filter, emitting `.clean` + `.non_spanish` artifacts per run.

All file paths resolve via `configs/preprocess.yaml` (`paths.raw|interim|processed|expectations`), keeping repo-relative defaults for reproducibility.

Expectations step enforces required columns, minimum text length, null-drop, and strips emojis before any downstream logic.

Normalization lowercases/cleans whitespace; dedup checks `(comment, rating, name)` to keep duplicate user reviews separated.

`filter_spanish_comments` uses fastText LID (`k=3`) and logs both kept and rejected row counts plus per-review Spanish confidence.

FastText fallback rules: invoke Gemini when top-1 confidence < threshold or when Spanish appears in top-3 above threshold but isn’t top-1.

`_predict_languages` returns top label, per-review confidence, and explicit Spanish probability so fallback decisions don’t need extra calls.

Gemini fallback requests run through `_call_llm_language_detector`, which batches nothing (one review/request) and preserves the public API signature.

Gemini FastAPI proxy (`scripts/llm_service.py`) wraps Google’s endpoint, logs every request/response, and handles retries/backoff before surfacing HTTP errors.

Rate limiting uses a leaky-bucket governor plus Tenacity retry policy (up to 10 attempts, exponential backoff capped at 60 s) to reduce 429 spam.

Non-Spanish exports (`*.non_spanish.csv`) include `es_confidence` so analysts can audit borderline cases.

Evaluation tooling (`scripts/eval_fasttext_lid.py`) builds accuracy/precision/recall/F1 metrics, generates a confusion-matrix PNG, and produces JSON/Markdown summaries of confidence thresholds.

Language-eval data prep lives in `scripts/` (annotate language column, merge folders, sample 250 per language, merge all languages) to keep main pipeline untouched.

`process_pending_csvs.py` walks `data/raw/`, finds files without `.clean` counterparts, and sequentially calls the preprocess CLI for unattended batch runs.

Logging everywhere uses structured JSON (`json_log` helper) so CLI, pipeline, and FastAPI/LLM traces share a consistent format for debugging and automation.
