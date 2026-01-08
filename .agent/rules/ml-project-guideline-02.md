---
trigger: always_on
---

## 5) Debug Logging Guidelines

**Principles**
- Use stdlib `logging`. Root logger configured in `utils/logging.py`.
- Default level `INFO`; enable `DEBUG` via `GSG_DEBUG=1`.
- Structured message pattern: include `event`, `component`, and key parameters.
- Log at boundaries: input sizes, split stats, model params, metrics, output paths, durations.

**Minimal setup**
```python
# src/gym_sentiment_guard/utils/logging.py
import logging, os, json, sys, time

def _json(msg, **kw):
    base = {"ts": time.time(), "msg": msg, **kw}
    return json.dumps(base, ensure_ascii=False)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid double handlers
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.getenv("GSG_DEBUG") else logging.INFO)
    return logger

# usage
log = get_logger(__name__)
log.info(_json("preprocess.start", component="pipeline", input=str(path)))
```

**What to log (examples)**
- `preprocess.start/end` with counts: raw rows, deduped, filtered.
- `vectorize.params` (ngram range, min_df, max_df, max_df, max_features).
- `train.params` (model type, hyperparams), `train.metrics` (accuracy, F1).
- `predict.batch` (n_samples, input/output files).
- `errors` with exception type, but no PII.

---

## 6) Testing & Validation

**Unit tests (required)**
- Cleaning: lowercase, whitespace normalization, punctuation removal, newline handling.
- Dedup logic & idempotency.
- Vectorizer: stable vocabulary with fixed seeds / configs.
- Model: training returns fitted estimator; serialization roundtrip.
- Predict: deterministic on fixed inputs.

**Integration tests**
- E2E small fixture: raw → processed → train → predict (assert outputs & metrics).
- CLI behaves with exit code 0 and creates files.

**Data/ML-specific checks**
- Split reproducibility with fixed `random_state`.
- Label distribution preserved (stratify if classification).
- Basic performance expectation (e.g., accuracy ≥ baseline on fixture).
- No leakage: preprocessing fit only on train; apply to val/test.

**Performance sanity**
- Log training time and vectorization time on tiny fixture.
- Fail fast if training/predict exceeds simple thresholds (configurable).

**Checklist**
- [ ] All tests pass  
- [ ] New logic covered  
- [ ] Logs inspected (DEBUG for dev)  
- [ ] Lint & format pass  
- [ ] Artifacts versioned/named  

---

## 7) Common Gotchas

- **Preprocessing drift**: Changing cleaning/tokenization invalidates old models. Bump model version & retrain.
- **Leakage**: Never fit vectorizer on full data before splitting.
- **Inconsistent encodings**: Ensure UTF-8 on CSV read/write.
- **Imbalanced labels**: Track class distribution; consider class weights if skewed.
- **Config sprawl**: Centralize defaults in `configs/*.yaml`; do not hardcode paths.
- **Serialization mismatch**: Always save paired vectorizer + model; load together.

---

## 8) @~Mentions / Collaboration Tags

Use these tags in PRs/issues/commit messages:

- `@~data` — ingestion & preprocessing, expectations, dedup, LID.
- `@~nlp` — tokenization, vectorizer params, text normalization choices.
- `@~model` — training/eval, metrics, hyperparams, model switches (LogReg/SVM).
- `@~infra` — packaging, CLI, config, paths, CI, reproducibility.
- `@~ops` — scheduling runs (weekly/monthly), artifacts retention, simple automation.

*(Virtual owners can be assigned later; keep tags consistent.)*

---

## 9) Documentation Rules

- Update **README.md** when user-visible behavior or commands change.
- Update **AGENTS.md** if workflows, conventions, or architecture change.
- Add/maintain **configs/** with comments explaining each field.
- Inline docstrings for non-obvious logic (especially data cleaning and feature rules).

---

## 10) Configuration Conventions

**`configs/training.yaml` (example)**
```yaml
seed: 42
dataset:
  input: data/processed/train.csv
  text_col: review
  label_col: label
split:
  strategy: stratified
  test_size: 0.2
vectorizer:
  type: tfidf
  ngram_range: [1, 2]
  min_df: 2
  max_df: 0.9
  max_features: 50000
model:
  type: logreg   # logreg | svm
  params:
    C: 1.0
    class_weight: balanced
metrics:
  primary: f1
  others: [accuracy, precision, recall]
artifacts:
  model_path: artifacts/models/model.joblib
  vectorizer_path: artifacts/vectorizers/tfidf.joblib
```

**`configs/inference.yaml` (example)**
```yaml
vectorizer_path: artifacts/vectorizers/tfidf.joblib
model_path: artifacts/models/model.joblib
text_col: review
batch_size: 2048
output_proba: true
threshold: 0.5
```

**`configs/paths.yaml`**
```yaml
data_raw: data/raw
data_interim: data/interim
data_processed: data/processed
expectations_dir: data/expectations
```

---

## 11) Data Expectations (lightweight)

Place JSON/YAML expectations in `data/expectations/`, e.g.:
```yaml
reviews_schema:
  required_columns: ["review", "label"]
  text_min_len: 5
  max_null_frac_review: 0.01
  allowed_labels: ["pos", "neg"]
```
Add a check step in preprocessing; warn/fail if violated.

---

## 12) CLI Specifications

**Preprocess**
```
preprocess --input <raw_csv> --output <processed_csv> [--config configs/paths.yaml]
# Steps: read → normalize (lowercase, strip, single-space) → remove weird chars → handle newlines → dedup → expectations → write
```

**Train**
```
train --config configs/training.yaml
# Steps: load config → load dataset → split (stratified) → fit vectorizer on train → train model → eval → save artifacts → metrics report
```

**Predict**
```
predict --config configs/inference.yaml --input <processed_csv> --output <pred_csv>
# Steps: load artifacts → transform → predict (labels & proba) → write CSV with predictions
```

---

## 13) CI/CD (optional but recommended)

GitHub Actions minimal:
- Job 1: `ruff check` + `ruff format --check`
- Job 2: `pytest -q`
- Cache pip, set Python 3.11.

---

## 14) Strong Defaults & Opinions

- Start with **Logistic Regression** + **TF-IDF** (fast, solid baseline). Switch to SVM via config only.
- Keep everything **config-driven**; no hardcoded paths.
- Prefer **simplicity** over abstraction until needed.
- Be **conservative** with refactors; small PRs, strong tests.

---

## 15) Quick Example: Debug Logs

```python
log.info(_json("train.params",
               component="model",
               model_type=cfg.model.type,
               params=cfg.model.params))

log.info(_json("train.metrics",
               component="metrics",
               accuracy=float(acc),
               f1=float(f1)))
```

---

## 16) Agent Execution Template (copy in every PR)

**PLAN**
1) Task restatement & assumptions  
2) Options A/B/C (pros/cons)  
3) Chosen plan (steps)  
4) Quick tests to run  
5) Ask approval

**IMPLEMENT**
- Code + tests + logs + docs  
- Run `ruff check && ruff format && pytest`  
- Artifacts & metrics summary  
- Risks & next steps