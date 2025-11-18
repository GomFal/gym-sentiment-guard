# Cleaning Notes — Gym Sentiment Guard

**File:** `docs/CLEANING_NOTES.md`  
**Scope:** Records all data-cleaning decisions that affect dataset content and modeling.

---

## Note 001 — Language Filtering (Spanish-only)

**Decision**  
- Only **Spanish (`es`)** reviews are included in this project.  
- Any review whose detected language is **not Spanish** will be **discarded**.

**Rationale**  
- Narrowing to a single language reduces label noise, vocabulary sparsity, and domain drift.  
- Enables clearer baselines (TF-IDF + linear models) and fairer evaluation for the chosen market.

**Implementation**  
- Add Language ID (LID) during cleaning (fastText `lid.176.bin`).  
- Accept a review if `language == 'es'`. No explicit probability threshold yet, but we log `es_confidence` per row to monitor borderline cases.  
- Rows rejected by LID go to `<stem>.non_spanish.csv` so analysts can review/whitelist if needed.  
- Store fields: `language`, `es_confidence`, `lid_backend`.

**Validation**  
- Expectation: `language ∈ {'es'}` in `data/processed`.  
- Weekly spot-check: sample 50 kept reviews; manual accuracy of LID ≥ 95%.

**Impact**  
- Non-Spanish reviews removed from modeling and metrics; Spanish counts logged alongside rejection counts.  
- Document counts updated in `reports/data_quality/` and `.non_spanish.csv` files created for auditing.

**Version**  
- ID: `CN-001`  
- Date: YYYY-MM-DD  
- Owner: (Javier Gómez)

---

## Note 002 — Preprocess Pipeline Order & Artifacts

Current CLI command: `python -m gym_sentiment_guard.cli.main main preprocess --input <raw.csv> --config configs/preprocess.yaml`

| Step | Function | Output file | Description / Why |
| --- | --- | --- | --- |
| 1 | `enforce_expectations` | `data/interim/<stem>.validated.csv` | Ensures mandatory columns exist, drops rows with null/too-short text. Keeps garbage/empty reviews away from later steps and gives a deterministic baseline row count. |
| 2 | `normalize_comments` | `data/interim/<stem>.normalized.csv` | Lowercases, strips, and collapses whitespace so downstream vectorizers see consistent tokens and dedup operates on canonicalized text. |
| 3 | `deduplicate_reviews` | `data/interim/<stem>.dedup.csv` | Removes duplicate rows (default subset: `comment`, `rating`, optional author ID). Prevents repeated reviews from biasing labels or metrics. |
| 4 | `filter_spanish_comments` | `data/interim/<stem>.spanish.csv` & `data/interim/<stem>.non_spanish.csv` | Runs fastText LID after cleaning, keeps only `es` rows, and logs the rest with `es_confidence`. Outputs: Spanish subset for modeling and audit file for rejected reviews. |
| 5 | Final copy | `data/processed/<stem>.clean.csv` | The CLI copies `<stem>.spanish.csv` into `data/processed/` for training/inference consumption. |

**Rationale**  
- Cleaning before LID removes blank strings and standardizes casing, reducing false negatives from fastText.  
- Each intermediate artifact in `data/interim/` acts as a checkpoint for debugging and reproducibility.  
- Non-Spanish audit files retain the same schema plus `es_confidence` for manual overrides.

**Version**  
- ID: `CN-002`  
- Date: YYYY-MM-DD  
- Owner: (Javier Gómez)
