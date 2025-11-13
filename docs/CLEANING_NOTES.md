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
- Add Language ID (LID) during cleaning.  
- Accept a review if:  
  - `language == 'es'` **and** `lid_confidence ≥ 0.85`.  
  - Otherwise: drop (log count in cleaning report).  
- Store fields: `language`, `lid_confidence`, `lid_backend`.

**Validation**  
- Expectation: `language ∈ {'es'}` in `data/processed`.  
- Weekly spot-check: sample 50 kept reviews; manual accuracy of LID ≥ 95%.

**Impact**  
- Non-Spanish reviews removed from modeling and metrics.  
- Document counts updated in `reports/data_quality/`.

**Version**  
- ID: `CN-001`  
- Date: YYYY-MM-DD  
- Owner: (Javier Gómez)
