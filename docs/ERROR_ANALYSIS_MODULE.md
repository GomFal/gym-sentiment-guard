# Error Analysis Module — Technical Context & Task Specification

## Module Name
`error_analysis`

## Module Purpose
This module implements **post-training error analysis** for a TF-IDF (1–3 grams) + Logistic Regression sentiment classifier.

Its purpose is **not** to improve model accuracy directly, but to:
- identify *where* the model fails,
- quantify *risk conditions* under which predictions are less trustworthy,
- explain *why* specific misclassifications occur,
- define *trust boundaries* for deployment,
- produce artifacts usable for monitoring and reporting.

This module operates on **trained artifacts and fixed datasets** and must be:
- deterministic,
- reproducible,
- side-effect free (no training, no tuning).

---

## Scope Constraints (Non-Goals)

The module **must NOT**:
- retrain or fine-tune models,
- modify thresholds based on test results,
- perform linguistic or semantic parsing,
- rely on external NLP services,
- introduce probabilistic heuristics.

All logic must be:
- rule-based,
- transparent,
- computationally cheap.

---

## Inputs

### Required Inputs
- Trained TF-IDF vectorizer (`ngram_range = (1,3)`)
- Trained Logistic Regression classifier
- Optional probability calibrator
- Dataset split (`train`, `val`, or `test`) with:
  - `id`
  - `text`
  - `y_true`

### Configuration Inputs
- Decision threshold(s)
- Near-threshold band
- Low-coverage thresholds
- Contrast keyword list
- Minimum slice size

---

## Outputs (Artifact Contract)

The module MUST generate the following artifacts per run:

```
error_analysis/
└── <run_id>/
    └── <split>/
        ├── error_table.parquet
        ├── ranked_errors/
        │   ├── high_confidence_wrong.csv
        │   ├── top_loss_wrong.csv
        │   └── near_threshold_wrong.csv
        ├── slice_metrics.json
        ├── model_coefficients.json
        ├── example_contributions/
        │   └── <example_id>.json
        ├── KNOWN_LIMITATIONS.md
        └── run_manifest.json
```

All artifacts must be derivable **only** from:
- model artifacts,
- dataset split,
- configuration.

---

## Core Concepts (Terminology)

### Model Coefficients
- Fixed weights learned during training
- One coefficient per n-gram feature
- Global and invariant across samples

### Example-Level Feature Contributions
Computed at inference:

```
contribution = tfidf_value × model_coefficient
```

- Used to explain individual predictions
- No learning occurs at this stage

### Risk Tags
- Binary, rule-based indicators
- Do NOT represent sentiment
- Indicate *conditions correlated with higher error probability*

---

## TASK SET (IMPLEMENTATION CONTRACT)

---

### TASK 1 — Entry Point & Orchestration

**Objective**  
Provide a single callable entry point that generates all artifacts.

**Requirements**
- One function or CLI command triggers the full pipeline
- Accepts:
  - model bundle
  - dataset split
  - output directory
  - config

**Acceptance Criteria**
- One invocation produces all artifacts
- Re-running with same inputs yields identical outputs

---

### TASK 2 — Model Bundle Loading

**Objective**  
Ensure analysis uses the exact trained artifacts.

**Implementation Requirements**
- Load vectorizer, classifier, calibrator (if any)
- Explicitly handle class ordering
- Compute a stable model hash

**Acceptance Criteria**
- Predictions match prior evaluation exactly
- Model version is logged in manifest

---

### TASK 3 — Error Table Construction (Core Artifact)

**Objective**  
Create a row-wise table describing predictions and risk signals.

**Required Columns**
- `id`
- `text`
- `y_true`
- `y_pred`
- `p_positive`
- `p_negative`
- `abs_margin`
- `loss`
- `is_error`

**Coverage Fields**
- `nnz` — number of non-zero TF-IDF features
- `tfidf_sum`
- `low_coverage` (boolean)

**Acceptance Criteria**
- Stable schema
- Saved as Parquet or CSV
- Used as the base for all downstream steps

---

### TASK 4 — Risk Tag Computation (Minimal Set)

**Objective**  
Assign *risk indicators*, not sentiment labels.

#### Required Tags

1. **near_threshold**
   - Prediction probability within configurable band
   - Indicates uncertainty

2. **low_coverage**
   - Derived from `nnz` and/or `tfidf_sum`
   - Indicates model blindness or drift

3. **has_contrast**
   - True if review contains explicit Spanish contrast markers

**Contrast Keyword List**
```
pero
sin embargo
no obstante
aunque
sino
en cambio
por el contrario
a pesar de
mientras que
por un lado
por otro lado
```

**Implementation Notes**
- Lowercase matching only
- Substring or token-based
- No NLP parsing

**Acceptance Criteria**
- Pure rule-based logic
- Computed for every example
- False positives acceptable

---

### TASK 5 — Misclassification Ranking

**Objective**  
Surface *different failure types*.

#### Rankings to Produce

1. **High-confidence wrong**
   - `is_error == True`
   - `abs_margin ≥ confidence_threshold`

2. **Top-loss wrong**
   - `is_error == True`
   - Sorted by `loss DESC`

3. **Near-threshold wrong**
   - `is_error == True`
   - `near_threshold == True`

**Acceptance Criteria**
- Each ranking saved as CSV
- Includes probabilities and risk tags

---

### TASK 6 — Slice Metrics Engine

**Objective**  
Quantify where error rates increase.

**Slices**
- Overall
- `near_threshold == True`
- `low_coverage == True`
- `has_contrast == True`

**Metrics per Slice**
- sample count
- error rate
- class-specific precision/recall
- mean confidence

**Acceptance Criteria**
- Ignore slices with small `n`
- Results saved as JSON

---

### TASK 7 — Model Coefficient Inspection (Model-Level)

**Objective**  
Document what the model has learned globally.

**Outputs**
- Top positive coefficients
- Top negative coefficients
- Grouped by n-gram length (1/2/3)

**Acceptance Criteria**
- Derived only from model artifacts
- Independent of error subset
- Saved as `model_coefficients.json`

---

### TASK 8 — Example-Level Contribution Analysis (Errors Only)

**Objective**  
Explain *why specific errors occurred*.

**Scope**
- Only selected misclassified examples:
  - high-confidence wrong
  - top-loss wrong

**Method**
```
contribution = tfidf_value × model_coefficient
```

**Outputs**
- Per-example top positive contributors
- Per-example top negative contributors

**Acceptance Criteria**
- Uses global coefficients
- No retraining
- Sparse-safe computation

---

### TASK 9 — KNOWN_LIMITATIONS.md Generation

**Objective**  
Translate metrics into deployment knowledge.

**Required Sections**
- Observed failure conditions
- When not to trust predictions
- Suggested escalation/abstain rules
- Monitoring signals to track
- Future experiments (non-test-tuned)

**Acceptance Criteria**
- Auto-generated
- Human-readable
- Derived from slice metrics + rankings

---

### TASK 10 — Run Manifest & Reproducibility

**Objective**  
Make every report auditable.

**Manifest Fields**
- model hash
- dataset hash
- config snapshot
- timestamp
- code version (if available)

**Acceptance Criteria**
- Any report can be reproduced exactly

---

### TASK 11 — Minimal Test Coverage

**Required Tests**
- Error table schema validation
- Determinism check
- Slice count consistency
- Non-empty contribution outputs

---

## Design Summary

- **Model coefficients** = learned once, global
- **Example contributions** = inference-time explanations
- **Risk tags** = indicators of untrustworthy conditions
- **Slices** = quantified failure regions
- **Escalation** = routing decision, not model output

This module is designed for **production-grade ML systems**, not academic experimentation.
