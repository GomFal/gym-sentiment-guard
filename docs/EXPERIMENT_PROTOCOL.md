# EXPERIMENT_PROTOCOL.md
## Experiment Protocol — LogReg Ablation (Gym Sentiment)

**Baseline model version pinned:** `2025.12.16_002`

This document is the **authoritative experimental contract** for all Logistic Regression (TF‑IDF + LogReg + Isotonic calibration) ablation runs in this repository.  
Its purpose is to ensure:

- **Comparability**: every run solves the *same* decision problem under the *same* evaluation policy.
- **Reproducibility**: every run can be reconstructed from logged artifacts.
- **Scientific validity**: no leakage or post‑hoc metric cherry‑picking.

> Scope: This protocol governs **model-level experimentation only**. Upstream preprocessing, deduplication, and split generation are assumed already executed and frozen.

---

## 0) Scope & Assumptions (Frozen Inputs)

The following are **frozen and out of scope** for this experiment suite:

- **Dataset content** (already preprocessed)
- **Train/Val/Test split CSVs** (materialized and versioned)
- **Deduplication / leakage prevention** (already enforced upstream)
- **Label construction rules** (already applied upstream; binary neg vs pos)
- **Language filtering** (already applied upstream)

All experiments must train and evaluate on the **same precomputed split files**.

**Hard rule:** Ablation decisions are made using **VAL only**. TEST is evaluated exactly once for the final locked configuration.

---

## 1) Optimization Objective & Selection Metrics

### 1.1 Primary selection objective (VAL)
Select the configuration that **maximizes**:

- **F1 for the Negative class** (`F1_neg`)
- **subject to**: `Recall_neg ≥ 0.90`

This defines the operating regime: the system is intended to **catch negative reviews reliably** while remaining precise.

### 1.2 Tie-breakers (VAL), in order
If two configurations tie on the primary objective:

1) **Macro F1**  
2) **PR AUC (Negative class)**, computed as Average Precision on:

- `p_neg = 1 − p_pos`

> Note: PR AUC for the Positive class may be logged for diagnostics, but **must not** be used for selection.

### 1.3 TEST policy (one-time final only)
- TEST metrics are computed **once** after the winner is locked.
- TEST must never be used to guide: parameter choice, threshold choice, calibration choice, or feature selection.

---

## 2) Fixed Prediction Target & Evaluation Mapping

### 2.1 Class semantics (frozen)
Binary sentiment labels:

- `negative = 0`
- `positive = 1`

### 2.2 Probability definitions (frozen)
Let:

- `p_pos = P(y = positive | x)`
- `p_neg = 1 − p_pos = P(y = negative | x)`

### 2.3 Decision rule (frozen)
Prediction is produced using a single scalar threshold `t` applied to `p_neg`:

- Predict **negative** if `p_neg ≥ t`
- Predict **positive** otherwise

This definition is critical: all thresholding and metrics must be consistent with this rule.

---

## 3) Threshold Selection Policy (VAL-only)

Threshold is **derived only from VAL** for each run, using the rule below.

### 3.1 Candidate enumeration (frozen method)

Candidate thresholds are enumerated using **unique sorted values of `p_neg` from VAL** (plus boundary values 0.0 and 1.0).

This strategy ensures all achievable operating points are evaluated without arbitrary grid spacing.

**Implementation:** [`threshold.py:select_threshold()`](file:///d:/Javier%202.0/Machine%20Learning/Gym%20Sentiment%20Guard/gym-sentiment-guard/src/gym_sentiment_guard/experiments/threshold.py#L63-L67)

### 3.2 Selection objective (frozen)
For each candidate threshold `t`:

1) Apply the decision rule (`p_neg ≥ t → negative`)
2) Compute confusion matrix
3) Compute `Recall_neg`, `Precision_neg`, `F1_neg`, and `Macro F1`

Filter candidates to those satisfying:

- `Recall_neg ≥ 0.90`

Select the candidate threshold that yields:

- maximum `F1_neg`

If multiple thresholds tie on `F1_neg`, apply tie-breakers in order:

1) higher **Macro F1**
2) higher **PR AUC (Negative)**

### 3.3 Constraint failure handling (valid outcome)
If **no threshold** satisfies `Recall_neg ≥ 0.90` on VAL:

- The run is **valid** (not invalidated).
- Mark it as: `constraint_not_met`.
- Record:
  - best achievable `Recall_neg`
  - corresponding `F1_neg`
  - chosen fallback threshold

**Fallback rule (frozen):**
- Choose `t` that **maximizes Recall_neg**.
- If multiple thresholds achieve the same maximum Recall_neg, select among them the `t` that **maximizes F1_neg**.
- If still tied, choose the **smallest `t`**.
- Report the constraint failure explicitly.

---

## 4) Fixed Model Family (What must not change)

All ablation runs must use the following model family components.

### 4.1 Feature extraction (fixed family)
- **TF‑IDF vectorizer** is mandatory.
- Tokenization/analyzer behavior must remain constant except where explicitly ablated (e.g., stopwords list).

Log all TF‑IDF parameters for every run, including:

- `ngram_range`
- `min_df`, `max_df`
- `max_features`
- `sublinear_tf`
- `stop_words` (see §6)

### 4.2 Classifier family (fixed family)
- **Logistic Regression** classifier.

Randomness control (frozen):
- Fix `random_state` for Logistic Regression (even if solver is mostly deterministic).
- Fix all randomness in any CV procedures.

Convergence policy (frozen):
- If a run raises `ConvergenceWarning`, perform **one automatic retry** with a larger `max_iter` (same value for all runs).
- If it still fails to converge, mark run `invalid` (see §8).

### 4.3 Probability calibration (frozen)
- Calibration must be **Isotonic Regression** via `CalibratedClassifierCV` with **5-fold CV**.

Calibration CV splitter (frozen, deterministic):

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`

Calibration must be fit **only on TRAIN** (never on VAL/TEST).  
VAL is reserved for threshold selection and model comparison.

---

## 5) Tunable Parameters (Ablation Surface)

Only the parameters listed here may vary during the ablation suite.

### 5.1 TF‑IDF parameters (candidate grid)
- `ngram_range`: `(1,1)`, `(1,2)`, `(1,3)`
- `min_df`: `1`, `2`, `5`
- `max_df`: `0.90`, `0.95`, `1.0`
- `max_features` (optional): `None` or fixed integer values (must be logged)
- `sublinear_tf`: `False`, `True`
- `norm`: default `"l2"` (treat changes as explicit ablations)

### 5.2 Logistic Regression parameters (candidate grid)
- `penalty`: `"l2"`, `"l1"`
- `C`: `0.1`, `0.3`, `1`, `3`, `10`
- `class_weight`: `None`, `"balanced"`
- `solver` (implementation constraint):
  - L2: `"lbfgs"` (recommended)
  - L1: `"saga"` (required)

> Solver changes required to enable a penalty are *implementation constraints* and do not count as a conceptual multi-factor ablation.

### 5.3 Calibration parameters (frozen)
- Calibration method: isotonic
- CV folds: 5 (deterministic splitter)

Calibration parameters must not be tuned in this experiment suite unless you create a **separate protocol** for calibration ablations.

---

## 6) Stopword Policy (TF‑IDF)

### 6.1 Rationale (sentiment-aware)
Stopword removal can reduce dimensionality and noise, but in **sentiment** it can also destroy signal if it removes:
- negation (`no`, `sin`, `ni`, `nada`)
- contrast (`pero`)
- intensifiers (`muy`, `más`, `súper`)
- quantity modifiers (`poco`, `bastante`)

Therefore:
- Generic Spanish stopword lists **must not** be used blindly.
- This experiment suite uses a **curated, sentiment-safe stopword list** (below).
- Any stopword ablation must be explicitly logged as a change in `stop_words`.

### 6.2 Curated stopword list (frozen for this suite)
The following tokens are considered **non-sentiment-bearing function words** in this dataset and are safe to remove for TF‑IDF.

> Format: one token per line.

```
de
el
que
la
en
es
con
las
un
los
me
por
para
lo
una
se
del
al
son
hay
este
te
como
he
mi
ha
tiene
su
todos
cuando
hace
esta
han
desde
parte
día
cada
ir
así
años
llevo
nos
le
vez
todo
muchas
```

### 6.3 Stoplist Rules (Mandatory)

> [!CAUTION]
> **The curated list (§6.2) is the only permitted stoplist for this suite.**

- **Do not use** generic Spanish stoplists (NLTK, spaCy, sklearn defaults, etc.).
- **Do not add or substitute** alternative stoplists under any circumstances.
- **Do not modify** the curated list without protocol amendment and re-approval.

The following **whitelist tokens must always be retained** (never removed, regardless of any future stoplist changes):

`no`, `ni`, `nada`, `sin`, `pero`, `muy`, `más`, `poco`, `solo`, `bastante`, `súper`, `siempre`, `además`

These tokens directly affect polarity or polarity modulation. Their removal would invalidate experiment results.

---

## 7) Metrics to Record (Every Run)

### 7.1 Required on VAL (for selection)
Record all of the following:

- `F1_neg` at selected threshold
- `Recall_neg` at selected threshold
- `Precision_neg` at selected threshold
- `Macro F1`
- `PR AUC (Negative)` (Average Precision on `p_neg`)
- `threshold` selected
- `constraint_status`: `met` / `not_met`

### 7.2 Required on TEST (final run only)
Compute the same metrics once for the locked winner:

- `F1_neg`, `Recall_neg`, `Precision_neg`
- `Macro F1`
- `PR AUC (Negative)`
- selected threshold (carried from VAL policy; not re-optimized on TEST)

### 7.3 Recommended diagnostics (log if available)
- Confusion matrix at chosen threshold
- Full per-class precision/recall/F1
- Support counts per class
- Timing breakdown:
  - TF‑IDF fit time
  - LogReg fit time
  - Calibration time
  - Total runtime
- Model complexity:
  - number of features
  - coefficient sparsity (% non-zero)

---

## 8) Run Validity Rules

A run is **invalid** (excluded from selection) if any of the following occur:

- Model family drift:
  - TF‑IDF not used
  - classifier not Logistic Regression
  - calibration not isotonic(5-fold)
- Randomness drift:
  - seeds or CV fold assignment not reproducible
- Training failure:
  - crash, NaNs, calibration failure
- Convergence failure:
  - after one standardized retry, solver still fails to converge

A run is **valid but underperforming** if:
- It completes successfully but cannot reach `Recall_neg ≥ 0.90` on VAL  
  → mark `constraint_not_met` and apply fallback rule (§3.3).

---

## 9) Ablation Procedure Rules

- Compare configurations using **VAL only**.
- Change **one conceptual factor at a time**. 
- Select exactly **one** winner configuration based on:
  - primary objective + tie-breakers (§1)
- Freeze the winner, then evaluate on TEST once.

---

## 10) Reproducibility Artifacts (Must Persist)

For each run, persist a `run.json` (or equivalent) containing at least:

- `run_id`, timestamp
- `git_commit`
- environment: Python + dependency versions
- dataset/split identifiers (frozen dataset paths)
- full TF‑IDF params (including stop_words)
- full LogReg params (`penalty`, `C`, `class_weight`, `solver`, `max_iter`, `random_state`)
- calibration config (isotonic, cv=5, splitter seed)
- threshold selection outcome:
  - selected threshold
  - constraint met/not met
  - fallback details if not met
- VAL metrics (§7.1)
- TEST metrics (§7.2) only for final winner

**Recommended** to persist:
- `val_predictions.csv` containing: `id`, `y_true`, `p_pos`, `p_neg`
- (final only) `test_predictions.csv`

---

## 11) Finalization Gate (Before Touching TEST)

TEST evaluation is permitted only if:

- Winner configuration is selected on VAL
- Validity checks pass for all contenders
- Threshold policy is applied on VAL and locked (not re-tuned on TEST)
- Baseline metrics for `2025.12.16_002` are stored for reference

---

### Status
✅ Protocol defined (this document)  
🚫 TEST usage locked until VAL winner is frozen
