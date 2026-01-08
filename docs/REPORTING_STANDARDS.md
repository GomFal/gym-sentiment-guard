# Ablation Suite Results — Visualization & Conclusions Guide

This document defines the **official visualization and reporting standard** for the
Logistic Regression (TF-IDF + LogReg + Isotonic) ablation suite in the *Gym Sentiment Guard* project.

Its goal is **not** to visualize everything, but to:
- justify **one final model choice** under a strict protocol,
- communicate decisions clearly to technical reviewers,
- reflect **production-grade ML experimentation practices**.

The ablation suite contains **432 runs**. Visualizing all of them directly would obscure insight.
Instead, we use a **four-layer reporting strategy**, from high-level decision evidence to deep model inspection.

---

## Layer 1 — Ablation Summary Table (Primary Artifact)

### Purpose
This is the **single most important artifact** of the experimentation phase.

It answers:
> *“Which configurations are valid, and why did we choose this one?”*

### What it contains
A machine-generated table (`ABLATION_TABLE.csv`) with **all runs**, one row per run.

Minimum required columns:
- `run_id`
- `constraint_met` (Recall_neg ≥ 0.90)
- `F1_neg`
- `F1_pos`
- `Recall_neg`
- `Macro_F1`
- `PR_AUC_neg`
- `threshold`
- `penalty`
- `C`
- `class_weight`
- `ngram_range`
- `min_df`
- `max_df`
- `sublinear_tf`
- `stopwords_enabled`
- `n_features`
- `coef_sparsity`
- `runtime_seconds`

### Sorting rule (critical)
The table **must be sorted** according to the experiment protocol:
1. `constraint_met` (True first)
2. `F1_neg` (descending)
3. `Macro_F1` (descending)
4. `PR_AUC_neg` (descending)
5) **Brier Score** (lower is better) — measures probability calibration quality
6) **ECE (Expected Calibration Error)** (lower is better) — binned calibration error



### Interpretation
- This table is **not meant to be plotted directly**.
- It is the **ground-truth ledger** of experimentation.
- Any visual shown later must be derivable from this table.

### Conclusion expected
At the bottom of this layer:
- Identify the **winner run_id**
- Explicitly state:
  - why it wins under the protocol,
  - which metrics it optimizes,
  - that **no TEST data was used**.

---

## Layer 2 — Top-K Results (Decision-Focused Visualization)

### Purpose
To show that the **winner is not arbitrary**, and that it sits at the top of a small, competitive set.

This layer is about **confidence**, not exploration.

### Selection rules
- Filter runs to: `constraint_met == True`
- Select **Top K = 5–10** runs after protocol sorting
- Never include invalid or constraint-failing runs

### Recommended visualizations

#### 1) Horizontal bar chart — F1_neg (VAL)
- X-axis: F1_neg
- Y-axis: run_id (or short config label)
- Annotate:
  - Recall_neg
  - threshold value

#### 2) Compact comparison table
A Markdown table summarizing:
- run_id
- F1_neg
- Recall_neg
- Macro_F1
- PR_AUC_neg
- Brier Score
- ECE
- key hyperparameters

### Interpretation
This answers:
> *“Are there multiple good solutions, and how close are they?”*

### Conclusion expected
- Explicitly state whether the winner is:
  - clearly dominant, or
  - part of a narrow Pareto frontier
- Justify the final choice using **tie-breaker rules**, not intuition.

---

## Layer 3 — Factor-Level Ablation Analysis (Insight Layer)

### Purpose
This is where **learning happens**.

Instead of comparing runs, we compare **design decisions**.

> Recruiters and senior ML engineers care more about this layer than raw scores.

### Key rule
❌ Do **not** plot run_id vs metric  
✅ Plot **aggregated metrics per factor value**

### Required factor analyses

#### 1) Regularization strength (C)
- Line plot or grouped points:
  - X-axis: log(C)
  - Y-axis: mean F1_neg
- Error bars: ± std

**Conclusion to extract:**
- Under- vs over-regularization regimes
- Sensitivity of performance to C

---

#### 2) sublinear_tf (True vs False)
- Bar plot:
  - mean F1_neg
  - mean PR_AUC_neg
- Count of valid runs per option

**Conclusion to extract:**
- Whether dampening term frequency improves generalization
- Whether effect is consistent or marginal

---

#### 4) ngram_range
- Line plot:
  - X-axis: ngram_range
  - Y-axis: mean F1_neg

**Conclusion to extract:**
- Marginal benefit of higher-order ngrams
- Cost vs gain trade-off (features, runtime)

---

#### 5) Stopword usage
- Paired comparison:
  - stopwords ON vs OFF
- Focus on:
  - constraint satisfaction rate
  - F1_neg stability

**Conclusion to extract:**
- Whether curated stopwords help or are neutral
- Confirmation that sentiment-bearing tokens were preserved

---

### Output
Each factor must end with a **1–2 sentence written conclusion**.

This layer proves:
> *“I understand why the model behaves the way it does.”*

---

## Layer 4 — Final Model Deep Dive (Production Readiness)

### Purpose
To validate that the **selected model is safe, calibrated, and deployable**.

Only **one model** is analyzed here: the final winner.

### Required visualizations

#### 1) Confusion matrices
- VAL confusion matrix (threshold selected on VAL)
- TEST confusion matrix (same threshold)

Interpretation:
- Stability of Recall_neg
- No collapse on TEST

---

#### 2) Precision–Recall curve (Negative class)
- Plot PR curve using `p_neg`
- Mark chosen operating point

Interpretation:
- Visual justification of Recall ≥ 0.90
- Threshold trade-off clarity

---

#### 3) Threshold vs metrics curve
- X-axis: threshold
- Y-axis: Recall_neg and F1_neg
- Highlight chosen threshold

Interpretation:
- Confirms threshold is not brittle

---

#### 4) Calibration curve
- Before vs after isotonic calibration
- Focus on Negative-class probabilities

Interpretation:
- Demonstrates probability reliability
- Justifies calibration choice

---

### Conclusion expected
A concise **Model Readiness Summary**:
- Metric performance (VAL vs TEST)
- Calibration quality
- Operational risks (if any)
- Decision: **approved for deployment**

---

## Reporting Implementation Standard — Plotting Module (Required)

### Purpose
All figures and tables used in Layers 1–4 must be generated by a **single reproducible module**
so the report can be rebuilt deterministically from ablation artifacts.

This module is the **source of truth** for report generation.

---

### Directory & File Outputs (Stable, GitHub-friendly)

```
reports/logreg_ablations/
├── figures/
├── tables/
├── TOP10_RESULTS.md
├── ABLATION_ANALYSIS.md
└── FINAL_MODEL_REPORT.md
```

Figures must use deterministic names (no timestamps):

- `figures/layer2_top10_f1neg.png`
- `figures/layer3_penalty_f1neg_boxplot.png`
- `figures/layer3_C_vs_f1neg.png`
- `figures/layer3_sublinear_tf_effect.png`
- `figures/layer3_ngram_effect.png`
- `figures/layer3_stopwords_effect.png`
- `figures/layer4_val_confusion_matrix.png`
- `figures/layer4_test_confusion_matrix.png`
- `figures/layer4_pr_curve_neg.png`
- `figures/layer4_threshold_curve.png`
- `figures/layer4_calibration_curve.png`

Tables:
- `tables/ablation_table_sorted.csv`
- `tables/top10_table.csv`
- `tables/factor_summary_*.csv`

---

### Recommended Module Location & Entry Point

```
src/gym_sentiment_guard/reports/logreg_ablation_report.py
```

CLI:

```bash
python -m gym_sentiment_guard.reports.logreg_ablation_report   --ablation-table artifacts/experiments/ABLATION_TABLE.csv   --out reports/logreg_ablations
```

---

### Module Responsibilities (Mandatory)

The module must:

1. Validate schema (fail fast if columns are missing)
2. Apply protocol sorting
3. Generate Layer 1 tables
4. Generate Layer 2 plots + tables
5. Generate Layer 3 factor analyses
6. Generate Layer 4 deep-dive plots
7. Write Markdown report skeletons referencing artifacts

Markdown must reference artifacts using **relative paths** only.

---

### Engineering Requirements

- Deterministic outputs
- No notebook-only logic
- Runs from clean checkout
- Runtime < ~60 seconds once artifacts exist

---

## Final Expected Outputs (Checklist)

- `ABLATION_TABLE.csv`
- `TOP10_RESULTS.md`
- `ABLATION_ANALYSIS.md`
- `FINAL_MODEL_REPORT.md`

Only after these are complete should the project proceed to:

> **Linear SVM Baseline (next model family)**

---

## Final Takeaway

> A mature ML project does not prove it tried everything —  
> it proves it made **one correct decision under constraints**, and can explain *why*.

This four-layer structure is how that proof is communicated.
