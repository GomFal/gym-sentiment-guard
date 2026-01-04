# ML Experimentation Module: A Complete Guide

> **For Machine Learning Engineers**  
> This guide explains the architecture and design decisions of our systematic hyperparameter search module, following rigorous scientific methodology.

---

## Table of Contents

1. [Why Systematic Experimentation Matters](#1-why-systematic-experimentation-matters)
2. [The Experiment Protocol](#2-the-experiment-protocol)
3. [Module Architecture](#3-module-architecture)
4. [Deep Dive: Each File Explained](#4-deep-dive-each-file-explained)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Scientific Validity Patterns](#6-scientific-validity-patterns)
7. [Quick Reference](#7-quick-reference)

---

## 1. Why Systematic Experimentation Matters

### The Problem: Ad-Hoc Hyperparameter Tuning

Most ML projects suffer from:

```
Monday:    "Let me try C=1.0..."    → 85% accuracy
Tuesday:   "Maybe C=0.1?"           → 84% accuracy
Wednesday: "What about ngrams?"     → 86% accuracy
Thursday:  "Which run was best?"    → 🤷 No record!
```

**Common Failures**:
- Cherry-picking metrics after seeing results
- Testing on validation set, then "optimizing" on test
- No reproducibility (forgot which parameters worked)
- Leaking test data into model selection

### The Solution: Rigorous Experimentation

```
┌─────────────────────────────────────────────────────────────┐
│                 EXPERIMENT PROTOCOL                          │
│                                                              │
│  1. Define objective BEFORE running experiments              │
│  2. Lock test set (never peek until final evaluation)        │
│  3. Log EVERYTHING (parameters, metrics, artifacts)          │
│  4. Reproducible (same config → same results)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. The Experiment Protocol

Our protocol (`EXPERIMENT_PROTOCOL.md`) defines:

### 2.1 Primary Objective (§1.1)

```
Maximize: F1_neg (F1 score for Negative class)
Subject to: Recall_neg ≥ 0.90
```

**Why this objective?**
- We're building a gym review sentiment analyzer
- **Negative reviews are high-value signals** (potential churn, complaints)
- We want to **catch 90%+ of negative reviews** (high recall)
- While maintaining **reasonable precision** (not too many false alarms)

### 2.2 Decision Rule (§2.3)

```python
# NOT the default sklearn behavior!
if p_neg >= threshold:
    predict = "negative"
else:
    predict = "positive"
```

**Why custom threshold?**
- Default sklearn uses 0.5 threshold
- Our business case requires tuning for negative class detection
- We select threshold on VAL that maximizes F1_neg while maintaining Recall_neg ≥ 0.90

### 2.3 Data Discipline (§0)

```
FROZEN (never touch):
├── Train set → Fit model
├── Val set   → Select threshold, compare configs
└── Test set  → Evaluate ONCE for final winner
```

**Why freeze splits?**
- Prevents data leakage
- Ensures fair comparison across experiments
- Scientific validity (no post-hoc selection bias)

---

## 3. Module Architecture

### 3.1 File Overview

```
experiments/
├── __init__.py      # Public API exports
├── grid.py          # Parameter grids + stopwords
├── threshold.py     # Threshold selection logic
├── metrics.py       # Metrics computation
├── artifacts.py     # Run persistence
├── runner.py        # Single experiment execution
└── ablation.py      # Grid search orchestrator
```

### 3.2 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│                      ablation.py                             │
│   (grid generation, ranking, winner selection)              │
├─────────────────────────────────────────────────────────────┤
│                   Execution Layer                            │
│                      runner.py                               │
│   (pipeline building, training, validation)                 │
├─────────────────────────────────────────────────────────────┤
│                   Selection Layer                            │
│                     threshold.py                             │
│   (threshold enumeration, constraint handling)              │
├─────────────────────────────────────────────────────────────┤
│                   Metrics Layer                              │
│                      metrics.py                              │
│   (F1, Recall, PR-AUC computation)                          │
├─────────────────────────────────────────────────────────────┤
│                   Configuration Layer                        │
│                      grid.py                                 │
│   (parameter grids, stopwords, fixed settings)              │
├─────────────────────────────────────────────────────────────┤
│                   Persistence Layer                          │
│                     artifacts.py                             │
│   (run.json, predictions CSV, git info)                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Why This Separation?

| Principle | Benefit |
|-----------|---------|
| **Single Responsibility** | Each file does ONE thing well |
| **Testability** | Test threshold logic without training models |
| **Reproducibility** | Grid definitions are declarative and version-controlled |
| **Extensibility** | Add new metrics without touching training code |

---

## 4. Deep Dive: Each File Explained

---

### 4.1 `grid.py` — Parameter Space Definition

**Purpose**: Define exactly what parameters can vary during experiments.

#### TF-IDF Parameter Grid (§5.1)

```python
TFIDF_GRID: dict[str, list] = {
    "ngram_range": [(1, 1), (1, 2), (1, 3)],
    "min_df": [1, 2, 5],
    "max_df": [0.90, 0.95, 1.0],
    "sublinear_tf": [False, True],
}
```

**Why these parameters?**

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `ngram_range` | (1,1), (1,2), (1,3) | Unigrams vs bigrams vs trigrams for context |
| `min_df` | 1, 2, 5 | Filter rare words (noise vs signal tradeoff) |
| `max_df` | 0.90, 0.95, 1.0 | Filter common words (stopword-like behavior) |
| `sublinear_tf` | True/False | Log scaling of term frequencies |

#### LogReg Parameter Grid (§5.2)

```python
LOGREG_GRID: dict[str, list] = {
    "penalty": ["l2", "l1"],
    "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "class_weight": [None, "balanced"],
}

SOLVER_BY_PENALTY: dict[str, str] = {
    "l2": "lbfgs",  # Recommended for L2
    "l1": "saga",   # Required for L1
}
```

**Why solver mapping?**

```python
# sklearn constraint: L1 penalty requires specific solvers
LogisticRegression(penalty="l1", solver="lbfgs")  # ❌ ERROR!
LogisticRegression(penalty="l1", solver="saga")   # ✅ Works
```

The solver is an **implementation constraint**, not a hyperparameter choice.

#### Curated Stopwords (§6.2)

```python
STOPWORDS_SAFE: list[str] = [
    "de", "el", "que", "la", "en", "es", "con", ...
]

STOPWORDS_NEVER_REMOVE: list[str] = [
    "no", "ni", "nada", "sin", "pero", "muy", "más", ...
]
```

**Why custom stopwords for sentiment?**

Generic Spanish stopwords would remove:
- **Negation**: "no", "ni", "nada" → destroys polarity
- **Contrast**: "pero" → critical for sentiment shifts
- **Intensifiers**: "muy", "más", "súper" → amplify sentiment

Our curated list keeps these sentiment-bearing words while removing true function words.

---

### 4.2 `threshold.py` — Threshold Selection

**Purpose**: Find the optimal decision threshold on VAL set.

#### The Core Algorithm

```python
def select_threshold(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    recall_constraint: float = 0.90,
) -> ThresholdResult:
```

**Step-by-step logic**:

```
1. Enumerate all unique p_neg values as candidate thresholds
   candidates = [0.0, 0.12, 0.34, 0.56, 0.78, 0.91, 1.0]

2. For each threshold t:
   - Apply decision rule: predict "negative" if p_neg >= t
   - Compute metrics: Recall_neg, Precision_neg, F1_neg

3. Filter candidates where Recall_neg >= 0.90
   valid_candidates = [t for t in candidates if recall(t) >= 0.90]

4. Select t that maximizes F1_neg among valid candidates
   winner = max(valid_candidates, key=lambda t: f1_neg(t))

5. Handle constraint failure:
   if no valid candidates:
       fallback = max(candidates, key=lambda t: recall(t))
       mark as "constraint_not_met"
```

#### Why This Approach?

**Problem with sklearn's default**:

```python
# sklearn default: threshold = 0.5
y_pred = model.predict(X)  # Uses 0.5 internally
```

But we need **custom threshold** for:
1. Business constraints (Recall ≥ 0.90)
2. Class-specific optimization (F1_neg, not accuracy)
3. Calibrated probabilities (isotonic calibration changes distribution)

#### The ThresholdResult Dataclass

```python
@dataclass
class ThresholdResult:
    threshold: float
    f1_neg: float
    recall_neg: float
    precision_neg: float
    macro_f1: float
    constraint_status: Literal["met", "not_met"]
    # Fallback details if constraint not met
    best_achievable_recall_neg: float | None = None
```

**Why track constraint status?**

Some configurations may **never achieve 90% recall** regardless of threshold. This is valuable information:
- Configuration is **valid but underperforming**
- We record what recall IS achievable
- Allows debugging (why did this config fail?)

---

### 4.3 `metrics.py` — Metrics Computation

**Purpose**: Compute all required metrics per protocol §7.

#### ValMetrics Dataclass

```python
@dataclass
class ValMetrics:
    # Required metrics (§7.1)
    f1_neg: float
    recall_neg: float
    precision_neg: float
    macro_f1: float
    pr_auc_neg: float  # Average Precision on p_neg

    # Threshold info
    threshold: float
    constraint_status: str  # "met", "not_met", or "final_test"

    # Diagnostics (§7.3)
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    support_neg: int
    support_pos: int
```

#### PR AUC (Negative) Computation

```python
# y_neg_binary = 1 when actual is negative
y_neg_binary = (y_true == 0).astype(int)
pr_auc_neg = average_precision_score(y_neg_binary, p_neg)
```

**Why Average Precision, not ROC-AUC?**

| Metric | Best For | Our Case |
|--------|----------|----------|
| ROC-AUC | Balanced classes | ❌ We may have imbalance |
| PR-AUC | Imbalanced, focus on positive class | ✅ Focus on "negative" as positive |

PR-AUC is more informative when:
- Classes are imbalanced
- We care about precision-recall tradeoff for specific class

---

### 4.4 `artifacts.py` — Run Persistence

**Purpose**: Save everything needed to reproduce any experiment.

#### RunConfig Dataclass

```python
@dataclass
class RunConfig:
    run_id: str           # "run.2025.01.04_001"
    timestamp: str        # ISO format
    git_commit: str       # For code version
    git_dirty: bool       # Uncommitted changes?
    python_version: str   # "3.12.4"
    sklearn_version: str  # "1.7.2"
    train_path: str
    val_path: str
    test_path: str
    tfidf_params: dict
    logreg_params: dict
    calibration_config: dict
```

**Why all this metadata?**

Six months later:
- "Which sklearn version was this?"
- "Was the code committed?"
- "What exact parameters?"

Without this, you cannot reproduce results.

#### Git Info Extraction

```python
def get_git_info() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"])
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = len(subprocess.check_output(["git", "status", "--porcelain"])) > 0
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}
```

**Why track `git_dirty`?**

```
Experiment 1: git_dirty=False → Reproducible from commit abc123
Experiment 2: git_dirty=True  → ⚠️ Has uncommitted changes!
```

If `git_dirty=True`, the exact code state may be lost.

---

### 4.5 `runner.py` — Single Experiment Execution

**Purpose**: Run one complete experiment from config to metrics.

#### The Pipeline Builder

```python
def _build_pipeline(config: ExperimentConfig) -> Pipeline:
    # 1. TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        stop_words=STOPWORDS_SAFE,
    )

    # 2. Base Classifier
    base_clf = LogisticRegression(
        penalty=config.penalty,
        C=config.C,
        solver=_get_solver(config.penalty),
        random_state=FIXED_PARAMS["random_state"],
    )

    # 3. Calibrated Classifier (§4.3)
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method="isotonic",
        cv=cv_splitter,
    )

    return Pipeline([("tfidf", vectorizer), ("classifier", calibrated)])
```

**Why Isotonic Calibration?**

```
Raw LogReg output:     p=0.7 means... roughly 70% confident?
After calibration:     p=0.7 means exactly 70% of such cases are positive!
```

Isotonic calibration ensures probabilities are **well-calibrated**:
- `predict_proba(X) = 0.8` means 80% of samples with that score are truly positive
- Critical for threshold selection (our core algorithm relies on calibrated probabilities)

#### Convergence Handling (§4.2)

```python
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

        if convergence_warnings:
            # Retry with higher max_iter
            log.warning("Convergence retry", new_max_iter=5000)
            # ... rebuild and retry
except:
    return RunResult(validity_status="invalid", reason="Convergence failed")
```

**Why automatic retry?**

Some parameter combinations (e.g., `L1 + C=0.1`) may need more iterations. Instead of failing immediately:
1. Retry once with higher `max_iter`
2. If still fails, mark as **invalid** (don't use for selection)

---

### 4.6 `ablation.py` — Grid Search Orchestrator

**Purpose**: Run all experiments and select the winner.

#### Grid Generation

```python
def generate_grid_configs(base_config, tfidf_grid, logreg_grid):
    for tfidf_combo in product(*tfidf_grid.values()):
        for logreg_combo in product(*logreg_grid.values()):
            yield ExperimentConfig(
                ngram_range=tfidf_combo["ngram_range"],
                C=logreg_combo["C"],
                ...
            )
```

**Grid Size Calculation**:
```
TF-IDF: 3 × 3 × 3 × 2 = 54 combinations
LogReg: 2 × 5 × 2 = 20 combinations
Total:  54 × 20 = 1,080 experiments
```

#### Ranking Logic (§1)

```python
def rank_results(results: list[RunResult]) -> list[RunResult]:
    def sort_key(r: RunResult) -> tuple:
        # Validity order: valid > constraint_not_met > invalid
        validity_order = 2 if r.validity_status == "valid" else 1 if "constraint" else 0

        # Constraint order: met > not_met
        constraint_order = 1 if r.val_metrics.constraint_status == "met" else 0

        return (
            validity_order,
            constraint_order,
            r.val_metrics.f1_neg,      # Primary: maximize F1_neg
            r.val_metrics.macro_f1,    # Tie-breaker 1
            r.val_metrics.pr_auc_neg,  # Tie-breaker 2
        )

    return sorted(results, key=sort_key, reverse=True)
```

**Why this ordering?**

1. **Valid runs first**: Invalid runs (convergence failure) are excluded
2. **Constraint met first**: Runs meeting Recall ≥ 0.90 beat those that don't
3. **F1_neg**: Primary optimization objective
4. **Tie-breakers**: Macro F1, then PR-AUC (in case of exact ties)

---

## 5. Key Design Decisions

### 5.1 Why No GridSearchCV?

```python
# sklearn's built-in:
GridSearchCV(estimator, param_grid, cv=5, scoring='f1')
```

**We can't use it because**:

| Requirement | GridSearchCV | Our Module |
|-------------|--------------|------------|
| Custom threshold selection | ❌ Uses 0.5 | ✅ Per-config threshold |
| Recall constraint | ❌ Single metric | ✅ Constrained optimization |
| Calibration as part of search | ❌ Separate step | ✅ Integrated |
| Detailed artifact logging | ❌ Limited | ✅ Full run.json |

### 5.2 Why Dataclasses Over Dicts?

```python
# With dict (error-prone):
result = {"f1_neg": 0.85, "recall_neg": 0.91}
print(result["f1_negg"])  # KeyError at runtime!

# With dataclass (type-safe):
@dataclass
class ValMetrics:
    f1_neg: float
    recall_neg: float

result = ValMetrics(f1_neg=0.85, recall_neg=0.91)
print(result.f1_negg)  # IDE catches typo immediately!
```

Benefits:
- Type hints for IDE support
- Immutability (`frozen=True` option)
- Auto-generated `__repr__`, `__eq__`
- Easy serialization via `dataclasses.asdict()`

### 5.3 Why Not Optuna/Ray Tune?

| Tool | Pros | Why We Didn't Use |
|------|------|-------------------|
| Optuna | Smart search, pruning | Adds dependency, overkill for grid |
| Ray Tune | Distributed search | We need custom threshold logic |
| MLflow | Tracking, UI | Heavy for single-model project |

**Our approach**: Minimal dependencies, protocol-compliant, auditable.

### 5.4 Why `itertools.product` for Grid?

```python
from itertools import product

# All combinations of ngram_range × min_df × max_df × ...
for combo in product(*grid.values()):
    yield combo
```

- Pure Python, no dependencies
- Lazy evaluation (memory efficient)
- Deterministic order (reproducible)

---

## 6. Scientific Validity Patterns

### 6.1 No Test Leakage

```python
# In runner.py:
pipeline.fit(X_train, y_train)  # Fit on TRAIN only

# In threshold.py:
threshold = select_threshold(y_val, p_neg_val)  # Select on VAL only

# In ablation.py:
test_metrics = evaluate_winner_on_test(winner)  # TEST only for final winner
```

**The firewall**:
- TRAIN: Model learning
- VAL: Threshold selection + config comparison
- TEST: One-time final evaluation

### 6.2 Reproducibility via Seeds

```python
FIXED_PARAMS: dict = {
    "random_state": 42,  # Same for all runs
}

CALIBRATION_CONFIG: dict = {
    "cv": 5,
    "random_state": 42,  # Deterministic CV splits
}
```

Same config + same data = identical results.

### 6.3 Validity Tracking

```python
validity_status: Literal["valid", "invalid", "constraint_not_met"]
```

Every run is categorized:
- **valid**: Completed, meets constraint
- **constraint_not_met**: Completed, but Recall < 0.90
- **invalid**: Crashed, NaN, convergence failure

Invalid runs are **excluded from ranking**, not hidden.

---

## 7. Quick Reference

### 7.1 Running Experiments

```bash
# Single experiment
from gym_sentiment_guard.experiments import run_single_experiment, ExperimentConfig

config = ExperimentConfig(
    train_path="data/frozen/.../train.csv",
    val_path="data/frozen/.../val.csv",
    ngram_range=(1, 2),
    C=1.0,
)
result = run_single_experiment(config)
print(result.val_metrics.f1_neg)

# Full ablation suite
from gym_sentiment_guard.experiments import run_ablation_suite

suite = run_ablation_suite(base_config)
print(f"Winner: {suite['winner'].config.run_id}")
```

### 7.2 Artifacts Produced

```
artifacts/experiments/
├── run.2025.01.04_001/
│   ├── run.json           # Full config + metrics
│   └── val_predictions.csv # id, y_true, p_pos, p_neg
├── run.2025.01.04_002/
│   └── ...
└── suite.2025.01.04_143022/
    └── suite_summary.json  # Ranked results
```

### 7.3 Key Types

| Type | File | Purpose |
|------|------|---------|
| `ExperimentConfig` | runner.py | Input config for one run |
| `ThresholdResult` | threshold.py | Threshold selection outcome |
| `ValMetrics` | metrics.py | All computed metrics |
| `RunResult` | artifacts.py | Complete run outcome |

---

## Summary

| Component | Purpose | Key Design Choice |
|-----------|---------|-------------------|
| `grid.py` | Define search space | Curated stopwords for sentiment |
| `threshold.py` | Find optimal threshold | Constrained optimization (Recall ≥ 0.90) |
| `metrics.py` | Compute metrics | PR-AUC for imbalanced focus |
| `artifacts.py` | Persist runs | Git commit + full config for reproducibility |
| `runner.py` | Execute one run | Isotonic calibration + convergence retry |
| `ablation.py` | Orchestrate search | Multi-level ranking with tie-breakers |

**You now have a scientifically rigorous experimentation framework that ensures reproducibility, prevents data leakage, and produces auditable results.**
