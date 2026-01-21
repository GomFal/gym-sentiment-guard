# Model-Based Modularization Plan

## Current State Analysis

### Source Code Structure (`src/gym_sentiment_guard/`)

| Module | Files | Model-Specific? | Issues |
|--------|-------|-----------------|--------|
| `training/` | `model.py` | ✅ LogReg hardcoded | Single file, `LogisticRegression` inline |
| `experiments/` | 6 files | ✅ LogReg hardcoded | `grid.py`, `runner.py` have LogReg params |
| `reports/` | 6 + `logreg_errors/` (12) | ✅ LogReg-specific | Already partially organized |
| `cli/` | `main.py` (789 lines) | ✅ LogReg commands | Monolithic, model-specific logic inline |
| `features/` | 2 files | ❌ Generic | TF-IDF vectorizers (shared) |
| `pipeline/` | 2 files | ❌ Generic | Data preprocessing (shared) |
| `serving/` | 5 files | ✅ LogReg hardcoded | API loads `logreg.joblib` |

### Output Artifacts Structure

| Directory | Current State | Model Organization |
|-----------|---------------|-------------------|
| `artifacts/models/` | `sentiment_logreg/` | ✅ Already organized |
| `artifacts/experiments/` | Flat (`run.*/`, `suite.*/`) | ❌ Not organized by model |
| `reports/error_analysis/` | Flat structure (232 files) | ❌ Not organized by model |
| `reports/logreg_ablations/` | LogReg-specific | ✅ Already named correctly |

### Config Files (`configs/`)

- `logreg_v1.yaml` to `logreg_v4.yaml` — Model training configs
- `experiment.yaml` — Ablation experiment config
- `error_analysis.yaml` — Error analysis config
- `serving.yaml` — API serving config

---

## Proposed Target Structure

### Source Code

```
src/gym_sentiment_guard/
├─ common/                          # [NEW] Shared utilities
│  ├─ __init__.py
│  ├─ protocols.py                  # ModelProtocol ABC
│  ├─ metrics.py                    # MOVE from experiments/metrics.py
│  ├─ threshold.py                  # MOVE from experiments/threshold.py
│  └─ artifacts.py                  # MOVE from experiments/artifacts.py (generic parts)
│
├─ models/                          # [NEW] Per-model implementations
│  ├─ __init__.py
│  ├─ logreg/                       # [NEW] LogReg-specific
│  │  ├─ __init__.py
│  │  ├─ pipeline.py                # REFACTOR from training/model.py
│  │  ├─ experiments/
│  │  │  ├─ __init__.py
│  │  │  ├─ grid.py                 # MOVE from experiments/grid.py
│  │  │  ├─ runner.py               # MOVE from experiments/runner.py
│  │  │  └─ ablation.py             # MOVE from experiments/ablation.py
│  │  └─ reports/
│  │     ├─ __init__.py
│  │     ├─ ablation_report.py      # MOVE from reports/logreg_ablation_report.py
│  │     └─ errors/                 # MOVE from reports/logreg_errors/
│  │        └─ (12 files)
│  │
│  └─ svm/                          # [FUTURE] SVM-specific
│     ├─ __init__.py
│     ├─ pipeline.py
│     ├─ experiments/
│     │  ├─ grid.py
│     │  ├─ runner.py
│     │  └─ ablation.py
│     └─ reports/
│
├─ cli/                             # Reorganized CLI
│  ├─ __init__.py
│  ├─ main.py                       # Top-level dispatcher
│  ├─ pipeline.py                   # Data pipeline commands (shared)
│  └─ logreg.py                     # [NEW] LogReg-specific: train, experiment, ablation
│
├─ serving/                         # KEEP but parameterize model loading
├─ features/                        # KEEP (shared TF-IDF utilities)
├─ pipeline/                        # KEEP (shared preprocessing)
├─ config/                          # KEEP (config loading)
├─ data/                            # KEEP (data utilities)
├─ io/                              # KEEP (I/O utilities)
└─ utils/                           # KEEP (logging, etc.)
```

### Output Artifacts

```
artifacts/
├─ models/
│  ├─ logreg/                       # RENAME from sentiment_logreg/
│  │  └─ model.YYYY-MM-DD_XXX/
│  └─ svm/                          # [FUTURE]
│
└─ experiments/
   ├─ logreg/                       # [NEW] Move existing runs here
   │  ├─ run.YYYY-MM-DD_XXX/
   │  └─ suite.YYYY-MM-DD_HHMMSS/
   └─ svm/                          # [FUTURE]
```

### Reports Output

```
reports/
├─ logreg/                          # [NEW] All LogReg reports
│  ├─ ablations/                    # RENAME from logreg_ablations/
│  └─ error_analysis/               # MOVE from error_analysis/
└─ svm/                             # [FUTURE]
```

### Config Files

```
configs/
├─ logreg/                          # [NEW] LogReg configs
│  ├─ training_v1.yaml
│  ├─ training_v4.yaml
│  ├─ experiment.yaml               # Ablation config
│  └─ error_analysis.yaml
├─ svm/                             # [FUTURE]
│  └─ ...
├─ preprocess.yaml                  # Shared
└─ serving.yaml                     # Parameterized
```

---

## Implementation Phases

### Phase 1: Create Common Module (Low Risk)
1. Create `src/gym_sentiment_guard/common/` directory
2. Create `protocols.py` with `ModelProtocol` ABC
3. Move shared utilities:
   - `experiments/metrics.py` → `common/metrics.py`
   - `experiments/threshold.py` → `common/threshold.py`
   - `experiments/stopwords.py` → `common/stopwords.py` (stopword utilities)
4. Update imports in `experiments/` to use `common.*`
5. Run tests to verify no regression

### Phase 1b: Move Artifacts to Common (Low Risk)
1. Move `experiments/artifacts.py` → `common/artifacts.py`
2. Rename `logreg_params` → `classifier_params` (model-agnostic naming)
3. Update all references to use new field name
4. Keep `experiments/artifacts.py` as backward-compat shim
5. Update `common/__init__.py` exports

### Phase 2: Create Models Directory Structure (Low Risk)
1. Create `src/gym_sentiment_guard/models/` directory
2. Create `models/logreg/` subdirectory structure
3. Create `models/logreg/experiments/` subdirectory
4. Create `models/logreg/reports/` subdirectory

### Phase 3: Move LogReg Training Module (Medium Risk)
1. Move `training/model.py` → `models/logreg/pipeline.py`
2. Rename function `train_from_config` → `train`
3. Keep `training/` as deprecated shim for backward compat
4. Update CLI imports

### Phase 4: Move LogReg Experiments Module (Medium Risk)
1. Move `experiments/grid.py` → `models/logreg/experiments/grid.py`
2. Move `experiments/runner.py` → `models/logreg/experiments/runner.py`
3. Move `experiments/ablation.py` → `models/logreg/experiments/ablation.py`
4. Keep `experiments/__init__.py` as re-export shim
5. Update CLI and tests

### Phase 5: Move LogReg Reports Module (Medium Risk)
1. Move `reports/logreg_errors/` → `models/logreg/reports/errors/`
2. Move `reports/logreg_ablation_report.py` → `models/logreg/reports/ablation_report.py`
3. Update imports in CLI `error_analysis` and `ablation_report` commands

### Phase 6: Reorganize CLI (Medium Risk)
1. Create `cli/logreg.py` with LogReg-specific commands
2. Extract data pipeline commands to `cli/pipeline.py`
3. Refactor `cli/main.py` to be a dispatcher
4. CLI structure: `gym logreg train`, `gym logreg ablation`, etc.

### Phase 7: Reorganize Artifacts (Data Migration)
1. Create `artifacts/experiments/logreg/` directory
2. Move existing `run.*/` and `suite.*/` into `logreg/`
3. Update `experiment.yaml` to use new output path
4. Rename `artifacts/models/sentiment_logreg/` → `artifacts/models/logreg/`

### Phase 8: Reorganize Reports Output (Data Migration)
1. Create `reports/logreg/` directory
2. Move `reports/error_analysis/` → `reports/logreg/error_analysis/`
3. Move `reports/logreg_ablations/` → `reports/logreg/ablations/`
4. Update `error_analysis.yaml` output paths

### Phase 9: Reorganize Configs (Low Risk)
1. Create `configs/logreg/` directory
2. Move `logreg_v*.yaml` → `configs/logreg/training_v*.yaml`
3. Move `experiment.yaml` → `configs/logreg/experiment.yaml`
4. Move `error_analysis.yaml` → `configs/logreg/error_analysis.yaml`
5. Update CLI default paths

---

## File Movement Summary

| Current Path | Target Path |
|--------------|-------------|
| `training/model.py` | `models/logreg/pipeline.py` |
| `experiments/grid.py` | `models/logreg/experiments/grid.py` |
| `experiments/runner.py` | `models/logreg/experiments/runner.py` |
| `experiments/ablation.py` | `models/logreg/experiments/ablation.py` |
| `experiments/metrics.py` | `common/metrics.py` |
| `experiments/threshold.py` | `common/threshold.py` |
| `experiments/artifacts.py` | `common/artifacts.py` + model-specific |
| `reports/logreg_errors/*` | `models/logreg/reports/errors/*` |
| `reports/logreg_ablation_report.py` | `models/logreg/reports/ablation_report.py` |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import breakage | High | Keep deprecated shims in old locations |
| CLI command changes | Medium | Add aliases for old commands |
| Artifact path changes | Low | Migration script for existing data |
| Test failures | Medium | Run tests after each phase |

---

## Verification Plan

1. **After each phase:**
   - Run `ruff check .` and `ruff format .`
   - Run `pytest -q`
   - Test affected CLI commands manually

2. **End-to-end validation:**
   - `gym logreg train --config configs/logreg/training_v4.yaml`
   - `gym logreg ablation --config configs/logreg/experiment.yaml --max-runs 3`
   - `gym logreg error-analysis --config configs/logreg/error_analysis.yaml`
