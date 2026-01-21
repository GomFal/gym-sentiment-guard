# Phase 6: Reorganize CLI - Technical Implementation Plan

## Objective
Split monolithic `cli/main.py` (789 lines) into:
- `cli/pipeline.py` — Data pipeline commands (shared)
- `cli/logreg.py` — LogReg-specific commands

---

## Command Categorization

| Command | Lines | Target |
|---------|-------|--------|
| `preprocess` | 57-112 | `pipeline.py` |
| `preprocess-batch` | 115-172 | `pipeline.py` |
| `merge-processed` | 175-221 | `pipeline.py` |
| `split-data` | 224-271 | `pipeline.py` |
| `run-full-pipeline` | 274-334 | `pipeline.py` |
| `normalize-dataset` | 337-378 | `pipeline.py` |
| `train-model` | 381-396 | `logreg.py` |
| `run-experiment` | 399-507 | `logreg.py` |
| `run-ablation` | 510-609 | `logreg.py` |
| `ablation-report` | 612-684 | `logreg.py` |
| `error-analysis` | 687-783 | `logreg.py` |

---

## Target CLI Structure

```
gym
├─ pipeline              # Shared data commands
│  ├─ preprocess
│  ├─ preprocess-batch
│  ├─ merge
│  ├─ split
│  ├─ full
│  └─ normalize
│
├─ logreg                # LogReg-specific
│  ├─ train
│  ├─ experiment
│  ├─ ablation
│  ├─ ablation-report
│  └─ error-analysis
│
└─ svm                   # [FUTURE]
```

---

## Implementation Steps

### Step 1: Create `cli/pipeline.py`
- Move: `preprocess`, `preprocess_batch`, `merge_processed`, `split_data`, `run_full_pipeline`, `normalize_dataset`
- Move helpers: `_resolve_path`, `_apply_path_overrides`
- Create: `pipeline_app = typer.Typer()`

### Step 2: Create `cli/logreg.py`
- Move: `train_model`, `run_experiment`, `run_ablation`, `ablation_report`, `error_analysis`
- Create: `logreg_app = typer.Typer()`

### Step 3: Refactor `cli/main.py`
```python
from .pipeline import pipeline_app
from .logreg import logreg_app

app = typer.Typer(help='Gym Sentiment Guard CLI')
app.add_typer(pipeline_app, name='pipeline')
app.add_typer(logreg_app, name='logreg')

# Backward compat: keep 'main' as deprecated alias
app.add_typer(pipeline_app, name='main', deprecated=True)
```

---

## Files Summary

| Action | File |
|--------|------|
| CREATE | `cli/pipeline.py` (~320 lines) |
| CREATE | `cli/logreg.py` (~400 lines) |
| MODIFY | `cli/main.py` (~50 lines) |

---

## Verification
```bash
gym --help
gym pipeline --help
gym logreg --help
gym logreg ablation --help
```
