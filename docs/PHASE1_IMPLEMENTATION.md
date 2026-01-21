# Phase 1: Create Common Module - Technical Implementation Plan

## Objective
Create `src/gym_sentiment_guard/common/` module containing shared utilities that are model-agnostic: metrics computation, threshold selection, and model protocols.

---

## Dependency Analysis

### `experiments/metrics.py` (199 lines)
**Contents:**
- `ValMetrics` dataclass — metrics computed on VAL/TEST sets
- `compute_brier_score()` — calibration metric
- `compute_ece()` — Expected Calibration Error
- `compute_skill_score()` — improvement over baseline
- `compute_val_metrics()` — main metrics computation
- `compute_test_metrics()` — wrapper for TEST set

**Imported by:**
| File | Import |
|------|--------|
| `experiments/__init__.py:19` | Full module re-export |
| `experiments/runner.py:44` | `compute_val_metrics` |
| `experiments/artifacts.py:21` | `ValMetrics` |
| `experiments/ablation.py:25` | `compute_test_metrics` |

**External deps:** `numpy`, `sklearn.metrics`, `sklearn.calibration`

---

### `experiments/threshold.py` (203 lines)
**Contents:**
- `ThresholdResult` dataclass — threshold selection result
- `select_threshold()` — optimized vectorized threshold selection
- `apply_threshold()` — apply decision rule

**Imported by:**
| File | Import |
|------|--------|
| `experiments/__init__.py:28` | Full module re-export |
| `experiments/runner.py:45` | `apply_threshold`, `select_threshold` |

**External deps:** `numpy` only

---

## Implementation Steps

### Step 1: Create Directory Structure
```bash
mkdir -p src/gym_sentiment_guard/common
```

### Step 2: Create `common/__init__.py`
```python
"""Common utilities shared across model implementations."""

from .metrics import (
    ValMetrics,
    compute_brier_score,
    compute_ece,
    compute_skill_score,
    compute_val_metrics,
    compute_test_metrics,
)
from .threshold import (
    ThresholdResult,
    apply_threshold,
    select_threshold,
)
from .protocols import ModelProtocol

__all__ = [
    'ValMetrics',
    'compute_brier_score',
    'compute_ece',
    'compute_skill_score',
    'compute_val_metrics',
    'compute_test_metrics',
    'ThresholdResult',
    'apply_threshold',
    'select_threshold',
    'ModelProtocol',
]
```

### Step 3: Create `common/protocols.py`
```python
"""Model protocols defining standard interfaces."""

from typing import Protocol, Any
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline


class ModelProtocol(Protocol):
    """Protocol for sentiment model implementations.
    
    All model types (LogReg, SVM, etc.) should implement this interface.
    """
    
    def train(self, config_path: str | Path) -> dict[str, Any]:
        """Train model from config file."""
        ...
    
    def build_pipeline(self, config: Any) -> Pipeline:
        """Build sklearn pipeline for this model type."""
        ...
    
    def predict_proba(self, pipeline: Pipeline, X: Any) -> np.ndarray:
        """Get probability predictions."""
        ...
```

### Step 4: Copy `experiments/metrics.py` → `common/metrics.py`
- Copy file contents (no changes needed, already model-agnostic)

### Step 5: Copy `experiments/threshold.py` → `common/threshold.py`
- Copy file contents (no changes needed, pure numpy logic)

### Step 6: Update Imports in `experiments/`

#### `experiments/__init__.py`
```diff
-from .metrics import (
+from ..common.metrics import (
     ValMetrics,
     compute_brier_score,
     ...
 )
-from .threshold import ThresholdResult, apply_threshold, select_threshold
+from ..common.threshold import ThresholdResult, apply_threshold, select_threshold
```

#### `experiments/runner.py`
```diff
-from .metrics import compute_val_metrics
-from .threshold import apply_threshold, select_threshold
+from ..common.metrics import compute_val_metrics
+from ..common.threshold import apply_threshold, select_threshold
```

#### `experiments/artifacts.py`
```diff
-from .metrics import ValMetrics
+from ..common.metrics import ValMetrics
```

#### `experiments/ablation.py`
```diff
-from .metrics import compute_test_metrics
+from ..common.metrics import compute_test_metrics
```

### Step 7: Keep Backward Compatibility Shims
Keep original files with deprecation re-exports:

#### `experiments/metrics.py` (shim)
```python
"""DEPRECATED: Use gym_sentiment_guard.common.metrics instead."""
import warnings
warnings.warn(
    "Importing from experiments.metrics is deprecated. "
    "Use gym_sentiment_guard.common.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)
from ..common.metrics import *  # noqa: F401, F403
```

#### `experiments/threshold.py` (shim)
```python
"""DEPRECATED: Use gym_sentiment_guard.common.threshold instead."""
import warnings
warnings.warn(
    "Importing from experiments.threshold is deprecated. "
    "Use gym_sentiment_guard.common.threshold instead.",
    DeprecationWarning,
    stacklevel=2,
)
from ..common.threshold import *  # noqa: F401, F403
```

---

## Verification

1. **Lint check:**
   ```bash
   ruff check src/gym_sentiment_guard/common/
   ruff check src/gym_sentiment_guard/experiments/
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -q
   ```

3. **Test CLI commands:**
   ```bash
   gym main run-experiment --help
   gym main run-ablation --help
   ```

---

## Files Changed Summary

| Action | File |
|--------|------|
| CREATE | `common/__init__.py` |
| CREATE | `common/protocols.py` |
| CREATE | `common/metrics.py` |
| CREATE | `common/threshold.py` |
| MODIFY | `experiments/__init__.py` |
| MODIFY | `experiments/runner.py` |
| MODIFY | `experiments/artifacts.py` |
| MODIFY | `experiments/ablation.py` |
| REPLACE | `experiments/metrics.py` (shim) |
| REPLACE | `experiments/threshold.py` (shim) |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Import errors | Low | High | Shim files for backward compat |
| Test failures | Low | Medium | Run full test suite after |
| External code breakage | Medium | Low | Deprecation warnings in shims |
