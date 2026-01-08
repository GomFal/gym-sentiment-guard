---
trigger: always_on
---

# Gym Sentiment Analyzer (NLP, ML, MLOps-lite)

**Source of truth for AI coding agents working in this repo.**  
Goal: a simple, modular, reproducible pipeline that ingests gym reviews, cleans them consistently, trains a classic ML model (LogReg/SVM), and predicts weekly/monthly on new data—without manual changes.

---

## 1) Tech Stack & Architecture Overview

**Languages / Tools**
- Python 3.11+
- ML: scikit-learn, pandas, numpy, scipy
- NLP: scikit-learn `TfidfVectorizer` (baseline); optional spaCy/fastText LID later
- CLI: `typer` or `argparse` (prefer `typer` for UX)
- Config: `yaml` via `pydantic-settings` or `ruamel.yaml`
- Packaging: `pyproject.toml` + `pip install -e .`
- Lint/Format: Ruff (`ruff check`, `ruff format`)
- Tests: pytest + hypothesis (optional)
- Logging: stdlib `logging` (structured, debug mode by env var)

**Suggested Folder Structure**
```
gym-sentiment-guard/
├─ pyproject.toml
├─ README.md
├─ AGENTS.md
├─ docs/
│  └─ CLEANING_NOTES.md       # canonical record of cleaning/preprocess decisions
├─ Makefile
├─ configs/
│  ├─ training.yaml            # model, vectorizer, split, metrics
│  ├─ inference.yaml           # paths, thresholds
│  └─ paths.yaml               # data dirs, models, artifacts
├─ data/
│  ├─ raw/                     # append-only CSVs
│  ├─ interim/                 # dedup/normalized intermediates
│  ├─ processed/               # clean, model-ready CSV/Parquet
│  └─ expectations/            # data quality expectations (YAML/JSON)
├─ artifacts/
│  ├─ models/                  # serialized models (e.g., model_YYYYMMDD.joblib)
│  └─ vectorizers/             # serialized vectorizers
├─ src/
│  └─ gym_sentiment_guard/
│     ├─ __init__.py
│     ├─ cli/                  # CLI entrypoints (preprocess/train/predict)
│     │  └─ main.py
│     ├─ io/                   # readers/writers, path utils
│     ├─ data/                 # cleaning, normalization, LID filters
│     ├─ features/             # tokenization, tf-idf, featurizers
│     ├─ models/               # train/evaluate, save/load, predict
│     ├─ pipeline/             # end-to-end orchestration functions
│     ├─ metrics/              # metrics, reports
│     ├─ config/               # config loading, schemas
│     └─ utils/                # logging, timing, misc
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ data/fixtures/
└─ notebooks/                  # optional, not part of pipeline
```

**High-Level Flow**
1. **Ingest** CSV(s) → `data/raw/`
2. **Preprocess** (clean, dedup, normalize, optional LID) → `data/processed/`
3. **Vectorize** (TF-IDF) → features matrix
4. **Train** (LogReg default; easy switch to SVM) → save model & vectorizer
5. **Predict** on new processed CSVs → write predictions CSV + report
6. **Schedule**: you run CLI weekly/monthly; pipeline is idempotent and modular.

---

## 2) Essential Commands

### Environment / Install
```bash
# (Recommended) create env
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scriptsctivate

# dev install
pip install -e .
pip install pytest ruff typer pydantic pyyaml joblib
```

### Run (Typical)
```bash
# Preprocess a new raw CSV into processed/
python -m gym_sentiment_guard.cli main preprocess   --input data/raw/reviews_2025-11-13.csv   --output data/processed/reviews_2025-11-13.clean.csv

# Train (reads configs/training.yaml)
python -m gym_sentiment_guard.cli main train   --config configs/training.yaml

# Predict on latest processed CSV
python -m gym_sentiment_guard.cli main predict   --config configs/inference.yaml   --input data/processed/reviews_2025-11-13.clean.csv   --output data/processed/reviews_2025-11-13.pred.csv
```

### Lint / Format / Tests
```bash
ruff check .
ruff format .
pytest -q
```

### Debug mode
```bash
# enable verbose debug logs
export GSG_DEBUG=1   # Windows: set GSG_DEBUG=1
```

### Makefile (optional)
```makefile
install:
	pip install -e .
	pip install -r requirements-dev.txt

lint:
	ruff check .

fmt:
	ruff format .

test:
	pytest -q

preprocess:
	python -m gym_sentiment_guard.cli main preprocess --input $(INPUT) --output $(OUTPUT)

train:
	python -m gym_sentiment_guard.cli main train --config configs/training.yaml

predict:
	python -m gym_sentiment_guard.cli main predict --config configs/inference.yaml --input $(INPUT) --output $(OUTPUT)
```

---

## 3) Code Style & Patterns

- **Formatting & Linting**: Use Ruff for lint + format. Run before every commit.
- **Naming**: `snake_case` for functions/vars; `PascalCase` for classes; modules are nouns (`io`, `features`).
- **Module responsibilities**:
  - `io/`: pure read/write & path utilities
  - `data/`: cleaning, normalization, LID (optional)
  - `features/`: vectorizers, featurization
  - `models/`: training, evaluation, save/load, predict
  - `pipeline/`: high-level orchestration for CLI
  - `metrics/`: metrics & reports
  - `config/`: schemas & loaders
  - `utils/`: logging, timing decorators
- **Docstrings**: Public functions/classes must have concise docstrings with args/returns.
- **Doctests**: Optional for small utilities. Prefer pytest for behavior.

**Example: model interface**
```python
class SentimentModel(Protocol):
    def fit(self, X, y) -> "SentimentModel": ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
```

**Switch model easily**
- Expose `--model=logreg|svm` or set in `configs/training.yaml`.
- Keep consistent interfaces; serialize with joblib.

---

## 4) Workflow Rules for Coding Agents

### ❗ PLAN FIRST — NO CODE UNTIL APPROVED
1. Restate task & assumptions.  
2. Propose 2–3 approaches (pros/cons, risks).  
3. Step-by-step plan (no code).  
4. Quick tests/validations you’ll run.  
5. Explicitly ask for approval to implement.

### ❗ IMPLEMENT AFTER APPROVAL
- Write code per plan.
- Add/Update tests.
- Add debug logs on new logic.
- Update docs (README/AGENTS/configs) if behavior changes.
- Run: `ruff check`, `ruff format`, `pytest`.
- Summarize changes & artifacts produced.