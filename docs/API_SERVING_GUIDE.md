# ML Model Serving with FastAPI: A Complete Guide

> **For aspiring Machine Learning Engineers**  
> This guide explains the fundamentals of API development for ML model serving, using our sentiment analysis API as a practical example.

---

## Table of Contents

1. [Why APIs Matter for ML Engineers](#1-why-apis-matter-for-ml-engineers)
2. [Core API Concepts](#2-core-api-concepts)
3. [The Serving Module Architecture](#3-the-serving-module-architecture)
4. [Deep Dive: Each File Explained](#4-deep-dive-each-file-explained)
5. [Essential Concepts for MLE](#5-essential-concepts-for-mle)
6. [Production Patterns](#6-production-patterns)
7. [Quick Reference](#7-quick-reference)

---

## 1. Why APIs Matter for ML Engineers

### The Gap Between Training and Value

A trained model sitting on your laptop is worthless to the business. **APIs bridge this gap**.

```
Training Phase          →    Serving Phase
─────────────────            ─────────────────
Notebooks/Scripts            REST API
.joblib file                 HTTP endpoints
Offline                      24/7 available
You run it                   Anyone can use it
```

### Why REST APIs?

**REST (Representational State Transfer)** is the standard way applications communicate over HTTP.

| Why REST? | Explanation |
|-----------|-------------|
| **Universal** | Any language/platform can call HTTP endpoints |
| **Stateless** | Each request is independent (scales easily) |
| **Standardized** | HTTP verbs (GET, POST) + status codes (200, 400, 500) |
| **Debuggable** | Use curl, Postman, browser—no special tools needed |

### The MLE Career Perspective

Companies expect MLEs to:
1. Train models (you already know this)
2. **Deploy models** (this is what we're learning)
3. Monitor models in production
4. Iterate based on real-world performance

> 💡 **Reality**: Many ML projects fail not because the model is bad, but because it never reaches production. Serving skills make you the engineer who ships.

---

## 2. Core API Concepts

### HTTP Methods (Verbs)

| Method | Purpose | Idempotent? | Our API Usage |
|--------|---------|-------------|---------------|
| **GET** | Retrieve data | Yes | `/health`, `/ready`, `/model/info` |
| **POST** | Send data for processing | No | `/predict`, `/predict/batch` |
| PUT | Update/replace resource | Yes | Not used |
| DELETE | Remove resource | Yes | Not used |

**Idempotent** = calling it multiple times has the same effect as calling it once.

### HTTP Status Codes

```
2xx = Success        200 OK, 201 Created
4xx = Client Error   400 Bad Request, 404 Not Found, 422 Validation Error
5xx = Server Error   500 Internal Error, 503 Service Unavailable
```

In our API:
- `200` - Prediction successful
- `400` - Text too large, batch too big
- `422` - Empty text, validation failed (Pydantic)
- `503` - Model not loaded

### Request/Response Cycle

```
Client                              Server
──────                              ──────
POST /predict         ───────────►  Receive request
{"text": "Great gym!"}              Parse JSON body
                                    Validate with Pydantic
                                    Run prediction
{"sentiment": "positive", ...}  ◄─  Return JSON response
```

### JSON: The Universal Data Format

```json
{
  "text": "Excelente gimnasio",
  "nested": {
    "key": "value"
  },
  "list": [1, 2, 3]
}
```

- Human-readable
- Language-agnostic
- Native to Python via `dict`

---

## 3. The Serving Module Architecture

### File Overview

```
serving/
├── __init__.py      # Module exports (what's public)
├── loader.py        # Model loading + artifact container
├── schemas.py       # Request/response validation (Pydantic)
├── predict.py       # Core ML logic (preprocessing + inference)
└── app.py           # FastAPI application (HTTP layer)
```

### Layered Architecture

```
┌─────────────────────────────────────────────────────┐
│                    HTTP Layer                        │
│                     app.py                           │
│   (routes, request handling, error responses)       │
├─────────────────────────────────────────────────────┤
│                 Validation Layer                     │
│                   schemas.py                         │
│   (Pydantic models, input/output contracts)         │
├─────────────────────────────────────────────────────┤
│                  Business Logic                      │
│                   predict.py                         │
│   (preprocessing, model inference, thresholding)    │
├─────────────────────────────────────────────────────┤
│                 Persistence Layer                    │
│                   loader.py                          │
│   (load model from disk, manage artifacts)          │
└─────────────────────────────────────────────────────┘
```

### Why This Separation?

| Principle | Benefit |
|-----------|---------|
| **Single Responsibility** | Each file does ONE thing well |
| **Testability** | Test prediction logic without HTTP server |
| **Maintainability** | Change validation without touching ML code |
| **Reusability** | Use `predict_single()` from CLI, notebooks, or API |

---

## 4. Deep Dive: Each File Explained

---

### 4.1 `__init__.py` — The Public Interface

```python
"""Serving module for gym_sentiment_guard."""

from .app import app
from .loader import ModelArtifact, ModelLoadError, load_model
from .predict import PredictionResult, predict_batch, predict_single, preprocess_text
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    # ... more schemas
)

__all__ = [
    'app',
    'ModelArtifact',
    # ... everything public
]
```

#### What It Does
- **Imports** all public symbols from submodules
- **Exports** them via `__all__` list

#### Why It Matters

```python
# Without __init__.py exports:
from gym_sentiment_guard.serving.predict import predict_single
from gym_sentiment_guard.serving.loader import load_model

# With proper __init__.py:
from gym_sentiment_guard.serving import predict_single, load_model  # Cleaner!
```

#### Key Concept: Public vs Internal

- **Public** (in `__all__`): Part of your API contract, don't break these
- **Internal** (not exported): Free to change without breaking consumers

---

### 4.2 `loader.py` — Model Artifact Management

#### The Problem It Solves

Your trained model is saved as:
```
artifacts/models/sentiment_logreg/model.2025-12-16_002/
├── logreg.joblib      # Serialized sklearn pipeline
└── metadata.json      # Version, threshold, label mapping
```

You need to:
1. Load both files
2. Handle missing/corrupted files gracefully
3. Bundle them into a single usable object

#### The Solution: `ModelArtifact` Dataclass

```python
@dataclass
class ModelArtifact:
    """Container for loaded model and its metadata."""

    model: Pipeline              # The sklearn model
    metadata: dict[str, Any]     # Raw JSON metadata
    version: str                 # "2025-12-16_002"
    threshold: float             # 0.44
    target_class: str            # "negative"
    label_mapping: dict[str, int]  # {"negative": 0, "positive": 1}

    @property
    def model_name(self) -> str:
        return self.metadata.get('model_name', 'unknown')
```

#### Why Dataclasses?

```python
# Without dataclass (verbose, error-prone):
class ModelArtifact:
    def __init__(self, model, metadata, version, threshold, target_class, label_mapping):
        self.model = model
        self.metadata = metadata
        # ... 6 more lines

# With dataclass (concise, auto-generates __init__, __repr__, etc):
@dataclass
class ModelArtifact:
    model: Pipeline
    metadata: dict[str, Any]
    # ... just declare fields
```

#### The `load_model` Function

```python
def load_model(model_dir: str | Path) -> ModelArtifact:
    model_path = Path(model_dir)

    # 1. Defensive checks
    if not model_path.exists():
        raise ModelLoadError(f'Model directory not found: {model_path}')

    # 2. Load files with error handling
    try:
        model = joblib.load(joblib_path)
    except Exception as exc:
        raise ModelLoadError(f'Failed to load model: {exc}') from exc

    # 3. Extract metadata with defaults
    version = metadata.get('version', 'unknown')
    threshold = metadata.get('threshold', 0.5)  # Safe default

    # 4. Log success for observability
    log.info(json_log('model.loaded', version=version, ...))

    # 5. Return unified artifact
    return ModelArtifact(model=model, ...)
```

#### Key Concepts

| Concept | Implementation | Why |
|---------|---------------|-----|
| **Custom Exception** | `ModelLoadError` | Distinguish ML errors from generic errors |
| **Defensive Programming** | Check existence before loading | Clear error messages |
| **Defaults** | `threshold = metadata.get('threshold', 0.5)` | Graceful degradation |
| **Structured Logging** | `json_log('model.loaded', ...)` | Machine-parseable for monitoring |

---

### 4.3 `schemas.py` — Request/Response Validation

#### The Problem It Solves

Users can send ANY JSON to your API:
```json
{"text": ""}                    // Empty!
{"text": "a" * 1000000}         // 1MB text!
{"typo": "good gym"}            // Wrong field name!
{"texts": []}                   // Empty batch!
```

Without validation, your model will:
- Crash on empty strings
- Run out of memory on huge inputs
- Return confusing errors

#### The Solution: Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    """Request schema for single prediction."""

    text: str = Field(
        ...,                    # Required field
        min_length=1,           # At least 1 character
        description='Review text to classify.',
        examples=['Excelente gimnasio, muy limpio.'],
    )

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError('Text cannot be empty or whitespace only')
        return v  # Return original (preserve whitespace for ML)
```

#### How Pydantic Works

```python
# Valid request - Pydantic creates object
request = PredictRequest(text="Great gym!")
print(request.text)  # "Great gym!"

# Invalid request - Pydantic raises ValidationError
request = PredictRequest(text="")
# 422 Unprocessable Entity: Text cannot be empty
```

#### Field Constraints Breakdown

```python
text: str = Field(
    ...,                    # Ellipsis = REQUIRED (no default)
    min_length=1,          # Built-in string constraint
    description='...',     # Shown in Swagger docs
    examples=['...'],      # Shown in Swagger docs
)
```

For numeric fields:
```python
confidence: float = Field(
    ...,
    ge=0.0,   # Greater or equal
    le=1.0,   # Less or equal
)
```

For lists:
```python
texts: list[str] = Field(
    ...,
    min_length=1,     # At least 1 item
    max_length=100,   # At most 100 items
)
```

#### Response Schemas

```python
class PredictResponse(BaseModel):
    """Response schema for single prediction with full probabilities."""

    sentiment: str = Field(..., description='Predicted sentiment.')
    confidence: float = Field(..., ge=0.0, le=1.0)
    probability_positive: float = Field(..., ge=0.0, le=1.0)
    probability_negative: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(...)
```

#### Why Response Schemas?

1. **Documentation**: Auto-generated Swagger shows response format
2. **Validation**: FastAPI validates your response matches the schema
3. **Serialization**: Pydantic converts dataclasses/objects to JSON

#### Key Concepts

| Concept | Implementation | Why |
|---------|---------------|-----|
| **Input Validation** | `field_validator` | Reject bad data early |
| **Type Hints** | `text: str` | Self-documenting + IDE support |
| **Constraints** | `min_length`, `ge`, `le` | Declarative validation |
| **API Contract** | Response schemas | Clients know exactly what to expect |

---

### 4.4 `predict.py` — Core ML Logic

#### The Problem It Solves

You have:
- A trained sklearn Pipeline (TF-IDF + LogReg)
- Training used specific preprocessing (lowercase, punctuation removal)
- You need to apply the SAME preprocessing at inference time

#### The `preprocess_text` Function

```python
def preprocess_text(
    text: str,
    structural_punctuation: str | None = None,
) -> str:
    """
    Apply the same preprocessing as training.

    Steps:
    1. Strip emojis
    2. Lowercase
    3. Replace structural punctuation with spaces
    4. Collapse whitespace
    5. Strip leading/trailing whitespace
    """
    # Strip emojis
    cleaned = EMOJI_PATTERN.sub('', text)

    # Lowercase
    cleaned = cleaned.lower()

    # Replace structural punctuation with spaces
    pattern = structural_punctuation or DEFAULT_STRUCTURAL_PUNCTUATION
    cleaned = re.sub(pattern, ' ', cleaned)

    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Strip
    cleaned = cleaned.strip()

    return cleaned
```

#### Why This Matters: Training-Serving Skew

```
Training:                          Inference (WRONG):
─────────────────                  ─────────────────
"GREAT gym!!!" → "great gym"       "GREAT gym!!!" → (no preprocessing)
                                   
Model learned:                     Model receives:
"great gym" = positive             "GREAT gym!!!" = ???
```

> ⚠️ **Training-Serving Skew**: When preprocessing differs between training and inference, model performance degrades silently.

#### The `predict_single` Function

```python
def predict_single(
    text: str,
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
    structural_punctuation: str | None = None,
) -> PredictionResult:
```

Let's break down the logic:

**Step 1: Preprocessing**
```python
processed_text = (
    preprocess_text(text, structural_punctuation) 
    if apply_preprocessing else text
)
```

**Step 2: Get probabilities**
```python
proba = artifact.model.predict_proba([processed_text])[0]
# proba = [0.13, 0.87]  # [P(negative), P(positive)]
```

**Step 3: Map classes to indices**
```python
classes = artifact.model.classes_  # [0, 1] or ['negative', 'positive']
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
# {0: 0, 1: 1} if numeric labels
```

**Step 4: Apply custom threshold**
```python
threshold = artifact.threshold        # 0.44
target_class = artifact.target_class  # 'negative'

if target_class == 'negative':
    if prob_negative >= threshold:
        sentiment = 'negative'
        confidence = prob_negative
    else:
        sentiment = 'positive'
        confidence = prob_positive
```

#### Why Custom Thresholds?

Default sklearn: `predict()` uses threshold=0.5

But in production:
- **High-stakes negative reviews** → Lower threshold (catch more negatives)
- **Spam detection** → Higher threshold (precision over recall)

Our model targets negative class at 0.44:
```
P(negative) >= 0.44  →  Predict negative
P(negative) < 0.44   →  Predict positive
```

#### The `PredictionResult` Dataclass

```python
@dataclass
class PredictionResult:
    """Result of a single prediction."""

    sentiment: str                # "positive" or "negative"
    confidence: float             # How confident (matches predicted class)
    probability_positive: float   # P(positive)
    probability_negative: float   # P(negative)
```

#### Why `predict_batch` Exists

```python
def predict_batch(texts: list[str], artifact: ModelArtifact) -> list[PredictionResult]:
    # Handle empty list edge case
    if not texts:
        return []
    
    # Vectorized preprocessing
    processed_texts = [preprocess_text(text) for text in texts]
    
    # SINGLE model call for all texts (efficient!)
    probas = artifact.model.predict_proba(processed_texts)
```

**Efficiency**: One `predict_proba([t1, t2, ..., t100])` is MUCH faster than 100 calls to `predict_proba([t1])` due to vectorization in numpy/sklearn.

---

### 4.5 `app.py` — The FastAPI Application

#### What FastAPI Provides

| Feature | Benefit |
|---------|---------|
| **Routing** | Map URL paths to Python functions |
| **Validation** | Automatic Pydantic integration |
| **OpenAPI** | Auto-generated `/docs` Swagger UI |
| **Async Support** | High concurrency (we use sync for sklearn) |
| **Dependency Injection** | Clean way to share resources |

#### Application Setup

```python
from fastapi import FastAPI

app = FastAPI(
    title='Gym Sentiment Guard API',       # Shown in Swagger
    description='Sentiment analysis API for gym reviews.',
    version='0.1.0',                       # API version
    lifespan=lifespan,                     # Startup/shutdown handler
)
```

#### The Lifespan Pattern (Critical for ML!)

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _artifact, _config
    
    # STARTUP: Load model ONCE when server starts
    _config = load_serving_config(config_path)
    _artifact = load_model(_config.model.path)
    
    log.info('serving.ready', model_version=_artifact.version)
    
    yield  # Server runs here, handles requests
    
    # SHUTDOWN: Cleanup (if needed)
    log.info('serving.shutdown')
```

#### Why Load Model at Startup?

```
BAD: Load per request               GOOD: Load once at startup
─────────────────────               ──────────────────────────
Request 1: load(500ms) + predict    Request 1: predict(5ms)
Request 2: load(500ms) + predict    Request 2: predict(5ms)
Request 3: load(500ms) + predict    Request 3: predict(5ms)

Each request: ~505ms                Each request: ~5ms
```

> 💡 **Cold Start**: Loading the model once avoids repeated I/O overhead and is essential for production latency.

#### Global State (A Necessary Evil)

```python
# Global state
_artifact: ModelArtifact | None = None
_config: ServingConfig | None = None
_structural_punctuation: str | None = None
```

**Why globals?** 
- Model must persist across requests
- Lifespan function initializes them
- Endpoints read from them

**Better alternative (for complex apps)**:
- FastAPI's Dependency Injection
- Application state via `app.state`

#### Defining Endpoints

**Health Check (Liveness)**
```python
@app.get('/health', response_model=HealthResponse, tags=['Health'])
def health() -> HealthResponse:
    """Liveness check endpoint."""
    return HealthResponse(status='ok')
```

- `@app.get('/health')` - Decorator maps GET /health to this function
- `response_model=HealthResponse` - FastAPI validates and documents response
- `tags=['Health']` - Groups in Swagger UI

**Prediction Endpoint**
```python
@app.post(
    '/predict',
    response_model=PredictResponse,
    responses={400: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    tags=['Prediction'],
)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a single review."""
```

- `@app.post('/predict')` - POST method (sending data to process)
- `responses={400: ...}` - Document error responses in Swagger
- `request: PredictRequest` - FastAPI auto-parses JSON body via Pydantic

#### Error Handling with HTTPException

```python
from fastapi import HTTPException

def predict(request: PredictRequest) -> PredictResponse:
    if _artifact is None or _config is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
```

| Status | When to Use |
|--------|-------------|
| `400` | Client sent bad data (their fault) |
| `422` | Pydantic validation failed (automatic) |
| `503` | Server not ready (our fault) |

#### Input Validation Beyond Pydantic

```python
def _validate_text_size(text: str, max_bytes: int, context: str = 'text') -> None:
    """Validate text size against limit."""
    text_bytes = len(text.encode('utf-8'))
    if text_bytes > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f'{context} exceeds maximum size of {max_bytes} bytes',
        )
```

**Why?** Pydantic's `max_length` counts characters, not bytes. UTF-8 characters can be 1-4 bytes.

#### Request Logging

```python
def _log_request(endpoint: str, input_length: int, prediction: str | None, latency_ms: float) -> None:
    if _config and _config.logging.mode == 'requests':
        log.info(json_log(
            'serving.request',
            endpoint=endpoint,
            input_length=input_length,
            prediction=prediction,
            latency_ms=round(latency_ms, 2),
        ))
```

**Why configurable?** 
- Development: Log everything for debugging
- Production: Minimal logging (performance + cost)

---

## 5. Essential Concepts for MLE

### 5.1 The Three Probes

Every production ML service needs:

| Probe | Endpoint | Purpose |
|-------|----------|---------|
| **Liveness** | `/health` | "Is the process running?" |
| **Readiness** | `/ready` | "Can it handle requests?" |
| **Model Info** | `/model/info` | "What model is deployed?" |

```python
# Kubernetes uses these:
# - Liveness fails → Restart container
# - Readiness fails → Stop sending traffic
```

### 5.2 Serialization Formats

| Format | Use Case | Library |
|--------|----------|---------|
| **joblib** | sklearn models (handles numpy) | `joblib.dump/load` |
| **pickle** | General Python objects | `pickle` (security risk!) |
| **ONNX** | Cross-framework models | `onnx`, `onnxruntime` |
| **JSON** | Metadata, configs | `json` |

### 5.3 Configuration Management

```yaml
# configs/serving.yaml
model:
  path: artifacts/models/sentiment_logreg/model.2025-12-16_002

preprocessing:
  enabled: true

validation:
  max_text_bytes: 51200  # 50KB
```

**Why config files?**
- Change behavior without code changes
- Different configs for dev/staging/prod
- Version control for reproducibility

### 5.4 Logging Best Practices

```python
# Structured JSON logs (machine-readable)
log.info(json_log(
    'serving.request',
    endpoint='/predict',
    latency_ms=5.2,
    prediction='positive',
))

# Output: {"msg": "serving.request", "endpoint": "/predict", "latency_ms": 5.2, ...}
```

**Why structured?**
- Parse with tools (Elasticsearch, Datadog)
- Filter: `jq '.latency_ms > 100'`
- Alert: "latency_ms > 500 → page on-call"

### 5.5 Type Hints Everywhere

```python
def predict_single(
    text: str,
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
) -> PredictionResult:
```

**Benefits**:
- Self-documenting code
- IDE autocomplete
- Static analysis (mypy)
- Pydantic uses them for validation

---

## 6. Production Patterns

### 6.1 Graceful Degradation

```python
version = metadata.get('version', 'unknown')  # Don't crash if missing
threshold = metadata.get('threshold', 0.5)    # Safe default
```

### 6.2 Early Validation

```python
# Validate BEFORE expensive operations
_validate_text_size(request.text, max_bytes)  # Fast check
result = predict_single(text)                  # Slow ML inference
```

### 6.3 Batching for Throughput

```python
# Bad: N model calls
for text in texts:
    predict_single(text)

# Good: 1 model call
predict_batch(texts)  # Vectorized internally
```

### 6.4 Latency Measurement

```python
start_time = time.perf_counter()
result = predict_single(text)
latency_ms = (time.perf_counter() - start_time) * 1000
```

---

## 7. Quick Reference

### How to Start the Server

```bash
uvicorn gym_sentiment_guard.serving.app:app --reload --port 8001
```

### How to Test Endpoints

```bash
# Health check
curl http://localhost:8001/health

# Single prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Excelente gimnasio!"}'

# Batch prediction
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Muy buen gym", "Pésimo servicio"]}'
```

### Swagger Documentation

Visit `http://localhost:8001/docs` for interactive API documentation.

---

## Summary: The 5 Files in Context

| File | Layer | Responsibility |
|------|-------|----------------|
| `__init__.py` | Interface | Define public API |
| `loader.py` | Persistence | Load model + metadata |
| `schemas.py` | Validation | Define + validate data shapes |
| `predict.py` | ML Logic | Preprocess + infer |
| `app.py` | HTTP | Routes + error handling |

---

## Next Steps

1. **Read the actual code** in `src/gym_sentiment_guard/serving/`
2. **Run the tests**: `pytest tests/unit/test_serving*.py -v`
3. **Start the server** and explore Swagger at `/docs`
4. **Break things**: Send invalid requests, see how errors are handled
5. **Add a feature**: Try adding a `/predict/explain` endpoint that returns feature importance

---

*Generated: 2025-12-17 | For educational purposes*
