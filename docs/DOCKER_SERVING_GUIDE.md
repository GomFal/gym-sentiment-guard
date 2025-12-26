# Docker Containerization for ML Model Serving: A Complete Guide

## Introduction

This guide will teach you how to containerize Machine Learning models for production deployment. By the end, you will understand:

1. **What** Docker is and why it matters for ML
2. **How** each component of our serving module works
3. **Why** we made specific architectural decisions
4. **When** to apply these patterns in your MLE career

---

## Part 1: Docker Fundamentals

### 1.1 What is Docker?

Docker is a platform that allows you to package an application and all its dependencies into a standardized unit called a **container**.

```
┌─────────────────────────────────────────────────┐
│                 Your Laptop                      │
│  ┌─────────────┐  ┌─────────────┐               │
│  │ Container A │  │ Container B │               │
│  │ Python 3.12 │  │ Python 3.9  │               │
│  │ scikit-1.4  │  │ scikit-1.0  │               │
│  └─────────────┘  └─────────────┘               │
│           ↓               ↓                      │
│        Docker Engine                             │
│           ↓                                      │
│        Host OS (Windows/Linux/Mac)              │
└─────────────────────────────────────────────────┘
```

**Key Insight**: Each container is isolated. Container A and Container B can have completely different Python versions and packages without conflict.

### 1.2 Why Docker Matters for ML Engineers

| Problem Without Docker | Solution With Docker |
|------------------------|----------------------|
| "Works on my machine" | Same environment everywhere |
| Dependency conflicts | Isolated dependencies per project |
| Complex setup instructions | `docker run` and it works |
| Production environment differs from dev | Identical containers in both |
| Model version rollback is hard | Tag images with model versions |

**Real MLE Scenario**: You trained a model with `scikit-learn==1.4.0`. Six months later, the production server has `scikit-learn==1.5.2` which changed the pickle format. Your model fails to load. With Docker, you would have baked `scikit-learn==1.4.0` into the image, and it would work forever.

### 1.3 Docker vs Virtual Machines

```
Virtual Machine:                    Docker Container:
┌─────────────────┐                 ┌─────────────────┐
│   Application   │                 │   Application   │
│   Dependencies  │                 │   Dependencies  │
│   Guest OS      │ ← Full OS!      │                 │
│   Hypervisor    │                 │   Docker Engine │
│   Host OS       │                 │   Host OS       │
│   Hardware      │                 │   Hardware      │
└─────────────────┘                 └─────────────────┘
     ~10 GB                              ~500 MB
     ~1 min startup                      ~1 sec startup
```

Containers share the host OS kernel, making them much lighter and faster.

### 1.4 Core Docker Concepts

#### Image
A **read-only template** containing the application and its dependencies. Think of it as a "snapshot" or "class definition."

```bash
docker build -t gym-sentiment-guard:v1.0 .
#            └── name:tag (version)
```

#### Container
A **running instance** of an image. Think of it as an "object" created from the "class."

```bash
docker run gym-sentiment-guard:v1.0
#          └── Creates container from image
```

#### Registry
A **storage location** for images (like GitHub for code).
- Docker Hub (public)
- Google Artifact Registry (GCP)
- Amazon ECR (AWS)
- Azure Container Registry

#### Dockerfile
A **text file with instructions** to build an image. This is where you define your environment.

---

## Part 2: The Serving Module Architecture

### 2.1 File Structure Overview

```
src/gym_sentiment_guard/serving/
├── __init__.py      # Module exports (what's public)
├── app.py           # FastAPI application (HTTP layer)
├── loader.py        # Model loading utilities (artifact management)
├── predict.py       # Core prediction logic (ML inference)
└── schemas.py       # Request/Response contracts (API definition)
```

Each file has a **single responsibility**. This is the Single Responsibility Principle (SRP) from software engineering.

### 2.2 File: `loader.py` - Model Artifact Management

**Purpose**: Load the trained model and its metadata from disk.

**Why it's separate**: 
- Model loading is complex (error handling, validation)
- May need to swap loading strategies (local files → cloud storage)
- Isolates ML artifact concerns from HTTP concerns

**Key Components**:

```python
@dataclass(frozen=True)
class ModelArtifact:
    """Container for loaded model and its metadata."""
    model: Pipeline           # The sklearn pipeline
    metadata: dict[str, Any]  # Training info (threshold, version, etc.)
    version: str              # For tracking which model is serving
    threshold: float          # Decision boundary
    target_class: str         # Which class the threshold applies to
    label_mapping: dict       # {"negative": 0, "positive": 1}
    model_name: str           # Human-readable name
```

**Why `@dataclass(frozen=True)`?**
- `frozen=True` makes it immutable (cannot be modified after creation)
- Prevents accidental mutation of model state during serving
- Thread-safe for concurrent requests

**Why Custom Exceptions?**

```python
class ModelLoadError(RuntimeError):
    """Raised when model loading fails."""

class ModelExplainError(RuntimeError):
    """Raised when model does not support explanation."""
```

Custom exceptions allow calling code to handle specific failures differently:

```python
try:
    artifact = load_model(path)
except ModelLoadError:
    # Log error, return 503 Service Unavailable
except ModelExplainError:
    # Log warning, return 400 Bad Request
```

### 2.3 File: `schemas.py` - API Contracts

**Purpose**: Define the exact shape of requests and responses using Pydantic.

**Why Pydantic?**
1. **Automatic Validation**: Invalid requests are rejected before your code runs
2. **Type Coercion**: `"100"` → `100` automatically
3. **Documentation**: OpenAPI/Swagger generated from schemas
4. **Serialization**: Python objects → JSON automatically

**Example Deep Dive**:

```python
class PredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,                    # Required (no default)
        min_length=1,           # At least 1 text
        max_length=100,         # At most 100 texts
        description='List of review texts to classify.',
    )

    @field_validator('texts')
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return v
```

**What happens when a request arrives**:

```
POST /predict {"texts": ["", "valid text"]}
         ↓
Pydantic validates → ValueError at index 0
         ↓
FastAPI catches → Returns 422 Unprocessable Entity
         ↓
{"detail": [{"loc": ["body", "texts"], "msg": "Text at index 0..."}]}
```

Your predict code **never sees invalid data**. This is called "Parse, don't validate."

### 2.4 File: `predict.py` - Core ML Logic

**Purpose**: Transform input text into predictions using the loaded model.

**Why it's separate from `app.py`**:
- Can be unit tested without HTTP
- Can be reused in batch processing scripts
- Keeps ML logic free of HTTP concerns

**Key Design Decisions**:

#### Vectorized Operations

```python
# BAD: Loop over each text (slow)
results = []
for text in texts:
    result = model.predict([text])
    results.append(result)

# GOOD: Batch prediction (fast)
results = model.predict_proba(texts)  # All at once
```

**Why?** NumPy/sklearn operations are optimized for batches. The overhead of calling `predict()` 100 times is much higher than calling it once with 100 items.

#### Preprocessing Consistency

```python
def preprocess_text(texts: list[str], ...) -> list[str]:
    """Apply same preprocessing as training."""
    series = pd.Series(texts)
    processed = (
        series.str.replace(EMOJI_PATTERN, '', regex=True)
        .str.lower()
        .str.replace(pattern, ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return processed.tolist()
```

**Critical MLE Concept**: The preprocessing at inference **must match** training preprocessing. If you lowercased during training but not during inference, the model sees different data distributions and predictions degrade.

#### Handling CalibratedClassifierCV

```python
def _get_classifier_coefficients(classifier) -> np.ndarray:
    # CalibratedClassifierCV wraps multiple estimators
    if hasattr(classifier, 'calibrated_classifiers_'):
        calibrated = classifier.calibrated_classifiers_
        coefs_sum = sum(cc.estimator.coef_ for cc in calibrated)
        coefs_avg = (coefs_sum / len(calibrated)).ravel()
        return coefs_avg
    
    # Direct linear model
    if hasattr(classifier, 'coef_'):
        return classifier.coef_[0]
```

**Why this complexity?** When you use `CalibratedClassifierCV` for probability calibration, it trains K models (one per CV fold). The coefficients are **inside** each fold's estimator. We average them to get a single interpretable coefficient per feature.

### 2.5 File: `app.py` - HTTP Layer

**Purpose**: Handle HTTP requests, validation, and responses.

**Key Patterns**:

#### Lifespan Management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _artifact, _config
    
    # ─── STARTUP ───────────────────────────────
    _config = load_serving_config(config_path)
    _artifact = load_model(_config.model.path)
    log.info("Model loaded successfully")
    
    yield  # ← Application runs here
    
    # ─── SHUTDOWN ──────────────────────────────
    log.info("Shutting down")
```

**Why `@asynccontextmanager`?**
- Guarantees cleanup code runs even if server crashes
- Cleaner than separate `on_startup` and `on_shutdown` handlers
- Variables defined before `yield` are accessible after `yield`

#### Global State for Model

```python
_artifact: ModelArtifact | None = None
_config: ServingConfig | None = None
```

**Why globals?** The model is loaded once at startup and shared across all requests. Alternatives:
- **Dependency Injection**: More testable, but adds complexity
- **`app.state`**: FastAPI's built-in, good for larger apps
- **Globals**: Simple, works well for single-model APIs

#### Health Endpoints

```python
@app.get('/health')      # Liveness: "Is the process running?"
@app.get('/ready')       # Readiness: "Can it serve traffic?"
```

**Why two endpoints?**

Kubernetes/Cloud Run use these differently:
- **Liveness** probe: If fails, container is **killed and restarted**
- **Readiness** probe: If fails, container is **removed from load balancer** (no traffic)

Example: Model is loading (takes 10 seconds). During this time:
- `/health` returns 200 (process is alive)
- `/ready` returns 503 (not ready to serve)

---

## Part 3: Docker Files Explained

### 3.1 Dockerfile - The Build Recipe

```dockerfile
# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies (only needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .
```

#### Multi-Stage Builds

**Problem**: Build tools (compilers, headers) are needed to install some packages but not to run them.

```
Traditional Build:
┌────────────────────────┐
│ Python + Build Tools   │  ← Big image (~1.5 GB)
│ + numpy + sklearn      │
│ + Your App             │
└────────────────────────┘

Multi-Stage Build:
Stage 1 (builder):          Stage 2 (runner):
┌──────────────────┐        ┌──────────────────┐
│ Python + Tools   │   →    │ Python (slim)    │
│ Install packages │  copy  │ Installed pkgs   │
└──────────────────┘        │ Your App         │
     (discarded)            └──────────────────┘
                                 (~500 MB)
```

The `--prefix=/install` puts all installed packages into a directory we can copy to stage 2.

#### Layer Caching

```dockerfile
# Layer 1: Copy dependency file (changes rarely)
COPY pyproject.toml ./

# Layer 2: Install dependencies (cached if pyproject.toml unchanged)
RUN pip install ...

# Layer 3: Copy source code (changes often)
COPY src/ ./src/
```

**Why this order?** Docker caches layers. If `pyproject.toml` hasn't changed, Docker reuses the cached dependencies and skips the slow `pip install`. Only the fast `COPY src/` runs.

```
First build:  pyproject.toml → pip install (5 min) → copy src (1 sec)
Second build: pyproject.toml → [CACHED] (0 sec)  → copy src (1 sec)
```

#### Non-Root User

```dockerfile
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
```

**Why?** Security principle of least privilege:
- If attacker exploits your app, they can't access system files
- Cloud Run requires non-root users
- Many container security scanners flag root users

#### Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
```

**What this means**:
- Every 30 seconds, run the health check
- Wait up to 10 seconds for response
- After container starts, wait 10 seconds before first check
- After 3 consecutive failures, mark container as unhealthy

### 3.2 .dockerignore - Exclude Unnecessary Files

```
# Testing
tests/
.pytest_cache/

# Data (too large)
data/

# Development
*.ipynb
.venv/
```

**Why it matters**:

```
Without .dockerignore:
COPY . /app  ← Copies 2 GB of data files!
Build time: 5 minutes

With .dockerignore:
COPY . /app  ← Copies only 50 MB
Build time: 30 seconds
```

Also, you don't want test files or notebooks in your production image.

### 3.3 docker-compose.yaml - Local Development

```yaml
services:
  api:
    build:
      context: .              # Build from current directory
      dockerfile: Dockerfile  # Using this Dockerfile
    image: gym-sentiment-guard:local
    ports:
      - "8001:8080"           # Host:Container port mapping
    environment:
      - GSG_SERVING_CONFIG=/app/configs/serving.yaml
    healthcheck:
      test: ["CMD", "python", "-c", "..."]
      interval: 30s
```

**Port Mapping Explained**:

```
Your Machine           Docker Container
┌────────────────┐     ┌────────────────┐
│                │     │                │
│  Port 8001 ───────────→ Port 8080    │
│                │     │  (uvicorn)     │
│  Browser       │     │                │
│  curl          │     │                │
└────────────────┘     └────────────────┘

curl localhost:8001/health  →  Reaches container's port 8080
```

### 3.4 serving.docker.yaml - Container Configuration

```yaml
model:
  # Absolute path inside container
  path: /app/artifacts/models/sentiment_logreg/model.2025-12-16_002

preprocessing:
  structural_punctuation_path: /app/configs/structural_punctuation.txt
```

**Why separate config?** Paths that work on your laptop (`../artifacts/models/...`) don't work inside the container. Container paths are absolute from `/app/`.

---

## Part 4: Essential MLE Docker Patterns

### 4.1 Model Versioning Strategy

**Bake Model Into Image** (Our Approach):

```dockerfile
COPY artifacts/models/sentiment_logreg/model.2025-12-16_002 ./artifacts/models/
```

- ✅ Immutable deployments
- ✅ Easy rollback (just deploy older image)
- ✅ No runtime downloads
- ❌ Large images if model is big
- ❌ New model = new image build

**Mount Model at Runtime** (Alternative):

```yaml
volumes:
  - ./artifacts/models:/app/artifacts/models:ro
```

- ✅ Change model without rebuilding
- ✅ Smaller images
- ❌ Dependency on external storage
- ❌ Model version not tracked in image tag

**Recommendation**: For models < 1GB, bake into image. For larger models or frequent updates, use cloud storage (GCS, S3) and download at startup.

### 4.2 Environment Variable Management

**Pattern**: Configuration via environment variables (12-Factor App)

```python
# app.py
config_path = os.getenv('GSG_SERVING_CONFIG', 'configs/serving.yaml')
```

```yaml
# docker-compose.yaml
environment:
  - GSG_SERVING_CONFIG=/app/configs/serving.yaml
  - LOG_LEVEL=INFO
```

**Why?** Same image can run in different environments:
- Development: `LOG_LEVEL=DEBUG`
- Production: `LOG_LEVEL=WARNING`

### 4.3 Logging Best Practices

```python
log.info(json_log(
    'serving.request',
    endpoint='/predict',
    latency_ms=42.5,
    input_count=3,
))
```

**Why JSON logs?**
- Cloud logging (Stackdriver, CloudWatch) parses JSON automatically
- Can query: "Show all requests where latency_ms > 100"
- Structured data is searchable

### 4.4 Graceful Shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup...
    yield
    # Shutdown (runs when container receives SIGTERM)
    log.info("Shutting down gracefully")
```

**Why it matters**: When Cloud Run scales down or deploys new version:
1. Sends `SIGTERM` to container
2. Container has 10 seconds to finish requests
3. `lifespan` shutdown code runs
4. Container exits cleanly

Without graceful shutdown, requests get dropped mid-processing.

---

## Part 5: MLE Interview Knowledge

### 5.1 Common Interview Questions

**Q: How do you ensure training/serving consistency?**
> A: We use the same preprocessing function in both training pipeline and serving module. The sklearn Pipeline object bundles the TF-IDF vectorizer with the model, ensuring identical feature transformation.

**Q: How do you version ML models in production?**
> A: The model directory includes a version string (e.g., `model.2025-12-16_002`). This version is baked into the Docker image tag. We can trace any prediction back to the exact model version via the `model_version` field in responses.

**Q: What's the difference between liveness and readiness probes?**
> A: Liveness checks if the process is running; failure triggers restart. Readiness checks if the service can handle traffic; failure removes it from load balancer but keeps it running (useful during model loading).

**Q: How do you reduce Docker image size?**
> A: Multi-stage builds (separate build dependencies from runtime), slim base images, .dockerignore to exclude unnecessary files, and avoid installing dev dependencies.

**Q: How do you handle model updates without downtime?**
> A: Blue-green deployment or rolling updates. Build new image with new model, deploy alongside old version, gradually shift traffic, monitor metrics, roll back if issues.

### 5.2 Key Metrics to Know

| Metric | Our Target | Why |
|--------|------------|-----|
| Image Size | < 500 MB | Fast pulls, low storage |
| Cold Start | < 10 sec | Acceptable for serverless |
| P99 Latency | < 100 ms | Good UX for real-time apps |
| Memory | < 512 MB | Fits cloud free tiers |

### 5.3 Tools to Know

| Tool | Purpose |
|------|---------|
| Docker | Containerization |
| docker-compose | Local multi-container orchestration |
| Kubernetes / Cloud Run | Production container orchestration |
| Artifact Registry / ECR | Container image storage |
| Prometheus | Metrics collection |
| Grafana | Metrics visualization |

---

## Part 6: Next Steps

Now that you understand the serving module:

1. **Build the image**: `docker build -t gym-sentiment-guard:local .`
2. **Run it**: `docker-compose up`
3. **Test it**: `curl http://localhost:8001/predict -d '{"texts": ["test"]}'`
4. **Explore**: Modify configs, see what breaks, fix it

This hands-on experience will solidify these concepts better than any guide.

---

## Summary

| Concept | Why It Matters |
|---------|----------------|
| Multi-stage builds | Smaller, more secure images |
| Non-root user | Security compliance |
| Health endpoints | Container orchestrator integration |
| Lifespan management | Graceful startup/shutdown |
| Layer caching | Fast rebuilds during development |
| JSON logging | Production observability |
| Environment variables | Configuration flexibility |
| Model versioning | Reproducibility and rollback |

**You now have the knowledge to containerize any ML model and defend your design decisions in MLE interviews.**
