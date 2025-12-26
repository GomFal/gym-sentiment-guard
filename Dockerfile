# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies (needed for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first (cache optimization)
COPY docker/requirements.txt ./

# Install serving-only dependencies into /install prefix
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: Runner - Minimal production image
# ============================================================
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code (not installed as package, just copied)
COPY --chown=appuser:appuser src/gym_sentiment_guard/ ./gym_sentiment_guard/

# Copy configuration files
COPY --chown=appuser:appuser configs/serving.docker.yaml ./configs/serving.yaml
COPY --chown=appuser:appuser configs/structural_punctuation.txt ./configs/structural_punctuation.txt

# Copy model artifacts
COPY --chown=appuser:appuser artifacts/models/sentiment_logreg/ ./artifacts/models/sentiment_logreg/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    GSG_SERVING_CONFIG=/app/configs/serving.yaml \
    PORT=8080

# Switch to non-root user
USER appuser

# Expose port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run with uvicorn
# - Single worker (Cloud Run scales via container instances)
# - Host 0.0.0.0 to accept external connections
# - Port from environment variable for flexibility
CMD ["sh", "-c", "python -m uvicorn gym_sentiment_guard.serving.app:app --host 0.0.0.0 --port ${PORT}"]

