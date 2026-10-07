# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install Python deps in an isolated layer so the final image stays smaller.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps:
#   gcc / g++ / cmake  — needed by chromadb's native bindings + scikit-learn
#   libxml2 / libxslt  — needed by feedparser/lxml
#   libopenblas-dev    — BLAS backend for numpy/scikit-learn (faster)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    libxml2-dev \
    libxslt-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example .env.example

# Create data directories (SQLite DB, HF model cache, ChromaDB)
RUN mkdir -p /app/data/chroma_db && chown -R appuser:appgroup /app/data

# Pre-download the sentence-transformers embedding model into the image
# so first-run doesn't hit the network. Skip with --build-arg SKIP_MODEL_DOWNLOAD=1
ARG SKIP_MODEL_DOWNLOAD=0
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/models
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "0" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; \
                   SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi || true

# Streamlit config — disable CORS and set server settings
RUN mkdir -p /home/appuser/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
address = \"0.0.0.0\"\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > /home/appuser/.streamlit/config.toml

USER appuser

# Data volume — mount a host directory here to persist the DB and model cache
VOLUME ["/app/data"]

EXPOSE 8501

# entrypoint.sh handles: init DB → optional pipeline run → start Streamlit
COPY entrypoint.sh /app/entrypoint.sh
# entrypoint must be copied before switching to appuser; chown here
USER root
RUN chmod +x /app/entrypoint.sh && chown appuser:appgroup /app/entrypoint.sh
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
