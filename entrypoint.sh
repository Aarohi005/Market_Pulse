#!/bin/sh
# entrypoint.sh — container startup sequence
#
# Steps:
#   1. Ensure data directory exists
#   2. Initialise the database (idempotent)
#   3. Optionally run the pipeline immediately (set SKIP_INITIAL_PIPELINE=true to skip)
#   4. Start Streamlit dashboard

set -e

echo "[entrypoint] Starting Financial News Sentiment Dashboard …"
echo "[entrypoint] DB_PATH=${DB_PATH}"

# ── 1. Data directory ─────────────────────────────────────────────────────────
mkdir -p "$(dirname "${DB_PATH:-/app/data/news_sentiment.db}")"

# ── 2. Initialise DB ──────────────────────────────────────────────────────────
echo "[entrypoint] Initialising database …"
python scripts/init_db.py

# ── 3. Initial pipeline run ───────────────────────────────────────────────────
if [ "${SKIP_INITIAL_PIPELINE:-false}" = "true" ]; then
    echo "[entrypoint] Skipping initial pipeline run (SKIP_INITIAL_PIPELINE=true)"
else
    echo "[entrypoint] Running initial pipeline (set SKIP_INITIAL_PIPELINE=true to skip) …"
    python scripts/run_pipeline.py --skip-market || echo "[entrypoint] Pipeline failed — dashboard will start with existing data"
fi

# ── 4. Start Streamlit ────────────────────────────────────────────────────────
echo "[entrypoint] Starting Streamlit on port ${PORT:-8501} …"
exec streamlit run src/dashboard/app.py \
    --server.port "${PORT:-8501}" \
    --server.address "0.0.0.0" \
    --server.headless true
