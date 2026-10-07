# Financial News Sentiment Dashboard

A production-ready pipeline that ingests financial news from NewsAPI and RSS feeds,
scores each headline with FinBERT, stores results in SQLite, and visualises
sentiment trends alongside market index data in a Streamlit dashboard.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Sentiment Model](#sentiment-model)
3. [Local Setup](#local-setup)
4. [Environment Variables](#environment-variables)
5. [Database Initialisation](#database-initialisation)
6. [Running the Pipeline](#running-the-pipeline)
7. [Running the Dashboard](#running-the-dashboard)
8. [Running Tests](#running-tests)
9. [Deploying with Docker](#deploying-with-docker)
10. [Scheduling Daily Refreshes](#scheduling-daily-refreshes)
11. [Operational Limits & Known Weaknesses](#operational-limits--known-weaknesses)

---

## Architecture

```
[NewsAPI]  ──►  newsapi_client.py  ──►
                                       normalizer.py  ──►  deduplicator.py  ──►
[RSS Feeds]──►  rss_client.py     ──►                                          │
                                                                                ▼
                                                                       database.py (articles)
                                                                                │
                                                                         scorer.py (FinBERT)
                                                                                │
                                                                       database.py (sentiment_results)
                                                                                │
[yfinance] ──►  market_client.py  ──►  database.py (market_cache)              │
                                                                                ▼
                                                                       aggregator.py
                                                                                │
                                                                         app.py (Streamlit)
```

**Module responsibilities:**

| Module | Responsibility |
|---|---|
| `config/settings.py` | All env-var config via Pydantic-settings |
| `ingestion/newsapi_client.py` | NewsAPI fetch with retry + malformed-item handling |
| `ingestion/rss_client.py` | RSS/Atom fetch via feedparser with retry |
| `ingestion/market_client.py` | yfinance OHLCV download |
| `normalization/normalizer.py` | Canonical `ArticleRecord`, UTC timestamps, content hash |
| `deduplication/deduplicator.py` | Pre-DB hash + URL dedup |
| `sentiment/scorer.py` | FinBERT batch inference |
| `storage/database.py` | SQLite connection mgmt, upsert helpers |
| `analytics/aggregator.py` | Pandas aggregations for charts |
| `dashboard/app.py` | Streamlit UI |
| `pipeline.py` | Orchestrator — chains all stages |

---

## Sentiment Model

**Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

FinBERT is BERT-base fine-tuned on ~10,000 Financial PhraseBank sentences.
It outputs three classes — **positive**, **negative**, **neutral** — with softmax confidence.

**Why FinBERT over alternatives:**
- VADER misreads finance idioms ("beat estimates" scores neutral, not positive).
- General-purpose sentiment models (Twitter-roBERTa, etc.) are trained on social text, not financial prose.
- FinBERT was trained specifically on analyst reports and Bloomberg/Reuters-style news.

**Known limitations:**
- 512-token input limit. We mitigate this by scoring `headline + first 300 chars of description`.
- English-only. Non-English articles get neutral scores by default.
- Domain shift: may underperform on social-media reposts of financial news.
- Model download is ~440 MB on first run. Set `HF_HOME` to control cache location.

---

## Local Setup

**Prerequisites:** Python 3.11+, pip

```bash
# 1. Clone / download the project
cd financial-news-sentiment

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the env template and fill in your values
cp .env.example .env
# Edit .env — at minimum set NEWS_API_KEY
```

---

## Environment Variables

Copy `.env.example` to `.env` and set the following:

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEWS_API_KEY` | **Yes** | — | NewsAPI key ([free tier](https://newsapi.org/register)) |
| `RSS_FEEDS` | No | Yahoo/Investing.com/MarketWatch | Comma-separated RSS URLs |
| `SENTIMENT_MODEL` | No | `ProsusAI/finbert` | HuggingFace model ID |
| `HF_HOME` | No | `./data/hf_cache` | HuggingFace cache dir |
| `DB_PATH` | No | `./data/news_sentiment.db` | SQLite file path |
| `MARKET_SYMBOLS` | No | `SPY,QQQ,DIA` | Market symbols for overlay |
| `NEWS_API_PAGE_SIZE` | No | `100` | Max articles per NewsAPI query |
| `NEWS_API_TOPICS` | No | `finance,stock market,...` | Comma-separated topics |
| `INGEST_LOOKBACK_DAYS` | No | `7` | Days back to fetch news |
| `LOG_LEVEL` | No | `INFO` | Python logging level |

---

## Database Initialisation

Run once before the first pipeline execution (safe to re-run):

```bash
python scripts/init_db.py
```

This creates the SQLite file at `DB_PATH` with four tables:
- `articles` — raw ingested articles (unique by `content_hash`)
- `sentiment_results` — one row per (article, model) pair
- `market_cache` — daily OHLCV per symbol
- `refresh_runs` — pipeline run audit log

---

## Running the Pipeline

```bash
# Standard run: ingest → score → store → refresh market data
python scripts/run_pipeline.py

# Skip market data refresh (faster, works offline)
python scripts/run_pipeline.py --skip-market

# Dry run: fetch and normalise but do NOT write to DB
python scripts/run_pipeline.py --dry-run
```

**On first run:** FinBERT (~440 MB) will download from HuggingFace Hub into `HF_HOME`.
Subsequent runs use the cached model.

Pipeline logs to stdout. Each run creates a row in `refresh_runs` with start/end time,
article counts, and error details.

---

## Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Dashboard features:**
- KPI cards: total articles, % positive/negative/neutral, avg confidence
- Sentiment trend chart: daily stacked bar + rolling net score line
- Source breakdown: pie chart
- Market overlay: SPY/QQQ/DIA price vs. scaled sentiment
- Top headlines: most-positive and most-negative (by FinBERT confidence)
- Full article table with all filters applied
- Manual refresh button (runs `run_pipeline.py` in a subprocess)

**Sidebar filters:** date range, keyword/ticker, source, label, market symbol, rolling window.

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_storage.py -v
```

Tests use mocked HTTP calls (no real API keys needed) and in-memory SQLite databases.
The FinBERT model is never downloaded during tests — the HuggingFace pipeline is mocked.

---

## Deploying with Docker

### Build and run locally

```bash
# Copy and configure .env
cp .env.example .env
# Set NEWS_API_KEY in .env

# Build image
docker build -t financial-news-sentiment:latest .

# Run (mounts ./data/ for DB + model cache persistence)
docker run -d \
  --name fns-app \
  -p 8501:8501 \
  --env-file .env \
  -e DB_PATH=/app/data/news_sentiment.db \
  -e HF_HOME=/app/data/hf_cache \
  -v "$(pwd)/data:/app/data" \
  financial-news-sentiment:latest
```

Open [http://localhost:8501](http://localhost:8501).

### Using docker-compose

```bash
# Start dashboard
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

### Deploying to a VM / cloud instance

```bash
# On the remote server
git clone <repo>
cd financial-news-sentiment
cp .env.example .env && vim .env   # set NEWS_API_KEY

docker-compose up -d
```

The container is stateless — the `./data/` volume holds all persistent state.
Back up the `data/` directory to preserve the database and model cache.

### Skip the initial pipeline run

By default the container runs `run_pipeline.py` on startup (to populate fresh data).
To skip this (e.g., if you are mounting an existing `data/` volume):

```bash
docker run -e SKIP_INITIAL_PIPELINE=true ...
```

---

## Scheduling Daily Refreshes

### Option A: cron on the host

```cron
# Run at 06:00 UTC daily
0 6 * * * cd /path/to/financial-news-sentiment && \
  python scripts/run_pipeline.py >> /var/log/fns_pipeline.log 2>&1
```

### Option B: Docker scheduler service

Uncomment the `scheduler` service block in `docker-compose.yml`.
It runs `run_pipeline.py` then sleeps 24 hours, looping indefinitely.

### Option C: GitHub Actions / CI

Create a scheduled workflow that runs `run_pipeline.py --skip-market`
and commits (or pushes) the updated SQLite file to a mounted volume or
object storage bucket.

---

## Operational Limits & Known Weaknesses

**NewsAPI free tier:**
- 100 requests/day. With 5 topics × 1 page each = 5 requests per run.
  Daily schedule is safe; more-frequent runs will exhaust the limit.
- Free tier returns articles up to 1 month old only.
- Upgrade to Developer ($449/mo) for historical access and higher rate limits.

**FinBERT:**
- 512-token input limit. Long article bodies are truncated; we score headline + 300-char description which fits comfortably.
- No fine-tuning on MSCI-specific language. If you need higher accuracy for specific domains (e.g., ESG, derivatives), fine-tune FinBERT on in-house labelled data.
- CPU inference is slow at scale (~50–100 articles/second on a modern CPU). Batch sizes of 16 work well on CPU; increase to 64+ with a GPU.

**SQLite:**
- Suitable for single-process deployments up to ~millions of rows. For multi-user or multi-writer scenarios, migrate to PostgreSQL (the schema is portable — no SQLite-specific types are used).

**yfinance:**
- Is a scraper, not an official API. Yahoo Finance occasionally changes its HTML structure, breaking downloads silently. The market overlay degrades gracefully (shows "no data" rather than crashing) if downloads fail.

**Deduplication:**
- The URL canonical form strips common tracking parameters but is not exhaustive. The DB UNIQUE(content_hash) constraint is the true safety net — a collision is impossible for articles with different titles or URLs.

**Dashboard:**
- The "Run Pipeline Now" button runs a subprocess. In Docker, this works because both the pipeline and dashboard share the same container. For production, separate the scheduler into its own container and use the `scheduler` service in `docker-compose.yml`.
