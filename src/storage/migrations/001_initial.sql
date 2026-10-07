-- Migration 001 — initial schema
-- Run once via: python scripts/init_db.py
-- All timestamps stored as ISO-8601 UTC strings (SQLite has no native datetime type).
-- UNIQUE constraints make every insert idempotent: INSERT OR IGNORE / INSERT OR REPLACE
-- will silently skip duplicates rather than raising.

PRAGMA journal_mode = WAL;        -- better concurrent read/write
PRAGMA foreign_keys = ON;

-- ── articles ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- content_hash = SHA-256(title + url) — dedup key across re-runs
    content_hash  TEXT NOT NULL,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    source_name   TEXT NOT NULL,
    source_id     TEXT,
    -- published_at is always UTC ISO-8601: "2024-01-15T14:30:00Z"
    published_at  TEXT NOT NULL,
    author        TEXT,
    -- raw_json stores the original payload for auditability / re-processing
    raw_json      TEXT,
    ingested_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (content_hash)          -- primary dedup guard
);

CREATE INDEX IF NOT EXISTS idx_articles_published_at  ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source_name   ON articles (source_name);
CREATE INDEX IF NOT EXISTS idx_articles_url           ON articles (url);

-- ── sentiment_results ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sentiment_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id    INTEGER NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    label         TEXT NOT NULL CHECK (label IN ('positive', 'negative', 'neutral')),
    -- score = softmax confidence of the winning label (0..1)
    score         REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    model_name    TEXT NOT NULL,
    scored_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- one result per (article, model) — allows future model comparison
    UNIQUE (article_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_article_id   ON sentiment_results (article_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_label        ON sentiment_results (label);
CREATE INDEX IF NOT EXISTS idx_sentiment_scored_at    ON sentiment_results (scored_at);

-- ── market_cache ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    date          TEXT NOT NULL,   -- YYYY-MM-DD
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        REAL,
    fetched_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_market_cache_symbol_date ON market_cache (symbol, date);

-- ── refresh_runs ─────────────────────────────────────────────────────────────
-- Audit log: one row per pipeline execution.
CREATE TABLE IF NOT EXISTS refresh_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'running'
                            CHECK (status IN ('running', 'success', 'failed')),
    articles_ingested   INTEGER DEFAULT 0,
    articles_scored     INTEGER DEFAULT 0,
    error_message       TEXT
);
