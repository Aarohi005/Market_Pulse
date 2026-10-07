-- =============================================================================
-- Migration 002: RAG upgrade — classification, embeddings, LLM analysis,
--                ensemble sentiment, confidence scoring, topic clustering,
--                cluster membership, and pipeline observability.
-- =============================================================================
-- All statements use CREATE TABLE IF NOT EXISTS so this migration is
-- idempotent and safe to re-run.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. article_classifications
--    Stores the rule-based and/or ML topic/sector classification for each
--    article.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_classifications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id          INTEGER NOT NULL,
    topic               TEXT    NOT NULL,           -- Topic enum string
    sector              TEXT,                        -- Sector enum string (optional)
    asset_class         TEXT,                        -- AssetClass enum string (optional)
    confidence          REAL    NOT NULL DEFAULT 0.0,
    classifier_version  TEXT    NOT NULL DEFAULT 'v1',
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_article_classifications_article_id
    ON article_classifications (article_id);

CREATE INDEX IF NOT EXISTS idx_article_classifications_topic
    ON article_classifications (topic);

CREATE INDEX IF NOT EXISTS idx_article_classifications_sector
    ON article_classifications (sector);


-- ---------------------------------------------------------------------------
-- 2. article_embeddings
--    Stores the dense embedding vector (serialised as JSON or blob) for
--    each article.  Used for similarity search and clustering.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER NOT NULL UNIQUE,
    model_name      TEXT    NOT NULL,               -- e.g. "all-MiniLM-L6-v2"
    embedding_json  TEXT    NOT NULL,               -- JSON array of floats
    dimensions      INTEGER NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_article_embeddings_article_id
    ON article_embeddings (article_id);

CREATE INDEX IF NOT EXISTS idx_article_embeddings_model_name
    ON article_embeddings (model_name);


-- ---------------------------------------------------------------------------
-- 3. llm_analyses
--    Stores the output of LLM-based sentiment / commentary / summary calls.
--    The UNIQUE constraint on (scope_hash, analysis_type, provider, model)
--    prevents duplicate analyses for the same input scope.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identifies the scope of the analysis: hash of article IDs or query string
    scope_hash      TEXT    NOT NULL,

    -- One of: "sentiment", "commentary", "summary"
    analysis_type   TEXT    NOT NULL,

    -- LLM provider and model used
    provider        TEXT    NOT NULL,               -- e.g. "anthropic", "openai", "ollama"
    model           TEXT    NOT NULL,               -- e.g. "claude-3-5-sonnet-20241022"

    -- Results
    sentiment_label TEXT,                           -- positive / negative / neutral
    sentiment_score REAL,                           -- float in [-1, 1]
    confidence      TEXT,                           -- HIGH / MEDIUM / LOW
    key_signals     TEXT,                           -- JSON array of signal strings
    raw_text        TEXT,                           -- Full LLM output text

    -- Optional reference fields
    symbol          TEXT,
    article_id      INTEGER,

    is_fallback     INTEGER NOT NULL DEFAULT 0,     -- 1 if Ollama/API was unavailable
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    latency_ms      REAL,

    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE (scope_hash, analysis_type, provider, model),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_scope_hash
    ON llm_analyses (scope_hash);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_symbol
    ON llm_analyses (symbol);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_analysis_type
    ON llm_analyses (analysis_type);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_provider_model
    ON llm_analyses (provider, model);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_created_at
    ON llm_analyses (created_at);


-- ---------------------------------------------------------------------------
-- 4. ensemble_sentiment
--    Stores the combined ensemble sentiment result for a given article or
--    symbol/date scope.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ensemble_sentiment (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id          INTEGER,                    -- NULL for symbol-level aggregates
    symbol              TEXT,
    date                TEXT,                       -- ISO date YYYY-MM-DD

    finbert_label       TEXT,
    finbert_score       REAL,
    llm_label           TEXT,
    llm_score           REAL,
    keyword_label       TEXT,
    keyword_score       REAL,

    ensemble_label      TEXT    NOT NULL,
    ensemble_score      REAL    NOT NULL,
    effective_weights   TEXT,                       -- JSON dict of weights used

    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ensemble_sentiment_article_id
    ON ensemble_sentiment (article_id);

CREATE INDEX IF NOT EXISTS idx_ensemble_sentiment_symbol
    ON ensemble_sentiment (symbol);

CREATE INDEX IF NOT EXISTS idx_ensemble_sentiment_date
    ON ensemble_sentiment (date);

CREATE INDEX IF NOT EXISTS idx_ensemble_sentiment_symbol_date
    ON ensemble_sentiment (symbol, date);


-- ---------------------------------------------------------------------------
-- 5. confidence_scores
--    Stores the computed confidence score and its component breakdown for
--    a given analysis scope.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS confidence_scores (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ensemble_sentiment_id   INTEGER,                -- optional FK to ensemble result
    symbol                  TEXT,
    date                    TEXT,

    -- Input signals (stored for auditability)
    retrieval_similarity    REAL,
    reranking_score         REAL,
    article_count           INTEGER,
    source_diversity        REAL,
    source_quality          REAL,
    sentiment_agreement     REAL,
    llm_confidence_label    TEXT,

    -- Computed outputs
    final_score             REAL    NOT NULL,       -- 0-100
    confidence_label        TEXT    NOT NULL,       -- HIGH / MEDIUM / LOW
    component_scores_json   TEXT,                   -- JSON dict of normalised components
    weights_json            TEXT,                   -- JSON dict of weights used

    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    FOREIGN KEY (ensemble_sentiment_id)
        REFERENCES ensemble_sentiment(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_confidence_scores_symbol
    ON confidence_scores (symbol);

CREATE INDEX IF NOT EXISTS idx_confidence_scores_date
    ON confidence_scores (date);

CREATE INDEX IF NOT EXISTS idx_confidence_scores_label
    ON confidence_scores (confidence_label);

CREATE INDEX IF NOT EXISTS idx_confidence_scores_ensemble_id
    ON confidence_scores (ensemble_sentiment_id);


-- ---------------------------------------------------------------------------
-- 6. topic_clusters
--    One row per cluster produced by a TopicClusterer run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topic_clusters (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL,            -- pipeline run that created this
    cluster_id          INTEGER NOT NULL,            -- K-Means cluster index
    cluster_label       TEXT    NOT NULL,
    topic_enum          TEXT    NOT NULL,
    dominant_sentiment  TEXT    NOT NULL,
    avg_sentiment_score REAL    NOT NULL,
    size                INTEGER NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_topic_clusters_run_id
    ON topic_clusters (run_id);

CREATE INDEX IF NOT EXISTS idx_topic_clusters_topic_enum
    ON topic_clusters (topic_enum);

CREATE INDEX IF NOT EXISTS idx_topic_clusters_dominant_sentiment
    ON topic_clusters (dominant_sentiment);


-- ---------------------------------------------------------------------------
-- 7. article_cluster_memberships
--    Many-to-one relationship between articles and topic clusters.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_cluster_memberships (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  INTEGER NOT NULL,
    cluster_id  INTEGER NOT NULL,   -- FK to topic_clusters.id
    run_id      INTEGER NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE (article_id, run_id),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES topic_clusters(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_article_cluster_memberships_article_id
    ON article_cluster_memberships (article_id);

CREATE INDEX IF NOT EXISTS idx_article_cluster_memberships_cluster_id
    ON article_cluster_memberships (cluster_id);

CREATE INDEX IF NOT EXISTS idx_article_cluster_memberships_run_id
    ON article_cluster_memberships (run_id);


-- ---------------------------------------------------------------------------
-- 8. pipeline_metrics
--    Stores per-stage metrics recorded by MetricsCollector during a run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    stage           TEXT    NOT NULL,
    metric_name     TEXT    NOT NULL,
    metric_value    REAL    NOT NULL,
    recorded_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_run_id
    ON pipeline_metrics (run_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_stage
    ON pipeline_metrics (stage);

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_run_stage
    ON pipeline_metrics (run_id, stage);

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_metric_name
    ON pipeline_metrics (metric_name);

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_recorded_at
    ON pipeline_metrics (recorded_at);
