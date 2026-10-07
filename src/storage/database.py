"""
database.py — SQLite connection management and CRUD helpers.

Covers both the original schema (migration 001) and the RAG upgrade
(migration 002: classifications, embeddings, LLM analyses, ensemble sentiment,
confidence scores, topic clusters, pipeline metrics).

Design decisions:
- Uses sqlite3 from the stdlib; no ORM. SQL is explicit and reviewable.
- WAL mode is set on every connection for safe concurrent reads.
- All writes use INSERT OR IGNORE / INSERT OR REPLACE to stay idempotent.
- A single helper (get_connection) is used throughout so the WAL pragma
  is always applied.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ── Connection factory ────────────────────────────────────────────────────────

@contextmanager
def get_connection(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a sqlite3 connection with WAL mode and row_factory set.

    Usage::

        with get_connection(settings.db_path) as conn:
            conn.execute(...)
    """
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema bootstrap ─────────────────────────────────────────────────────────

def init_db(db_path: str | Path) -> None:
    """
    Run all SQL migrations against *db_path* in order.

    Safe to call multiple times — all CREATE statements use IF NOT EXISTS.
    Runs migration 001 (base schema) then 002 (RAG upgrade).
    """
    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    conn = sqlite3.connect(str(db_path))
    try:
        for mf in migration_files:
            sql = mf.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.commit()
            logger.debug("Applied migration: %s", mf.name)
    finally:
        conn.close()

    logger.info("Database initialised at %s (%d migrations)", db_path, len(migration_files))


# ── Article helpers ───────────────────────────────────────────────────────────

def upsert_article(conn: sqlite3.Connection, article: Dict[str, Any]) -> Optional[int]:
    """
    Insert *article* into the articles table, ignoring duplicates.

    Returns the rowid of the inserted/existing row, or None on error.

    The INSERT OR IGNORE means a re-run with the same content_hash is a no-op.
    """
    sql = """
        INSERT OR IGNORE INTO articles
            (content_hash, url, title, description, source_name, source_id,
             published_at, author, raw_json)
        VALUES
            (:content_hash, :url, :title, :description, :source_name, :source_id,
             :published_at, :author, :raw_json)
    """
    try:
        cur = conn.execute(sql, article)
        if cur.lastrowid:
            return cur.lastrowid
        # Row already existed — fetch its id
        row = conn.execute(
            "SELECT id FROM articles WHERE content_hash = ?",
            (article["content_hash"],),
        ).fetchone()
        return row["id"] if row else None
    except sqlite3.Error as exc:
        logger.error("Failed to upsert article %s: %s", article.get("url"), exc)
        return None


def fetch_unscored_articles(
    conn: sqlite3.Connection, model_name: str, limit: int = 500
) -> List[sqlite3.Row]:
    """
    Return articles that have no sentiment result for *model_name*.

    Joining on sentiment_results so we never re-score work already done.
    """
    sql = """
        SELECT a.*
        FROM   articles a
        LEFT JOIN sentiment_results sr
               ON sr.article_id = a.id AND sr.model_name = :model
        WHERE  sr.id IS NULL
        ORDER  BY a.published_at DESC
        LIMIT  :limit
    """
    return conn.execute(sql, {"model": model_name, "limit": limit}).fetchall()


# ── Sentiment helpers ─────────────────────────────────────────────────────────

def upsert_sentiment(conn: sqlite3.Connection, result: Dict[str, Any]) -> None:
    """
    Insert or replace a sentiment result.

    Uses INSERT OR REPLACE so that re-scoring with the same model overwrites
    the old value (e.g., after a model upgrade).
    """
    sql = """
        INSERT OR REPLACE INTO sentiment_results
            (article_id, label, score, model_name)
        VALUES
            (:article_id, :label, :score, :model_name)
    """
    try:
        conn.execute(sql, result)
    except sqlite3.Error as exc:
        logger.error("Failed to upsert sentiment for article %s: %s",
                     result.get("article_id"), exc)


# ── Market cache helpers ──────────────────────────────────────────────────────

def upsert_market_row(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    """Upsert one day of OHLCV data for a symbol."""
    sql = """
        INSERT OR REPLACE INTO market_cache
            (symbol, date, open, high, low, close, volume)
        VALUES
            (:symbol, :date, :open, :high, :low, :close, :volume)
    """
    conn.execute(sql, row)


def fetch_market_data(
    conn: sqlite3.Connection,
    symbol: str,
    start_date: str,
    end_date: str,
) -> List[sqlite3.Row]:
    """Return cached OHLCV rows for *symbol* between *start_date* and *end_date*."""
    sql = """
        SELECT * FROM market_cache
        WHERE  symbol = ? AND date BETWEEN ? AND ?
        ORDER  BY date
    """
    return conn.execute(sql, (symbol, start_date, end_date)).fetchall()


# ── Refresh run helpers ───────────────────────────────────────────────────────

def start_refresh_run(conn: sqlite3.Connection, started_at: str) -> int:
    """Insert a new refresh_run row and return its id."""
    cur = conn.execute(
        "INSERT INTO refresh_runs (started_at, status) VALUES (?, 'running')",
        (started_at,),
    )
    return cur.lastrowid  # type: ignore[return-value]


def finish_refresh_run(
    conn: sqlite3.Connection,
    run_id: int,
    finished_at: str,
    status: str,
    articles_ingested: int,
    articles_scored: int,
    error_message: Optional[str] = None,
) -> None:
    """Update the refresh_run row when the pipeline finishes."""
    conn.execute(
        """
        UPDATE refresh_runs
        SET    finished_at        = ?,
               status             = ?,
               articles_ingested  = ?,
               articles_scored    = ?,
               error_message      = ?
        WHERE  id = ?
        """,
        (finished_at, status, articles_ingested, articles_scored, error_message, run_id),
    )


# ── Analytics query helpers ───────────────────────────────────────────────────

def fetch_sentiment_with_articles(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[sqlite3.Row]:
    """
    Join articles + sentiment_results for dashboard consumption.

    *start_date* / *end_date* are ISO-8601 date strings (YYYY-MM-DD).
    """
    conditions: List[str] = []
    params: List[Any] = []

    if start_date:
        conditions.append("a.published_at >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.published_at <= ?")
        params.append(end_date + "T23:59:59Z")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            a.id,
            a.title,
            a.url,
            a.source_name,
            a.published_at,
            a.description,
            sr.label,
            sr.score,
            sr.model_name
        FROM   articles a
        JOIN   sentiment_results sr ON sr.article_id = a.id
        {where}
        ORDER  BY a.published_at DESC
    """
    return conn.execute(sql, params).fetchall()


# ── Migration 002 helpers ─────────────────────────────────────────────────────

# ── Article classifications ───────────────────────────────────────────────────

def upsert_classification(conn: sqlite3.Connection, data: Dict[str, Any]) -> None:
    """
    Insert or replace an article classification row.

    Expected keys: article_id, topic, asset_class, region, importance,
                   source_type, ticker_mentions (JSON str), classified_by
    """
    sql = """
        INSERT OR REPLACE INTO article_classifications
            (article_id, topic, sector, asset_class, confidence, classifier_version)
        VALUES (:article_id, :topic, :asset_class, :region, :confidence, :classified_by)
    """
    # Remap to migration 002 column names
    row = {
        "article_id": data["article_id"],
        "topic": data.get("topic", "OTHER"),
        "asset_class": data.get("asset_class", "UNKNOWN"),
        "region": data.get("region", "UNKNOWN"),
        "confidence": data.get("confidence", 1.0),
        "classified_by": data.get("classified_by", "rules"),
    }
    try:
        conn.execute(sql, row)
    except sqlite3.Error as exc:
        logger.error("Failed to upsert classification for article %s: %s",
                     data.get("article_id"), exc)


def fetch_classified_articles(
    conn: sqlite3.Connection,
    topics: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    importance_levels: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000,
) -> List[sqlite3.Row]:
    """Fetch articles joined with their classifications, filtered by metadata."""
    conditions: List[str] = []
    params: List[Any] = []

    if start_date:
        conditions.append("a.published_at >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.published_at <= ?")
        params.append(end_date + "T23:59:59Z")
    if topics:
        placeholders = ",".join("?" * len(topics))
        conditions.append(f"ac.topic IN ({placeholders})")
        params.extend(topics)
    if regions:
        placeholders = ",".join("?" * len(regions))
        conditions.append(f"ac.sector IN ({placeholders})")
        params.extend(regions)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    sql = f"""
        SELECT a.*, ac.topic, ac.sector AS region, ac.asset_class,
               ac.confidence AS classification_confidence
        FROM   articles a
        LEFT JOIN article_classifications ac ON ac.article_id = a.id
        {where}
        ORDER  BY a.published_at DESC
        LIMIT  ?
    """
    return conn.execute(sql, params).fetchall()


# ── Ensemble sentiment ────────────────────────────────────────────────────────

def upsert_ensemble_sentiment(conn: sqlite3.Connection, data: Dict[str, Any]) -> Optional[int]:
    """Insert or replace an ensemble sentiment record."""
    sql = """
        INSERT OR REPLACE INTO ensemble_sentiment
            (article_id, finbert_label, finbert_score, llm_label, llm_score,
             keyword_label, keyword_score, ensemble_label, ensemble_score,
             effective_weights)
        VALUES
            (:article_id, :finbert_label, :finbert_score, :llm_label, :llm_score,
             :keyword_label, :keyword_score, :ensemble_label, :ensemble_score,
             :effective_weights)
    """
    try:
        cur = conn.execute(sql, data)
        return cur.lastrowid
    except sqlite3.Error as exc:
        logger.error("Failed to upsert ensemble sentiment: %s", exc)
        return None


def fetch_ensemble_with_articles(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[sqlite3.Row]:
    """Join articles + ensemble_sentiment for dashboard consumption."""
    conditions: List[str] = []
    params: List[Any] = []

    if start_date:
        conditions.append("a.published_at >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.published_at <= ?")
        params.append(end_date + "T23:59:59Z")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            a.id, a.title, a.url, a.source_name, a.published_at, a.description,
            es.ensemble_label AS label, es.ensemble_score AS score,
            es.finbert_label, es.finbert_score,
            es.llm_label, es.llm_score,
            es.keyword_label, es.keyword_score,
            ac.topic, ac.sector AS region, ac.asset_class
        FROM   articles a
        JOIN   ensemble_sentiment es ON es.article_id = a.id
        LEFT JOIN article_classifications ac ON ac.article_id = a.id
        {where}
        ORDER  BY a.published_at DESC
    """
    return conn.execute(sql, params).fetchall()


# ── LLM analyses ──────────────────────────────────────────────────────────────

def upsert_llm_analysis(conn: sqlite3.Connection, data: Dict[str, Any]) -> Optional[int]:
    """Insert or replace an LLM analysis (cache by scope_hash + type + provider + model)."""
    sql = """
        INSERT OR REPLACE INTO llm_analyses
            (scope_hash, analysis_type, provider, model,
             sentiment_label, sentiment_score, confidence,
             key_signals, raw_text, latency_ms)
        VALUES
            (:scope_hash, :analysis_type, :provider, :model,
             :sentiment_label, :sentiment_score, :confidence,
             :key_signals, :raw_text, :latency_ms)
    """
    try:
        cur = conn.execute(sql, data)
        return cur.lastrowid
    except sqlite3.Error as exc:
        logger.error("Failed to upsert LLM analysis: %s", exc)
        return None


def fetch_cached_llm_analysis(
    conn: sqlite3.Connection,
    scope_hash: str,
    analysis_type: str,
    provider: str,
    model: str,
) -> Optional[sqlite3.Row]:
    """Return a cached LLM analysis if it exists."""
    return conn.execute(
        """SELECT * FROM llm_analyses
           WHERE scope_hash=? AND analysis_type=? AND provider=? AND model=?
           LIMIT 1""",
        (scope_hash, analysis_type, provider, model),
    ).fetchone()


# ── Confidence scores ─────────────────────────────────────────────────────────

def insert_confidence_score(conn: sqlite3.Connection, data: Dict[str, Any]) -> Optional[int]:
    """Insert a confidence score record."""
    sql = """
        INSERT INTO confidence_scores
            (ensemble_sentiment_id, retrieval_similarity, reranking_score,
             article_count, source_diversity, source_quality, sentiment_agreement,
             llm_confidence_label, final_score, confidence_label,
             component_scores_json, weights_json)
        VALUES
            (:ensemble_sentiment_id, :retrieval_similarity, :reranking_score,
             :article_count, :source_diversity, :source_quality, :sentiment_agreement,
             :llm_confidence_label, :final_score, :confidence_label,
             :component_scores_json, :weights_json)
    """
    try:
        cur = conn.execute(sql, data)
        return cur.lastrowid
    except sqlite3.Error as exc:
        logger.error("Failed to insert confidence score: %s", exc)
        return None


# ── Topic clusters ────────────────────────────────────────────────────────────

def insert_topic_cluster(conn: sqlite3.Connection, data: Dict[str, Any]) -> Optional[int]:
    """Insert a topic cluster record."""
    sql = """
        INSERT INTO topic_clusters
            (run_id, cluster_id, cluster_label, topic_enum,
             dominant_sentiment, avg_sentiment_score, size)
        VALUES
            (:run_id, :cluster_id, :cluster_label, :topic_enum,
             :dominant_sentiment, :avg_sentiment_score, :size)
    """
    try:
        cur = conn.execute(sql, data)
        return cur.lastrowid
    except sqlite3.Error as exc:
        logger.error("Failed to insert topic cluster: %s", exc)
        return None


def insert_cluster_memberships(
    conn: sqlite3.Connection,
    memberships: List[Dict[str, Any]],
) -> None:
    """Batch insert article→cluster memberships."""
    sql = """
        INSERT OR IGNORE INTO article_cluster_memberships
            (article_id, cluster_id, run_id)
        VALUES (:article_id, :cluster_id, :run_id)
    """
    try:
        conn.executemany(sql, memberships)
    except sqlite3.Error as exc:
        logger.error("Failed to insert cluster memberships: %s", exc)


def fetch_clusters_with_articles(
    conn: sqlite3.Connection, run_id: Optional[int] = None
) -> List[sqlite3.Row]:
    """Fetch topic clusters and their article members for the latest (or given) run."""
    if run_id is None:
        row = conn.execute(
            "SELECT MAX(run_id) as rid FROM topic_clusters"
        ).fetchone()
        run_id = row["rid"] if row and row["rid"] else 0

    sql = """
        SELECT tc.*, COUNT(acm.article_id) AS member_count
        FROM   topic_clusters tc
        LEFT JOIN article_cluster_memberships acm
               ON acm.cluster_id = tc.id AND acm.run_id = tc.run_id
        WHERE  tc.run_id = ?
        GROUP  BY tc.id
        ORDER  BY tc.avg_sentiment_score DESC
    """
    return conn.execute(sql, (run_id,)).fetchall()


# ── Pipeline metrics ──────────────────────────────────────────────────────────

def insert_pipeline_metrics(
    conn: sqlite3.Connection, metrics: List[Dict[str, Any]]
) -> None:
    """Batch insert pipeline metric rows."""
    sql = """
        INSERT INTO pipeline_metrics (run_id, stage, metric_name, metric_value)
        VALUES (:run_id, :stage, :metric_name, :metric_value)
    """
    try:
        conn.executemany(sql, metrics)
    except sqlite3.Error as exc:
        logger.error("Failed to insert pipeline metrics: %s", exc)


def fetch_pipeline_metrics(
    conn: sqlite3.Connection, run_id: int
) -> List[sqlite3.Row]:
    """Fetch all metrics for a given pipeline run."""
    return conn.execute(
        "SELECT * FROM pipeline_metrics WHERE run_id=? ORDER BY recorded_at",
        (run_id,),
    ).fetchall()


def fetch_latest_run_id(conn: sqlite3.Connection) -> Optional[int]:
    """Return the most recent refresh_run id."""
    row = conn.execute(
        "SELECT MAX(id) AS rid FROM refresh_runs WHERE status='success'"
    ).fetchone()
    return row["rid"] if row else None
