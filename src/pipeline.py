"""
pipeline.py — upgraded orchestrator for the full RAG + confidence pipeline.

Stages (in order):
  1.  Ingest          — NewsAPI + RSS
  2.  Normalise       — ArticleRecord, UTC timestamps, content hash
  3.  Classify        — rule-based topic/asset/region/importance
  4.  Deduplicate     — in-memory hash + URL canonical
  5.  Store articles  — INSERT OR IGNORE (idempotent)
  6.  Embed           — sentence-transformers → ChromaDB upsert
  7.  FinBERT score   — unscored articles only
  8.  Keyword score   — lightweight keyword sentiment
  9.  Ensemble        — FinBERT + LLM + keyword → ensemble_sentiment
  10. Confidence      — 7-input confidence engine
  11. LLM analysis    — market overview via configured LLM provider
  12. Topic cluster   — K-means on embeddings
  13. Market refresh  — yfinance OHLCV cache
  14. Metrics flush   — write pipeline_metrics to DB

Every run is idempotent: duplicate articles are skipped, duplicate embeddings
are upserted, and LLM results are cached by scope_hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config.settings import settings

logging.basicConfig(
    level=settings.log_level_int,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# ── Imports (lazy where needed to avoid start-up cost) ───────────────────────
from src.classification.classifier import RuleBasedClassifier
from src.confidence.engine import ConfidenceEngine, ConfidenceInput
from src.deduplication.deduplicator import deduplicate
from src.ingestion.market_client import fetch_market_data
from src.ingestion.newsapi_client import NewsAPIIngestionClient
from src.ingestion.rss_client import RSSIngestionClient
from src.normalization.normalizer import normalise_many
from src.observability.metrics import MetricsCollector
from src.sentiment.ensemble import EnsembleScorer
from src.sentiment.keyword import KeywordSentimentScorer
from src.sentiment.scorer import SentimentScorer
from src.storage.database import (
    fetch_unscored_articles,
    finish_refresh_run,
    get_connection,
    init_db,
    insert_cluster_memberships,
    insert_confidence_score,
    insert_pipeline_metrics,
    insert_topic_cluster,
    start_refresh_run,
    upsert_article,
    upsert_classification,
    upsert_ensemble_sentiment,
    upsert_llm_analysis,
    upsert_market_row,
    upsert_sentiment,
)


def run_pipeline(
    skip_market_refresh: bool = False,
    skip_llm: bool = False,
    skip_clustering: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Execute the full pipeline and return a summary dict.

    Parameters
    ----------
    skip_market_refresh : bool  Skip yfinance download.
    skip_llm            : bool  Skip LLM analysis (faster, no API cost).
    skip_clustering     : bool  Skip topic clustering (slower first run).
    dry_run             : bool  No DB writes — for smoke testing.

    Returns
    -------
    dict: articles_ingested, articles_scored, articles_embedded,
          clusters_created, status, error
    """
    db_path = settings.db_path_resolved
    init_db(db_path)

    metrics = MetricsCollector()
    started_at = _now_utc()
    run_id: Optional[int] = None
    articles_ingested = articles_scored = articles_embedded = clusters_created = 0
    status = "success"
    error_message: Optional[str] = None

    try:
        if not dry_run:
            with get_connection(db_path) as conn:
                run_id = start_refresh_run(conn, started_at)
            logger.info("Pipeline started — run_id=%s", run_id)

        # ── Stage 1: Ingest ───────────────────────────────────────────────────
        with metrics.timed_stage("ingestion"):
            logger.info("Stage 1: Ingesting …")
            raw_articles = []

            if settings.news_api_key:
                try:
                    newsapi = NewsAPIIngestionClient(
                        api_key=settings.news_api_key,
                        page_size=settings.news_api_page_size,
                        topics=settings.news_api_topics,
                        lookback_days=settings.ingest_lookback_days,
                    )
                    raw_articles.extend(newsapi.fetch())
                except Exception as exc:
                    logger.warning("NewsAPI failed: %s", exc)
            else:
                logger.warning("NEWS_API_KEY not set — skipping NewsAPI")

            if settings.rss_feeds:
                try:
                    rss = RSSIngestionClient(feed_urls=settings.rss_feeds)
                    raw_articles.extend(rss.fetch())
                except Exception as exc:
                    logger.warning("RSS failed: %s", exc)

            metrics.record("ingestion", "raw_articles", len(raw_articles))
            logger.info("Stage 1 done: %d raw articles", len(raw_articles))

        # ── Stage 2: Normalise ────────────────────────────────────────────────
        with metrics.timed_stage("normalisation"):
            normalised = normalise_many(raw_articles)
            metrics.record("normalisation", "normalised", len(normalised))

        # ── Stage 3: Classify ─────────────────────────────────────────────────
        with metrics.timed_stage("classification"):
            classifier = RuleBasedClassifier()
            classifications = {}  # content_hash → ArticleClassification
            for record in normalised:
                cls = classifier.classify(
                    title=record.title,
                    description=record.description or "",
                    source_name=record.source_name,
                    url=record.url,
                )
                classifications[record.content_hash] = cls
            metrics.record("classification", "classified", len(classifications))
            logger.info("Stage 3 done: %d articles classified", len(classifications))

        # ── Stage 4: Deduplicate ──────────────────────────────────────────────
        with metrics.timed_stage("deduplication"):
            existing_hashes: set = set()
            if not dry_run:
                with get_connection(db_path) as conn:
                    rows = conn.execute("SELECT content_hash FROM articles").fetchall()
                    existing_hashes = {r["content_hash"] for r in rows}
            unique, dropped = deduplicate(normalised, existing_hashes=existing_hashes)
            metrics.record("deduplication", "unique", len(unique))
            metrics.record("deduplication", "dropped", dropped)

        # ── Stage 5: Store articles + classifications ─────────────────────────
        with metrics.timed_stage("storage_articles"):
            if not dry_run:
                with get_connection(db_path) as conn:
                    for record in unique:
                        article_id = upsert_article(conn, record.to_db_dict())
                        if article_id:
                            cls = classifications.get(record.content_hash)
                            if cls:
                                upsert_classification(conn, {
                                    "article_id": article_id,
                                    "topic": cls.topic.value,
                                    "asset_class": cls.asset_class.value,
                                    "region": cls.region.value,
                                    "confidence": 1.0,
                                    "classified_by": cls.classified_by,
                                })
            articles_ingested = len(unique)
            metrics.record("storage_articles", "stored", articles_ingested)

        # ── Stage 6: Embed articles ───────────────────────────────────────────
        with metrics.timed_stage("embedding"):
            try:
                from src.retrieval.embedder import TextEmbedder
                from src.retrieval.vector_store import VectorStore

                embedder = TextEmbedder(model_name=settings.embedding_model)
                vector_store = VectorStore(
                    persist_directory=str(settings.chroma_persist_dir_resolved),
                    collection_name=settings.chroma_collection,
                )

                # Fetch articles that aren't yet embedded
                if not dry_run:
                    with get_connection(db_path) as conn:
                        rows = conn.execute("""
                            SELECT a.id, a.title, a.description, a.source_name,
                                   a.published_at, ac.topic, ac.sector, ac.asset_class
                            FROM   articles a
                            LEFT JOIN article_embeddings ae ON ae.article_id = a.id
                            LEFT JOIN article_classifications ac ON ac.article_id = a.id
                            WHERE  ae.article_id IS NULL
                            ORDER  BY a.published_at DESC
                            LIMIT  500
                        """).fetchall()

                    if rows:
                        texts = [
                            f"{r['title']} {(r['description'] or '')[:300]}".strip()
                            for r in rows
                                                ]
                        article_ids = [r["id"] for r in rows]
                        logger.info("STEP A: Starting embedding generation")
                        embeddings = embedder.embed_batch(texts)
                        logger.info("STEP B: Embeddings generated (%d)", len(embeddings))
                        if not dry_run:
                            vector_store.add_articles(
                                ids=[str(aid) for aid in article_ids],
                                embeddings=embeddings,
                                texts=texts,
                                metadatas=[
                                    {
                                        "article_id": r["id"],
                                        "source_name": r["source_name"] or "",
                                        "topic": r["topic"] or "",
                                        "asset_class": r["asset_class"] or "",
                                        "published_at": r["published_at"] or "",
                                    }
                                    for r in rows
                                ],
                            )
                            logger.info("STEP D: Chroma write finished")
                        articles_embedded = len(rows)
                        metrics.record("embedding", "embedded", articles_embedded)
                        logger.info("Stage 6 done: %d articles embedded", articles_embedded)
            except Exception as exc:
                logger.warning("Embedding stage failed (non-fatal): %s", exc)

        # ── Stage 7: FinBERT scoring ──────────────────────────────────────────
        with metrics.timed_stage("finbert"):
            try:
                scorer = SentimentScorer(model_name=settings.sentiment_model)
                unscored = []
                if not dry_run:
                    with get_connection(db_path) as conn:
                        unscored = fetch_unscored_articles(conn, settings.sentiment_model,limit=50)
                if unscored:
                    for article in unscored:
                        description = article["description"] if article["description"] else ""
                        text = f"{article['title']} {description[:300]}".strip()
                        results = scorer.score_batch([text])
                        result = results[0]
                        if not dry_run:
                            with get_connection(db_path) as conn:
                                upsert_sentiment(conn, {
                                    "article_id": article["id"],
                                    "label": result.label,
                                    "score": result.score,
                                    "model_name": settings.sentiment_model,
                                })
                    articles_scored = len(unscored)
                    metrics.record("finbert", "scored", articles_scored)
                    logger.info("Stage 7 done: %d articles FinBERT-scored", articles_scored)
            except Exception as exc:
                logger.warning("FinBERT stage failed (non-fatal): %s", exc)

        # ── Stage 8 + 9 + 10: Keyword, Ensemble, Confidence ──────────────────
        with metrics.timed_stage("ensemble"):
            try:
                keyword_scorer = KeywordSentimentScorer()
                ensemble_scorer = EnsembleScorer()
                confidence_engine = ConfidenceEngine()

                if not dry_run:
                    with get_connection(db_path) as conn:
                        # Articles with FinBERT scores but no ensemble yet
                        rows = conn.execute("""
                            SELECT a.id, a.title, a.description,
                                   s.label AS finbert_label,
                                   s.positive_score, s.negative_score, s.neutral_score,
                                   ac.topic, ac.importance, ac.source_type
                            FROM   articles a
                            JOIN   sentiment_scores s      ON s.article_id = a.id
                            LEFT JOIN article_classifications ac ON ac.article_id = a.id
                            LEFT JOIN ensemble_sentiments es ON es.article_id = a.id
                            WHERE  es.article_id IS NULL
                            LIMIT  500
                        """).fetchall()

                    for row in rows:
                        title = row["title"] or ""
                        description = row["description"] if row["description"] else ""
                        text = f"{title} {description[:300]}".strip()
                        kw_label, kw_score = keyword_scorer.score(text)
                        finbert_conf = max(
                            row["positive_score"],
                            row["negative_score"],
                            row["neutral_score"],
                        )
                        ens = ensemble_scorer.score(
                            finbert_result=(row["finbert_label"], finbert_conf),
                            llm_result=None,
                            keyword_result=(kw_label, abs(kw_score)),
                        )
                        conf_in = ConfidenceInput(
                            finbert_confidence=finbert_conf,
                            ensemble_agreement=1.0 if ens.ensemble_label == row["finbert_label"] else 0.0,
                            source_quality=0.7,
                            article_length=len(text),
                            has_llm_confirmation=False,
                            llm_confidence_label="LOW",
                            keyword_signal_strength=abs(kw_score),
                        )
                        conf = confidence_engine.compute(conf_in)
                        if not dry_run:
                            with get_connection(db_path) as conn:
                                upsert_ensemble_sentiment(conn, {
                                    "article_id": row["id"],
                                    "ensemble_label": ens.ensemble_label,
                                    "ensemble_score": ens.ensemble_score,
                                    "finbert_weight": ens.effective_weights.get("finbert", 0.0),
                                    "llm_weight": ens.effective_weights.get("llm", 0.0),
                                    "keyword_weight": ens.effective_weights.get("keyword", 0.0),
                                    "scored_at": _now_utc(),
                                })
                                insert_confidence_score(conn, {
                                    "article_id": row["id"],
                                    "confidence_score": conf.confidence_score,
                                    "confidence_label": conf.confidence_label,
                                    "component_scores": json.dumps(conf.component_scores),
                                    "scored_at": _now_utc(),
                                })
                    metrics.record("ensemble", "processed", len(rows) if not dry_run else 0)
                    logger.info("Stage 8-10 done: ensemble + confidence for %d articles",
                                len(rows) if not dry_run else 0)
            except Exception as exc:
                logger.warning("Ensemble/confidence stage failed (non-fatal): %s", exc)

        # ── Stage 11: LLM market commentary ──────────────────────────────────
        if not skip_llm:
            with metrics.timed_stage("llm"):
                try:
                    from src.llm.factory import create_llm_provider

                    provider_name = settings.llm_provider
                    llm_kwargs: dict = {}

                    if provider_name == "gemini":
                        llm_kwargs = {
                            "api_key": settings.google_api_key or None,
                            "model": settings.gemini_model,
                        }
                    elif provider_name == "anthropic":
                        llm_kwargs = {
                            "api_key": settings.anthropic_api_key or None,
                            "model": settings.anthropic_model,
                        }
                    elif provider_name == "openai":
                        llm_kwargs = {
                            "api_key": settings.openai_api_key or None,
                            "model": settings.openai_model,
                        }
                    elif provider_name == "ollama":
                        llm_kwargs = {
                            "base_url": settings.ollama_base_url,
                            "model": settings.ollama_model,
                        }

                    llm = create_llm_provider(provider_name, **llm_kwargs)

                    if not dry_run:
                        with get_connection(db_path) as conn:
                            recent = conn.execute("""
                                SELECT a.id, a.title, a.description, a.source_name,
                                       es.ensemble_label, es.ensemble_score
                                FROM   articles a
                                JOIN   ensemble_sentiments es ON es.article_id = a.id
                                ORDER  BY a.published_at DESC
                                LIMIT  20
                            """).fetchall()

                        if recent:
                            headlines = "\n".join(
                                f"- [{r['ensemble_label']}] {r['title']}"
                                for r in recent
                            )
                            scope_hash = hashlib.md5(headlines.encode()).hexdigest()

                            # Check cache
                            with get_connection(db_path) as conn:
                                cached = conn.execute(
                                    "SELECT id FROM llm_analyses WHERE scope_hash = ?",
                                    (scope_hash,),
                                ).fetchone()

                            if not cached:
                                commentary = llm.generate_market_commentary(
                                    recent_headlines=[r["title"] for r in recent],
                                    sentiment_summary={
                                        "positive": sum(1 for r in recent if r["ensemble_label"] == "positive"),
                                        "negative": sum(1 for r in recent if r["ensemble_label"] == "negative"),
                                        "neutral": sum(1 for r in recent if r["ensemble_label"] == "neutral"),
                                    },
                                )
                                with get_connection(db_path) as conn:
                                    upsert_llm_analysis(conn, {
                                        "scope_hash": scope_hash,
                                        "provider": llm.provider_name(),
                                        "model": llm.model_name(),
                                        "analysis_type": "market_commentary",
                                        "content": commentary,
                                        "generated_at": _now_utc(),
                                    })
                                logger.info("Stage 11 done: LLM market commentary generated (%s)",
                                            llm.provider_name())
                            else:
                                logger.info("Stage 11: LLM commentary cached (scope_hash=%s)", scope_hash)
                except Exception as exc:
                    logger.warning("LLM stage failed (non-fatal): %s", exc)
        else:
            logger.info("Stage 11: LLM skipped (--skip-llm)")

        # ── Stage 12: Topic clustering ────────────────────────────────────────
        if not skip_clustering:
            with metrics.timed_stage("clustering"):
                try:
                    from src.clustering.topic_clusterer import TopicClusterer

                    if not dry_run:
                        with get_connection(db_path) as conn:
                            emb_rows = conn.execute("""
                                SELECT ae.article_id, ae.embedding_json
                                FROM   article_embeddings ae
                                ORDER  BY ae.article_id DESC
                                LIMIT  1000
                            """).fetchall()

                        if len(emb_rows) >= settings.min_cluster_size:
                            import numpy as np
                            ids = [r["article_id"] for r in emb_rows]
                            embeddings_matrix = np.array(
                                [json.loads(r["embedding_json"]) for r in emb_rows]
                            )
                            clusterer = TopicClusterer(n_clusters=settings.cluster_count)
                            labels = clusterer.fit_predict(embeddings_matrix)
                            if not dry_run:
                                with get_connection(db_path) as conn:
                                    cluster_id = insert_topic_cluster(conn, {
                                        "created_at": _now_utc(),
                                        "n_clusters": settings.cluster_count,
                                        "algorithm": "kmeans",
                                    })
                                    insert_cluster_memberships(conn, [
                                        {"cluster_run_id": cluster_id,
                                         "article_id": aid,
                                         "cluster_label": int(lbl)}
                                        for aid, lbl in zip(ids, labels)
                                    ])
                            clusters_created = settings.cluster_count
                            metrics.record("clustering", "clusters", clusters_created)
                            logger.info("Stage 12 done: %d clusters created", clusters_created)
                except Exception as exc:
                    logger.warning("Clustering stage failed (non-fatal): %s", exc)
        else:
            logger.info("Stage 12: Clustering skipped (--skip-clustering)")

        # ── Stage 13: Market data refresh ─────────────────────────────────────
        if not skip_market_refresh:
            with metrics.timed_stage("market_refresh"):
                try:
                    for symbol in settings.market_symbols:
                        rows_fetched = fetch_market_data(symbol)
                        if not dry_run and rows_fetched:
                            with get_connection(db_path) as conn:
                                for row in rows_fetched:
                                    upsert_market_row(conn, row)
                    logger.info("Stage 13 done: market data refreshed for %s",
                                settings.market_symbols)
                except Exception as exc:
                    logger.warning("Market refresh stage failed (non-fatal): %s", exc)
        else:
            logger.info("Stage 13: Market refresh skipped")

    except Exception as exc:
        status = "error"
        error_message = str(exc)
        logger.exception("Pipeline failed: %s", exc)

    finally:
        # ── Stage 14: Flush metrics ───────────────────────────────────────────
        if not dry_run and run_id is not None:
            try:
                with get_connection(db_path) as conn:
                    finish_refresh_run(conn, run_id, status, error_message,articles_ingested, articles_scored)
                    insert_pipeline_metrics(conn, {
                        "run_id": run_id,
                        "stage_timings": json.dumps(metrics.stage_timings),
                        "stage_counts": json.dumps(metrics.stage_counts),
                        "created_at": _now_utc(),
                    })
            except Exception as exc:
                logger.warning("Metrics flush failed: %s", exc)

    summary = {
        "articles_ingested": articles_ingested,
        "articles_scored": articles_scored,
        "articles_embedded": articles_embedded,
        "clusters_created": clusters_created,
        "status": status,
        "error": error_message,
    }
    logger.info("Pipeline complete: %s", summary)
    return summary


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
