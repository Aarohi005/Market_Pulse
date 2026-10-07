"""
Retrieval pipeline for financial news sentiment analysis.

Implements a three-stage retrieval process:
  1. Metadata filtering via SQLite
  2. Semantic search via ChromaDB
  3. Re-ranking using composite signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .reranker import RankedDocument, rerank

logger = logging.getLogger(__name__)

# ChromaDB's $in operator has a practical upper limit; above this threshold
# we skip the where-filter to avoid errors and rely on post-filter reranking.
_CHROMA_IN_LIMIT = 500


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Container for the output of a full retrieval pipeline run."""

    documents: List[RankedDocument]
    metadata_candidates: int
    semantic_candidates: int
    final_count: int
    query: str


# ---------------------------------------------------------------------------
# Type stubs for injected dependencies
# ---------------------------------------------------------------------------
# These are declared as protocols / duck-typed; the actual implementations
# live in their own modules.  Importing them here would create circular deps.

class MetadataFilter:
    """Value object carrying user-supplied filter criteria.

    Expected attributes (all optional):
        topic (str | None): e.g. "earnings", "macro"
        sources (List[str] | None): e.g. ["Reuters", "Bloomberg"]
        importance (str | None): "HIGH" | "MEDIUM" | "LOW"
        date_from (str | None): ISO-8601 date string lower bound
        date_to (str | None): ISO-8601 date string upper bound
    """


class VectorStore:
    """Duck-type contract for a ChromaDB-backed vector store.

    Expected methods:
        search(query_embedding, n_results, where) -> List[Dict[str, Any]]
            Returns dicts with keys: ``id`` (int), ``score`` (float).
    """


class TextEmbedder:
    """Duck-type contract for a text embedding model.

    Expected methods:
        embed(text: str) -> List[float]
    """


# ---------------------------------------------------------------------------
# Metadata filter helper
# ---------------------------------------------------------------------------

def apply_metadata_filter(
    db_path: str,
    metadata_filter: Any,
) -> List[int]:
    """Query SQLite for article IDs matching the given metadata filter.

    Builds a parameterised SQL query dynamically based on whichever filter
    fields are populated.

    Args:
        db_path: Filesystem path to the SQLite database file.
        metadata_filter: An object with optional attributes: ``topic``,
            ``sources`` (list), ``importance``, ``date_from``, ``date_to``.

    Returns:
        A list of integer article IDs that pass all active filter criteria.
        Returns an empty list if no rows match.
    """
    import sqlite3

    conditions: List[str] = []
    params: List[Any] = []

    topic = getattr(metadata_filter, "topic", None)
    sources = getattr(metadata_filter, "sources", None)
    importance = getattr(metadata_filter, "importance", None)
    date_from = getattr(metadata_filter, "date_from", None)
    date_to = getattr(metadata_filter, "date_to", None)

    if topic:
        conditions.append("topic = ?")
        params.append(topic)

    if sources:
        placeholders = ", ".join("?" * len(sources))
        conditions.append(f"source_name IN ({placeholders})")
        params.extend(sources)

    if importance:
        conditions.append("importance = ?")
        params.append(importance.upper())

    if date_from:
        conditions.append("published_at >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("published_at <= ?")
        params.append(date_to)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT id FROM articles {where_clause}"  # noqa: S608

    logger.debug("Metadata filter SQL: %s | params: %s", sql, params)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
        ids = [int(row[0]) for row in rows]
        logger.info("Metadata filter returned %d candidate article IDs.", len(ids))
        return ids
    except sqlite3.Error as exc:
        logger.error("SQLite error during metadata filter: %s", exc)
        return []


def _fetch_article_rows(
    db_path: str,
    article_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Fetch full article metadata for a list of IDs from SQLite.

    Args:
        db_path: Filesystem path to the SQLite database.
        article_ids: List of article IDs to fetch.

    Returns:
        Dict mapping article_id -> row dict with keys:
        title, description, source_name, published_at, topic, importance.
    """
    import sqlite3

    if not article_ids:
        return {}

    placeholders = ", ".join("?" * len(article_ids))
    sql = (
        f"SELECT id, title, description, source_name, published_at, topic, importance "  # noqa: S608
        f"FROM articles WHERE id IN ({placeholders})"
    )

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(sql, article_ids)
            rows = cursor.fetchall()
        return {
            int(row[0]): {
                "title": row[1],
                "description": row[2],
                "source_name": row[3],
                "published_at": row[4],
                "topic": row[5],
                "importance": row[6],
            }
            for row in rows
        }
    except sqlite3.Error as exc:
        logger.error("SQLite error fetching article rows: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """Three-stage retrieval pipeline for financial news articles.

    Stage 1 – Metadata filtering:
        Queries SQLite to narrow the corpus to articles that satisfy
        structured filter criteria (topic, source, importance, date range).

    Stage 2 – Semantic search:
        Embeds the user query and searches the ChromaDB vector store.
        If the candidate set from stage 1 is small enough (≤500 IDs),
        a ``$in`` where-filter is applied so ChromaDB only scores those
        candidates.  For larger sets the filter is omitted and ChromaDB
        searches the full index; stage-3 reranking then enforces the
        metadata constraints.

    Stage 3 – Reranking:
        Combines the ChromaDB cosine-similarity score with freshness,
        importance, and source-quality signals to produce a final ranked
        list capped at ``top_k`` documents.

    Args:
        db_path: Path to the SQLite database containing article metadata.
        vector_store: An object implementing the :class:`VectorStore` duck type.
        embedder: An object implementing the :class:`TextEmbedder` duck type.
    """

    def __init__(
        self,
        db_path: str,
        vector_store: Any,
        embedder: Any,
    ) -> None:
        self._db_path = db_path
        self._vector_store = vector_store
        self._embedder = embedder
        logger.debug(
            "RetrievalPipeline initialised | db=%s | vector_store=%s | embedder=%s",
            db_path,
            type(vector_store).__name__,
            type(embedder).__name__,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        filter: Any,  # MetadataFilter
        top_k: int = 8,
    ) -> RetrievalResult:
        """Run the full three-stage retrieval pipeline.

        Args:
            query: Natural-language query string from the user.
            filter: A :class:`MetadataFilter` (or compatible duck-type)
                specifying structured filter criteria.
            top_k: Maximum number of documents to return after reranking.

        Returns:
            A :class:`RetrievalResult` containing the ranked documents and
            diagnostic metadata about each pipeline stage.
        """
        logger.info("RetrievalPipeline.retrieve | query=%r | top_k=%d", query, top_k)

        # ------------------------------------------------------------------
        # Stage 1: Metadata filtering
        # ------------------------------------------------------------------
        logger.info("Stage 1: Applying metadata filter …")
        candidate_ids: List[int] = apply_metadata_filter(self._db_path, filter)
        metadata_candidates = len(candidate_ids)
        logger.info("Stage 1 complete: %d candidate IDs.", metadata_candidates)

        if metadata_candidates == 0:
            logger.warning("No articles passed the metadata filter; returning empty result.")
            return RetrievalResult(
                documents=[],
                metadata_candidates=0,
                semantic_candidates=0,
                final_count=0,
                query=query,
            )

        # ------------------------------------------------------------------
        # Stage 2: Semantic search via ChromaDB
        # ------------------------------------------------------------------
        logger.info("Stage 2: Embedding query and searching vector store …")
        query_embedding: List[float] = self._embedder.embed(query)

        # Build the ChromaDB where-filter only for small candidate sets.
        # ChromaDB's $in operator can be slow or fail for very large lists.
        chroma_where: Optional[Dict[str, Any]] = None
        if metadata_candidates <= _CHROMA_IN_LIMIT:
            chroma_where = {"id": {"$in": candidate_ids}}
            logger.debug(
                "Using ChromaDB $in filter with %d IDs.", metadata_candidates
            )
        else:
            logger.info(
                "Candidate set (%d) exceeds ChromaDB $in limit (%d); "
                "skipping vector-store filter and relying on post-filter reranking.",
                metadata_candidates,
                _CHROMA_IN_LIMIT,
            )

        # Request more results than top_k so the reranker has room to work.
        n_results = min(max(top_k * 5, 50), 500)
        raw_results: List[Dict[str, Any]] = self._vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=chroma_where,
        )
        semantic_candidates = len(raw_results)
        logger.info(
            "Stage 2 complete: %d results from vector store (requested %d).",
            semantic_candidates,
            n_results,
        )

        if semantic_candidates == 0:
            logger.warning("Vector store returned no results; returning empty result.")
            return RetrievalResult(
                documents=[],
                metadata_candidates=metadata_candidates,
                semantic_candidates=0,
                final_count=0,
                query=query,
            )

        # ------------------------------------------------------------------
        # Stage 3: Reranking
        # ------------------------------------------------------------------
        logger.info("Stage 3: Reranking %d candidates …", semantic_candidates)

        # For large candidate sets where ChromaDB was not filtered, we restrict
        # the reranker to only articles that passed the metadata filter.
        if chroma_where is None:
            candidate_id_set = set(candidate_ids)
            raw_results = [r for r in raw_results if int(r.get("id", -1)) in candidate_id_set]
            logger.debug(
                "Post-filter: %d results remain after metadata-ID intersection.",
                len(raw_results),
            )

        # Fetch full metadata for the semantic candidates
        result_ids = [int(r.get("id", -1)) for r in raw_results]
        article_rows = _fetch_article_rows(self._db_path, result_ids)

        ranked_docs = rerank(
            search_results=raw_results,
            article_rows=article_rows,
        )

        final_docs = ranked_docs[:top_k]
        logger.info(
            "Stage 3 complete: returning top %d / %d reranked documents.",
            len(final_docs),
            len(ranked_docs),
        )

        return RetrievalResult(
            documents=final_docs,
            metadata_candidates=metadata_candidates,
            semantic_candidates=semantic_candidates,
            final_count=len(final_docs),
            query=query,
        )
