"""
Reranker module for financial news retrieval.

Combines semantic similarity scores with freshness, importance, and source quality
signals to produce a composite ranking of candidate articles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RankedDocument:
    """A retrieved article annotated with all ranking signals and a composite score."""

    article_id: int
    title: str
    description: str
    source_name: str
    published_at: str
    topic: str
    importance: str
    semantic_score: float
    freshness_score: float
    importance_score: float
    quality_score: float
    composite_score: float


# ---------------------------------------------------------------------------
# Source quality lookup
# ---------------------------------------------------------------------------

SOURCE_QUALITY: Dict[str, float] = {
    "Reuters": 0.90,
    "Bloomberg": 0.90,
    "WSJ": 0.88,
    "The Wall Street Journal": 0.88,
    "FT": 0.87,
    "Financial Times": 0.87,
    "CNBC": 0.75,
    "MarketWatch": 0.70,
    "Yahoo Finance": 0.65,
    "default": 0.55,
}

_IMPORTANCE_TO_SCORE: Dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.3,
}

_HALF_LIFE_DAYS: float = 3.0  # freshness half-life in days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_freshness(published_at_iso: str) -> float:
    """Compute a freshness score in [0, 1] using exponential decay.

    The score is 1.0 for a brand-new article and decays toward 0 with a
    half-life of 3 days.  Articles with unparseable timestamps receive a
    score of 0.0.

    Args:
        published_at_iso: ISO-8601 timestamp string, e.g. ``"2024-01-15T09:30:00Z"``.

    Returns:
        A float in the range [0.0, 1.0].
    """
    if not published_at_iso:
        return 0.0

    try:
        ts = published_at_iso.strip()
        # Replace trailing Z with +00:00 for fromisoformat compatibility
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"

        try:
            published_dt = datetime.fromisoformat(ts)
        except ValueError:
            # Fallback: truncate to seconds precision and try again
            published_dt = datetime.fromisoformat(published_at_iso[:19])

        # Ensure offset-aware before subtraction
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_seconds = (now - published_dt).total_seconds()
        age_days = max(age_seconds / 86400.0, 0.0)

        # Exponential decay: score = 2^(-age / half_life)
        score = math.pow(2.0, -age_days / _HALF_LIFE_DAYS)
        return float(max(0.0, min(1.0, score)))

    except Exception:  # noqa: BLE001
        return 0.0


def _get_source_quality(source_name: str) -> float:
    """Look up source quality score, falling back to the default.

    Args:
        source_name: The name of the news source as stored in the database.

    Returns:
        A quality score in [0.0, 1.0].
    """
    if not source_name:
        return SOURCE_QUALITY["default"]
    # Exact match first
    if source_name in SOURCE_QUALITY:
        return SOURCE_QUALITY[source_name]
    # Partial match (case-insensitive)
    source_lower = source_name.lower()
    for key, value in SOURCE_QUALITY.items():
        if key == "default":
            continue
        if key.lower() in source_lower or source_lower in key.lower():
            return value
    return SOURCE_QUALITY["default"]


def _get_importance_score(importance: str) -> float:
    """Convert an importance enum string to a numeric score.

    Args:
        importance: One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"`` (case-insensitive).

    Returns:
        A float score: HIGH=1.0, MEDIUM=0.6, LOW=0.3.
    """
    if not importance:
        return _IMPORTANCE_TO_SCORE["LOW"]
    return _IMPORTANCE_TO_SCORE.get(importance.upper(), _IMPORTANCE_TO_SCORE["LOW"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    search_results: List[Dict[str, Any]],
    article_rows: Dict[int, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[RankedDocument]:
    """Re-rank retrieved articles using a weighted combination of signals.

    The composite score is:

        composite = w_semantic * semantic_score
                  + w_freshness * freshness_score
                  + w_importance * importance_score
                  + w_quality * quality_score

    Args:
        search_results: List of dicts returned by the vector store.  Each
            dict must contain at least ``{"id": int, "score": float}``.
        article_rows: Mapping from ``article_id`` to a dict containing full
            article metadata (title, description, source_name, published_at,
            topic, importance).  Rows missing from this mapping are skipped.
        weights: Optional override for signal weights.  Expected keys:
            ``"semantic"``, ``"freshness"``, ``"importance"``, ``"quality"``.
            Defaults to ``{"semantic": 0.40, "freshness": 0.25,
            "importance": 0.20, "quality": 0.15}``.

    Returns:
        A list of :class:`RankedDocument` objects sorted descending by
        ``composite_score``.
    """
    if weights is None:
        weights = {
            "semantic": 0.40,
            "freshness": 0.25,
            "importance": 0.20,
            "quality": 0.15,
        }

    w_semantic = float(weights.get("semantic", 0.40))
    w_freshness = float(weights.get("freshness", 0.25))
    w_importance = float(weights.get("importance", 0.20))
    w_quality = float(weights.get("quality", 0.15))

    ranked: List[RankedDocument] = []

    for result in search_results:
        article_id = int(result.get("id", -1))
        if article_id not in article_rows:
            continue

        row = article_rows[article_id]
        semantic_score = float(result.get("score", 0.0))

        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        source_name = str(row.get("source_name", "") or "")
        published_at = str(row.get("published_at", "") or "")
        topic = str(row.get("topic", "") or "")
        importance = str(row.get("importance", "") or "LOW")

        freshness_score = compute_freshness(published_at)
        importance_score = _get_importance_score(importance)
        quality_score = _get_source_quality(source_name)

        composite_score = (
            w_semantic * semantic_score
            + w_freshness * freshness_score
            + w_importance * importance_score
            + w_quality * quality_score
        )

        ranked.append(
            RankedDocument(
                article_id=article_id,
                title=title,
                description=description,
                source_name=source_name,
                published_at=published_at,
                topic=topic,
                importance=importance,
                semantic_score=semantic_score,
                freshness_score=freshness_score,
                importance_score=importance_score,
                quality_score=quality_score,
                composite_score=composite_score,
            )
        )

    ranked.sort(key=lambda doc: doc.composite_score, reverse=True)
    return ranked
