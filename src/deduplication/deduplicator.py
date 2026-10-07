"""
deduplicator.py — in-memory deduplication before database writes.

The database enforces uniqueness via UNIQUE(content_hash), so any true
duplicate will be silently dropped on INSERT OR IGNORE.  This module adds a
pre-storage pass to:
  1. Avoid sending 1,000 rows to SQLite only to have 800 rejected.
  2. Provide an observable count of duplicates in pipeline logs.
  3. Allow cross-source deduplication (same story on NewsAPI + RSS feed).

Strategy:
  - Primary key:  content_hash = SHA-256(title.lower() + url.lower())
  - Secondary key: normalised URL (strip UTM params, trailing slashes)

Design note: The secondary URL key is intentionally lenient — it won't catch
every variant, but it catches the most common "same article, different query
string" pattern.  The DB unique constraint is the safety net.
"""

from __future__ import annotations

import logging
import re
from typing import List, Set, Tuple
from urllib.parse import urlparse, urlencode, parse_qs

from src.normalization.normalizer import ArticleRecord

logger = logging.getLogger(__name__)

# UTM and common tracking parameters to strip from URLs
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "ref", "referer", "referrer", "source", "mc_cid", "mc_eid",
    }
)


def deduplicate(
    articles: List[ArticleRecord],
    existing_hashes: Optional[Set[str]] = None,
) -> Tuple[List[ArticleRecord], int]:
    """
    Remove duplicates from *articles*.

    Parameters
    ----------
    articles : list[ArticleRecord]
        Normalised articles from this ingestion run.
    existing_hashes : set[str] | None
        Content hashes already present in the database.  When provided,
        articles matching existing DB records are also dropped.

    Returns
    -------
    unique : list[ArticleRecord]
        Deduplicated articles.
    dropped : int
        Number of articles removed as duplicates.
    """
    seen_hashes: Set[str] = set(existing_hashes or [])
    seen_urls: Set[str] = set()
    unique: List[ArticleRecord] = []

    for article in articles:
        canonical_url = _canonical_url(article.url)

        if article.content_hash in seen_hashes:
            logger.debug("Dedup: dropping hash duplicate — %s", article.url)
            continue
        if canonical_url in seen_urls:
            logger.debug("Dedup: dropping URL duplicate — %s", article.url)
            continue

        seen_hashes.add(article.content_hash)
        seen_urls.add(canonical_url)
        unique.append(article)

    dropped = len(articles) - len(unique)
    if dropped:
        logger.info("Dedup: dropped %d duplicates from %d articles", dropped, len(articles))

    return unique, dropped


def _canonical_url(url: str) -> str:
    """
    Normalise a URL for deduplication purposes:
    - Lowercase scheme + host
    - Remove tracking query parameters
    - Strip trailing slash from path
    """
    try:
        parsed = urlparse(url.lower())
        filtered_params = {
            k: v
            for k, v in parse_qs(parsed.query).items()
            if k not in _TRACKING_PARAMS
        }
        clean_path = parsed.path.rstrip("/")
        canonical = parsed._replace(
            path=clean_path,
            query=urlencode(filtered_params, doseq=True),
            fragment="",
        )
        return canonical.geturl()
    except Exception:
        # If URL parsing fails, fall back to raw URL
        return url.lower().rstrip("/")


# Allow Optional import
from typing import Optional
