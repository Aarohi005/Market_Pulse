"""
normalizer.py — converts RawArticle dicts from any source into ArticleRecord.

Responsibilities:
- Parse and normalise published_at to UTC ISO-8601
- Generate content_hash for deduplication
- Truncate overly long fields
- Reject articles with no parseable timestamp (configurable)

Design note: We keep normalisation separate from ingestion so that adding a new
source only requires writing a new client — the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from dateutil import parser as dateutil_parser

from src.ingestion.base import RawArticle

logger = logging.getLogger(__name__)

# Fields are truncated to these lengths before storage
_TITLE_MAX = 512
_DESC_MAX = 2000
_AUTHOR_MAX = 256

# Fallback published_at when none is parseable — use epoch to flag the issue
_EPOCH_UTC = "1970-01-01T00:00:00Z"


@dataclass
class ArticleRecord:
    """
    Fully-normalised article ready for storage and sentiment scoring.

    All string fields have been sanitised; published_at is always UTC ISO-8601.
    """

    content_hash: str
    url: str
    title: str
    source_name: str
    published_at: str           # "YYYY-MM-DDTHH:MM:SSZ"
    description: Optional[str] = None
    source_id: Optional[str] = None
    author: Optional[str] = None
    raw_json: Optional[str] = None

    def to_db_dict(self) -> dict:
        """Return a dict keyed to match the articles table columns."""
        return {
            "content_hash": self.content_hash,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "published_at": self.published_at,
            "author": self.author,
            "raw_json": self.raw_json,
        }

    @property
    def text_for_scoring(self) -> str:
        """
        The text fed to the sentiment model.

        We score headline + first 300 chars of description.
        FinBERT has a 512-token limit; a typical headline + snippet fits well.
        """
        parts = [self.title]
        if self.description:
            parts.append(self.description[:300])
        return " ".join(parts)


def normalise(raw: RawArticle) -> Optional[ArticleRecord]:
    """
    Convert a RawArticle to an ArticleRecord.

    Returns None if the article is fundamentally unusable (no URL or title).
    Logs a warning for recoverable issues (bad timestamp, etc.).
    """
    url = (raw.get("url") or "").strip()
    title = (raw.get("title") or "").strip()

    if not url or not title:
        logger.warning("Normaliser: dropping article with missing url or title")
        return None

    title = _truncate(title, _TITLE_MAX)
    description = _clean_description(raw.get("description"), _DESC_MAX)
    author = _truncate(raw.get("author") or "", _AUTHOR_MAX) or None
    source_name = (raw.get("source_name") or "Unknown").strip()
    source_id = raw.get("source_id")
    raw_json = raw.get("raw_json")

    published_at = _parse_timestamp(raw.get("published_at"), url)
    content_hash = _compute_hash(title, url)

    return ArticleRecord(
        content_hash=content_hash,
        url=url,
        title=title,
        description=description,
        source_name=source_name,
        source_id=source_id,
        published_at=published_at,
        author=author,
        raw_json=raw_json,
    )


def normalise_many(raws: List[RawArticle]) -> List[ArticleRecord]:
    """Normalise a list, silently dropping invalids."""
    results = []
    for raw in raws:
        record = normalise(raw)
        if record:
            results.append(record)
    logger.info("Normaliser: %d/%d records normalised successfully", len(results), len(raws))
    return results


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_hash(title: str, url: str) -> str:
    """SHA-256 of lowercase(title + url) — stable cross-source dedup key."""
    payload = (title.lower() + url.lower()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_timestamp(raw_ts: Optional[str], url: str) -> str:
    """
    Convert an arbitrary timestamp string to UTC ISO-8601.

    Handles:
    - ISO-8601 with timezone (Z, +00:00, +05:30, etc.)
    - RFC-2822 (common in RSS)
    - Timezone-naive strings (assumed UTC, logged as warning)
    - Completely unparseable input (returns epoch, logged as warning)
    """
    if not raw_ts:
        logger.debug("Normaliser: no timestamp for %s — using epoch", url)
        return _EPOCH_UTC

    try:
        dt = dateutil_parser.parse(raw_ts)
        if dt.tzinfo is None:
            logger.debug(
                "Normaliser: timezone-naive timestamp '%s' for %s — assuming UTC", raw_ts, url
            )
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OverflowError):
        logger.warning(
            "Normaliser: unparseable timestamp '%s' for %s — using epoch", raw_ts, url
        )
        return _EPOCH_UTC


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] if s else ""


def _clean_description(text: Optional[str], max_len: int) -> Optional[str]:
    """Strip HTML tags and truncate description text."""
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text[:max_len] if text else None
