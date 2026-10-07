"""
rss_client.py — ingestion client for arbitrary RSS feeds via feedparser.

Handles:
  - Network failures (tenacity retry)
  - Empty feeds (returns [] gracefully)
  - Missing/malformed fields (logged, skipped)
  - Timezone-naive timestamps (assumed UTC, flagged in log)
  - Bozo flag from feedparser (malformed XML — still try to parse entries)
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional
from urllib.parse import urlparse

import feedparser
import requests

from src.ingestion.base import IngestionError, RawArticle, make_retry_decorator

logger = logging.getLogger(__name__)

_RETRY = make_retry_decorator(max_attempts=3, min_wait=3.0, max_wait=30.0)

# feedparser's struct_time fields
_TIME_FIELDS = ("published_parsed", "updated_parsed", "created_parsed")


class RSSIngestionClient:
    """
    Fetches articles from one or more RSS/Atom feed URLs.

    Parameters
    ----------
    feed_urls : list[str]
        List of RSS feed URLs to ingest.
    request_timeout : int
        Per-feed HTTP timeout in seconds.
    """

    def __init__(
        self,
        feed_urls: Optional[List[str]] = None,
        request_timeout: int = 15,
    ) -> None:
        self._feed_urls = feed_urls or []
        self._timeout = request_timeout

    def fetch(self) -> List[RawArticle]:
        """Fetch all configured feeds and return a combined list of RawArticles."""
        all_articles: List[RawArticle] = []
        for url in self._feed_urls:
            try:
                articles = self._fetch_feed(url)
                logger.info("RSS: fetched %d entries from %s", len(articles), url)
                all_articles.extend(articles)
            except Exception as exc:
                # One bad feed should never block the rest
                logger.error("RSS: failed to fetch %s — %s", url, exc)
            time.sleep(0.2)

        logger.info("RSS: total raw articles fetched = %d", len(all_articles))
        return all_articles

    @_RETRY
    def _fetch_feed(self, url: str) -> List[RawArticle]:
        """
        Download and parse a single RSS/Atom feed.

        feedparser handles most format variations (RSS 0.9x, 1.0, 2.0, Atom).
        We pass the raw response bytes to avoid feedparser's built-in urllib
        call so our retry decorator controls the HTTP layer.
        """
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("RSS: feed not found (404): %s", url)
                return []
            raise  # re-raise for tenacity

        feed = feedparser.parse(resp.content)

        if feed.bozo and not feed.entries:
            logger.warning(
                "RSS: bozo feed with no entries at %s: %s", url, feed.bozo_exception
            )
            return []

        if feed.bozo:
            logger.debug("RSS: bozo flag set for %s (still has %d entries)", url, len(feed.entries))

        source_name = self._extract_source_name(feed, url)
        articles = []
        for entry in feed.entries:
            parsed = self._parse_entry(entry, source_name, url)
            if parsed:
                articles.append(parsed)

        return articles

    def _parse_entry(
        self, entry: feedparser.FeedParserDict, source_name: str, feed_url: str
    ) -> Optional[RawArticle]:
        """
        Convert a feedparser entry to RawArticle, or return None if invalid.
        """
        link = getattr(entry, "link", None) or getattr(entry, "id", None)
        title = getattr(entry, "title", None)

        if not link or not title:
            logger.debug("RSS: dropping entry with missing link/title from %s", feed_url)
            return None

        # Clean up title (some feeds include HTML entities)
        title = title.strip()
        if not title:
            return None

        description = self._get_summary(entry)
        published_at = self._get_published_at(entry, feed_url)
        author = getattr(entry, "author", None)

        raw_payload = {
            "link": link,
            "title": title,
            "summary": description,
            "author": author,
            "published": published_at,
            "feed_url": feed_url,
            "source_name": source_name,
        }

        return RawArticle(
            url=link,
            title=title,
            description=description,
            source_name=source_name,
            source_id=None,
            published_at=published_at,
            author=author,
            raw_json=json.dumps(raw_payload, default=str),
        )

    @staticmethod
    def _get_summary(entry: feedparser.FeedParserDict) -> Optional[str]:
        """Extract and truncate description/summary."""
        text = getattr(entry, "summary", None) or getattr(entry, "description", None)
        if not text:
            return None
        # Strip HTML tags with a simple approach — use lxml for richer cleaning
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        return text[:2000] if text else None

    @staticmethod
    def _get_published_at(
        entry: feedparser.FeedParserDict, feed_url: str
    ) -> Optional[str]:
        """
        Return an ISO-8601 UTC string from a feedparser entry's timestamp fields.

        feedparser normalises most timezone formats into struct_time (UTC).
        If none are present we return None — the normalizer will handle it.
        """
        import calendar
        from datetime import datetime, timezone

        for field in _TIME_FIELDS:
            t = getattr(entry, field, None)
            if t is not None:
                try:
                    ts = calendar.timegm(t)         # struct_time → unix timestamp (UTC)
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, ValueError, OverflowError):
                    continue

        # Fall back to the raw published string
        raw = getattr(entry, "published", None)
        if raw:
            logger.debug(
                "RSS: could not parse timestamp '%s' from %s — keeping raw", raw, feed_url
            )
            return raw

        return None

    @staticmethod
    def _extract_source_name(
        feed: feedparser.FeedParserDict, url: str
    ) -> str:
        """Derive a human-readable source name from feed metadata or URL."""
        if hasattr(feed, "feed") and feed.feed.get("title"):
            return feed.feed.title.strip()
        domain = urlparse(url).netloc
        return domain.replace("www.", "") or "RSS"
