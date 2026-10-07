"""
newsapi_client.py — ingestion client for NewsAPI.

Fetches top/everything headlines for configurable topics.
Handles:
  - Missing API key (raises IngestionError early)
  - HTTP 429 rate-limiting (exponential back-off via tenacity)
  - Empty result sets
  - Malformed article payloads (logged, skipped)
  - Pagination (iterates pages until page_size exhausted or no more results)

Design note: newsapi-python wraps the REST API but doesn't expose retry logic.
We call it inside our own retry decorator for consistency.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

import requests
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from src.ingestion.base import IngestionError, RawArticle, make_retry_decorator

logger = logging.getLogger(__name__)

_RETRY = make_retry_decorator(max_attempts=3, min_wait=5.0, max_wait=60.0)


class NewsAPIIngestionClient:
    """
    Fetches financial news articles from NewsAPI.

    Parameters
    ----------
    api_key : str
        NewsAPI key. Raise IngestionError if empty.
    page_size : int
        Max articles per query (NewsAPI free tier max is 100).
    topics : list[str]
        Search terms / topics to query.
    lookback_days : int
        How many days back to fetch articles.
    """

    def __init__(
        self,
        api_key: str,
        page_size: int = 100,
        topics: Optional[List[str]] = None,
        lookback_days: int = 7,
    ) -> None:
        if not api_key:
            raise IngestionError(
                "NEWS_API_KEY is not set. "
                "Get a free key at https://newsapi.org/register"
            )
        self._client = NewsApiClient(api_key=api_key)
        self._page_size = min(page_size, 100)   # API hard cap
        self._topics = topics or ["finance", "stock market"]
        self._lookback_days = lookback_days

    def fetch(self) -> List[RawArticle]:
        """
        Fetch articles for all configured topics and return a combined list.

        Each topic is queried separately; duplicates are handled downstream
        by the deduplicator (URL + content hash).
        """
        all_articles: List[RawArticle] = []
        for topic in self._topics:
            try:
                articles = self._fetch_topic(topic)
                logger.info("NewsAPI: fetched %d articles for topic '%s'", len(articles), topic)
                all_articles.extend(articles)
            except IngestionError as exc:
                logger.error("NewsAPI: skipping topic '%s': %s", topic, exc)
            # Polite pause between topics to avoid triggering rate limits
            time.sleep(0.5)

        logger.info("NewsAPI: total raw articles fetched = %d", len(all_articles))
        return all_articles

    @_RETRY
    def _fetch_topic(self, topic: str) -> List[RawArticle]:
        """
        Fetch up to *page_size* articles for a single *topic*.

        Raises IngestionError on non-retryable API errors (e.g. invalid key,
        maxed-out plan).  Retryable network errors bubble up to the decorator.
        """
        from datetime import datetime, timedelta, timezone

        since = (
            datetime.now(timezone.utc) - timedelta(days=self._lookback_days)
        ).strftime("%Y-%m-%d")

        try:
            response = self._client.get_everything(
                q=topic,
                language="en",
                sort_by="publishedAt",
                page_size=self._page_size,
                from_param=since,
            )
        except NewsAPIException as exc:
            # 401 / 403 / 426 are not retryable — fail fast
            if exc.get_status() in (401, 403, 426):
                raise IngestionError(
                    f"NewsAPI non-retryable error for '{topic}': {exc}"
                ) from exc
            # Other errors (e.g. 500) let tenacity retry
            raise

        if response.get("status") != "ok":
            raise IngestionError(
                f"NewsAPI bad status for '{topic}': {response.get('message')}"
            )

        raw_articles = response.get("articles") or []
        if not raw_articles:
            logger.debug("NewsAPI: no articles returned for topic '%s'", topic)
            return []

        return [
            self._parse_article(a, topic)
            for a in raw_articles
            if self._is_valid(a, topic)
        ]

    def _is_valid(self, article: dict, topic: str) -> bool:
        """Return False (and log a warning) if required fields are missing."""
        if not article.get("url") or not article.get("title"):
            logger.warning(
                "NewsAPI: dropping malformed article (missing url/title) topic='%s'", topic
            )
            return False
        if article.get("title") == "[Removed]":
            return False
        return True

    def _parse_article(self, article: dict, topic: str) -> RawArticle:
        """Convert a raw NewsAPI dict to RawArticle."""
        source = article.get("source") or {}
        return RawArticle(
            url=article["url"],
            title=article["title"],
            description=article.get("description"),
            source_name=source.get("name") or "NewsAPI",
            source_id=source.get("id"),
            published_at=article.get("publishedAt"),
            author=article.get("author"),
            raw_json=json.dumps(article, default=str),
        )
