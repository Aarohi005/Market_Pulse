"""
base.py — shared utilities for all ingestion clients.

Provides:
- RawArticle: a minimal typed dict for unprocessed article payloads
- retry_request: a tenacity-based decorator for HTTP calls
- IngestionError: base exception class
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TypedDict

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when an ingestion source fails unrecoverably."""


class RawArticle(TypedDict, total=False):
    """
    Minimally-typed dict produced by every ingestion client.

    Keys are intentionally loose here; the Normalizer enforces the final schema.
    """
    url: str
    title: str
    description: Optional[str]
    source_name: str
    source_id: Optional[str]
    published_at: Optional[str]   # raw string — may be any timezone format
    author: Optional[str]
    raw_json: str                 # JSON-serialised original payload


def make_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 30.0,
) -> Any:
    """
    Return a tenacity retry decorator for HTTP operations.

    Retries on connection errors and HTTP 429 / 5xx responses.
    Uses exponential back-off so we don't hammer rate-limited APIs.
    """
    return retry(
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout)
        ),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
