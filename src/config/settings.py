"""
settings.py — application-wide configuration via environment variables.

All values are read from the environment (or a .env file loaded by the caller).
Pydantic-settings validates types and provides sensible defaults.

Design note:
  Using pydantic-settings rather than raw os.getenv() gives us:
  - Type coercion (str → int, comma-list → List[str])
  - Validation errors with clear messages at startup
  - A single place to audit every config knob
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently drop unknown env vars
    )

    # ── NewsAPI ────────────────────────────────────────────────────────────────
    news_api_key: str = os.getenv("news_Api_key")
    news_api_page_size: int = 100
    news_api_topics: List[str] = [
        "finance",
        "stock market",
        "economy",
        "earnings",
        "Federal Reserve",
    ]

    # ── RSS ───────────────────────────────────────────────────────────────────
    rss_feeds: List[str] = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
        "https://www.investing.com/rss/news.rss",
        "https://www.marketwatch.com/rss/topstories",
    ]

    # ── Sentiment ─────────────────────────────────────────────────────────────
    sentiment_model: str = "ProsusAI/finbert"
    hf_home: str = "./data/hf_cache"

    # ── Database ──────────────────────────────────────────────────────────────
    db_path: str = "./data/news_sentiment.db"

    # ── Market data ───────────────────────────────────────────────────────────
    market_symbols: List[str] = ["SPY", "QQQ", "DIA"]

    # ── Pipeline ──────────────────────────────────────────────────────────────
    ingest_lookback_days: int = 7

    # ── Embeddings / Retrieval ────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "news_articles"
    retrieval_top_k: int = 20
    rerank_top_k: int = 8

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = "gemini"
    # Gemini (LangChain backend)
    google_api_key: str = "Gemini_Api_Key"
    gemini_model: str = "gemini-2.5-flash"
    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence_high_threshold: float = 80.0
    confidence_medium_threshold: float = 50.0

    # ── Topic clustering ──────────────────────────────────────────────────────
    cluster_count: int = 15
    min_cluster_size: int = 3

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def chroma_persist_dir_resolved(self) -> Path:
        """Return ChromaDB persist path, creating parent dirs if needed."""
        p = Path(self.chroma_persist_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path_resolved(self) -> Path:
        """Return DB path as an absolute Path, creating parent dirs if needed."""
        p = Path(self.db_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_level_int(self) -> int:
        return getattr(logging, self.log_level.upper(), logging.INFO)


# Module-level singleton — import this everywhere
settings = Settings()
