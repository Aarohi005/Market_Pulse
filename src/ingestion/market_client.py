"""
market_client.py — yfinance wrapper for market index OHLCV data.

Fetches daily price history for configured symbols (SPY, QQQ, DIA).
Stores results in market_cache table for dashboard overlays.

Design note: yfinance is a scraper, not an official API, so it can break
silently on Yahoo Finance HTML changes.  We treat any download failure as
non-fatal — the dashboard simply omits the market overlay if data is missing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_market_data(
    symbols: List[str],
    lookback_days: int = 90,
) -> List[Dict]:
    """
    Download daily OHLCV for *symbols* going back *lookback_days*.

    Returns a list of dicts suitable for ``upsert_market_row``.
    Returns an empty list if yfinance is unavailable or download fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed — market data unavailable")
        return []

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    rows: List[Dict] = []
    for symbol in symbols:
        symbol_rows = _fetch_symbol(yf, symbol, start, end)
        rows.extend(symbol_rows)
        logger.info("Market: fetched %d rows for %s", len(symbol_rows), symbol)

    return rows


def _fetch_symbol(
    yf,
    symbol: str,
    start: datetime,
    end: datetime,
) -> List[Dict]:
    """Download and normalise one symbol. Returns [] on any error."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
    except Exception as exc:
        logger.warning("Market: failed to fetch %s: %s", symbol, exc)
        return []

    if hist is None or hist.empty:
        logger.warning("Market: no data returned for %s", symbol)
        return []

    rows = []
    for date_idx, row_data in hist.iterrows():
        try:
            date_str = date_idx.strftime("%Y-%m-%d")
            rows.append(
                {
                    "symbol": symbol,
                    "date": date_str,
                    "open": float(row_data.get("Open", 0)) or None,
                    "high": float(row_data.get("High", 0)) or None,
                    "low": float(row_data.get("Low", 0)) or None,
                    "close": float(row_data.get("Close", 0)) or None,
                    "volume": float(row_data.get("Volume", 0)) or None,
                }
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Market: skipping row for %s/%s: %s", symbol, date_idx, exc)

    return rows
