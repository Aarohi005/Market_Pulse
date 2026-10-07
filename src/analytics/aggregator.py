"""
aggregator.py — compute analytics DataFrames from the SQLite store.

All functions return pandas DataFrames ready for Plotly charts.
No business logic lives in the dashboard layer — it just calls these functions
and passes results to Plotly.

Caching: Streamlit's @st.cache_data is applied at the call site in app.py.
This module is cache-agnostic to keep it testable without Streamlit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from src.storage.database import fetch_market_data, fetch_sentiment_with_articles, get_connection

logger = logging.getLogger(__name__)


# ── Core query ────────────────────────────────────────────────────────────────

def load_scored_articles(
    db_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all scored articles as a DataFrame.

    Columns: id, title, url, source_name, published_at, description,
             label, score, model_name
    """
    with get_connection(db_path) as conn:
        rows = fetch_sentiment_with_articles(conn, start_date, end_date)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df


# ── Aggregations ─────────────────────────────────────────────────────────────

def daily_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily sentiment counts and net score.

    Returns columns: date, positive, negative, neutral, total, net_score

    net_score = (positive - negative) / total  — ranges -1 to +1
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "positive", "negative", "neutral", "total", "net_score"])

    pivoted = (
        df.groupby(["date", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ("positive", "negative", "neutral"):
        if col not in pivoted.columns:
            pivoted[col] = 0

    pivoted["total"] = pivoted["positive"] + pivoted["negative"] + pivoted["neutral"]
    pivoted["net_score"] = (pivoted["positive"] - pivoted["negative"]) / pivoted["total"].clip(lower=1)
    pivoted = pivoted.sort_values("date")
    return pivoted


def source_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Article counts and avg sentiment score per source.

    Returns columns: source_name, count, avg_score, dominant_label
    """
    if df.empty:
        return pd.DataFrame(columns=["source_name", "count", "avg_score", "dominant_label"])

    agg = (
        df.groupby("source_name")
        .agg(
            count=("id", "count"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )

    dominant = (
        df.groupby(["source_name", "label"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("source_name")
        .rename(columns={"label": "dominant_label"})[["source_name", "dominant_label"]]
    )

    return agg.merge(dominant, on="source_name", how="left").sort_values("count", ascending=False)


def top_headlines(
    df: pd.DataFrame,
    label: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the *n* highest-confidence articles for a given *label*.

    Returns columns: title, url, source_name, published_at, score
    """
    if df.empty:
        return pd.DataFrame()

    subset = df[df["label"] == label].nlargest(n, "score")
    return subset[["title", "url", "source_name", "published_at", "score"]].copy()


def sentiment_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Filter DataFrame to articles containing *keyword* in title or description.
    """
    if df.empty or not keyword:
        return df

    keyword_lower = keyword.lower()
    mask = (
        df["title"].str.lower().str.contains(keyword_lower, na=False)
        | df["description"].str.lower().str.contains(keyword_lower, na=False)
    )
    return df[mask].copy()


def rolling_net_score(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Compute a rolling *window*-day average of net_score.

    Returns columns: date, net_score, rolling_net_score
    """
    trend = daily_sentiment_trend(df)
    if trend.empty:
        return trend

    trend = trend.set_index("date").sort_index()
    trend["rolling_net_score"] = trend["net_score"].rolling(window=window, min_periods=1).mean()
    return trend.reset_index()


# ── Market data ───────────────────────────────────────────────────────────────

def load_market_data(
    db_path: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load cached market OHLCV as a DataFrame.

    Returns columns: symbol, date, open, high, low, close, volume
    """
    with get_connection(db_path) as conn:
        rows = fetch_market_data(conn, symbol, start_date, end_date)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


# ── Summary stats ─────────────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> dict:
    """
    Return a dict of headline KPIs for the dashboard summary cards.
    """
    if df.empty:
        return {
            "total": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "avg_confidence": 0.0,
        }

    total = len(df)
    counts = df["label"].value_counts()
    return {
        "total": total,
        "positive_pct": round(counts.get("positive", 0) / total * 100, 1),
        "negative_pct": round(counts.get("negative", 0) / total * 100, 1),
        "neutral_pct": round(counts.get("neutral", 0) / total * 100, 1),
        "avg_confidence": round(df["score"].mean(), 3),
    }
