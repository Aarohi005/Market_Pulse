"""
shared.py — utilities and shared state for all dashboard pages.

Provides:
- Sidebar filter widget (renders + returns filter values)
- Cached data loaders (wrapped in @st.cache_data)
- Common colour palette
- Pipeline refresh helper
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.aggregator import (
    daily_sentiment_trend,
    load_market_data,
    rolling_net_score,
    sentiment_by_keyword,
    source_breakdown,
    summary_stats,
    top_headlines,
)
from src.config.settings import settings
from src.storage.database import get_connection

# ── Colours ───────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}
CONFIDENCE_COLORS = {
    "HIGH":   "#2ecc71",
    "MEDIUM": "#f39c12",
    "LOW":    "#e74c3c",
}


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_articles(start: str, end: str) -> pd.DataFrame:
    """Load ensemble-scored articles with classification columns."""
    from src.storage.database import fetch_ensemble_with_articles
    with get_connection(settings.db_path) as conn:
        rows = fetch_ensemble_with_articles(conn, start_date=start, end_date=end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df


@st.cache_data(ttl=300)
def load_finbert_articles(start: str, end: str) -> pd.DataFrame:
    """Load FinBERT-only scored articles (legacy / comparison view)."""
    from src.storage.database import fetch_sentiment_with_articles
    with get_connection(settings.db_path) as conn:
        rows = fetch_sentiment_with_articles(conn, start_date=start, end_date=end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df


@st.cache_data(ttl=600)
def load_confidence_scores() -> pd.DataFrame:
    """Load all confidence scores."""
    with get_connection(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM confidence_scores ORDER BY created_at DESC LIMIT 1000"
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=600)
def load_topic_clusters() -> pd.DataFrame:
    """Load the most recent topic cluster results."""
    from src.storage.database import fetch_clusters_with_articles
    with get_connection(settings.db_path) as conn:
        rows = fetch_clusters_with_articles(conn)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=600)
def load_llm_overview() -> Optional[dict]:
    """Fetch the most recent LLM market overview analysis."""
    with get_connection(settings.db_path) as conn:
        row = conn.execute(
            """SELECT * FROM llm_analyses
               WHERE analysis_type='market_overview'
               ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
    return dict(row) if row else None


@st.cache_data(ttl=3600)
def load_market(symbol: str, start: str, end: str) -> pd.DataFrame:
    return load_market_data(settings.db_path, symbol, start, end)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render the shared sidebar and return filter values.

    Returns a dict with keys: start_str, end_str, keyword, label_filter,
    source_filter, market_symbol, rolling_window.
    """
    with st.sidebar:
        st.title("📊 Filters")

        today = date.today()
        default_start = today - timedelta(days=30)

        date_start = st.date_input("From", value=default_start, max_value=today)
        date_end   = st.date_input("To",   value=today,          max_value=today)

        if date_start > date_end:
            st.error("Start date must be ≤ end date.")
            st.stop()

        st.divider()
        keyword = st.text_input("Keyword / Ticker", placeholder="e.g. AAPL, Fed, inflation")

        st.divider()
        label_filter = st.multiselect(
            "Sentiment labels",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )

        market_symbol = st.selectbox(
            "Market overlay", options=["(none)"] + settings.market_symbols
        )
        rolling_window = st.slider("Rolling avg (days)", 3, 30, 7)

        st.divider()
        if st.button("🔄 Refresh Pipeline", type="secondary", use_container_width=True):
            _run_pipeline_subprocess()

    return {
        "start_str": date_start.strftime("%Y-%m-%d"),
        "end_str":   date_end.strftime("%Y-%m-%d"),
        "keyword": keyword.strip(),
        "label_filter": label_filter,
        "market_symbol": market_symbol,
        "rolling_window": rolling_window,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply keyword + label filters to a scored articles DataFrame."""
    if filters["keyword"]:
        kw = filters["keyword"].lower()
        mask = (
            df["title"].str.lower().str.contains(kw, na=False)
            | df.get("description", pd.Series(dtype=str)).str.lower().str.contains(kw, na=False)
        )
        df = df[mask]
    if filters["label_filter"]:
        df = df[df["label"].isin(filters["label_filter"])]
    return df


# ── KPI cards ─────────────────────────────────────────────────────────────────

def render_kpi_row(df: pd.DataFrame) -> None:
    """Render 5 KPI metric cards from a scored DataFrame."""
    stats = summary_stats(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Articles",  f"{stats['total']:,}")
    c2.metric("Positive",        f"{stats['positive_pct']}%")
    c3.metric("Negative",        f"{stats['negative_pct']}%")
    c4.metric("Neutral",         f"{stats['neutral_pct']}%")
    c5.metric("Avg Confidence",  f"{stats['avg_confidence']:.1%}")


# ── Pipeline launcher ─────────────────────────────────────────────────────────

def _run_pipeline_subprocess() -> None:
    with st.spinner("Running pipeline …"):
        try:
            result = subprocess.run(
                [sys.executable, "scripts/run_pipeline.py"],
                capture_output=True, text=True, timeout=300, cwd=str(_ROOT),
            )
            if result.returncode == 0:
                st.success("Pipeline complete — reload to see fresh data.")
                st.cache_data.clear()
            else:
                st.error(f"Pipeline failed:\n```\n{result.stderr[-1500:]}\n```")
        except subprocess.TimeoutExpired:
            st.error("Timed out (5 min).")
        except Exception as exc:
            st.error(f"Failed: {exc}")
