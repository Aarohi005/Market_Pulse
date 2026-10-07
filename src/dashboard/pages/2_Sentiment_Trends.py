"""
Page 2 — Sentiment Trends

Shows:
- Daily stacked bar (positive/negative/neutral counts)
- Rolling net score overlay
- Ensemble vs FinBERT comparison line
- Full filterable article table
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.analytics.aggregator import daily_sentiment_trend, rolling_net_score
from src.dashboard.shared import (
    LABEL_COLORS,
    apply_filters,
    load_articles,
    load_finbert_articles,
    render_kpi_row,
    render_sidebar,
)

filters = render_sidebar()
df_raw = load_articles(filters["start_str"], filters["end_str"])

st.title("📈 Sentiment Trends")

if df_raw.empty:
    st.warning("No data. Run the pipeline first.")
    st.stop()

df = apply_filters(df_raw.copy(), filters)
if df.empty:
    st.info("No articles match current filters.")
    st.stop()

render_kpi_row(df)
st.divider()

# ── Trend chart ───────────────────────────────────────────────────────────────
st.subheader("Daily Sentiment Trend")
trend_df = rolling_net_score(df, window=filters["rolling_window"])

if not trend_df.empty:
    daily_df = daily_sentiment_trend(df)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    fig = go.Figure()
    for lbl, color in LABEL_COLORS.items():
        if lbl in daily_df.columns:
            fig.add_bar(
                name=lbl.capitalize(),
                x=daily_df["date"],
                y=daily_df[lbl],
                marker_color=color,
                opacity=0.65,
            )

    trend_df["date"] = pd.to_datetime(trend_df["date"])
    fig.add_trace(go.Scatter(
        name=f"{filters['rolling_window']}d Rolling Net Score",
        x=trend_df["date"],
        y=trend_df["rolling_net_score"],
        mode="lines",
        line=dict(color="#f39c12", width=2.5),
        yaxis="y2",
    ))

    fig.update_layout(
        barmode="stack",
        yaxis_title="Article Count",
        yaxis2=dict(title="Net Score", overlaying="y", side="right",
                    range=[-1, 1], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── FinBERT vs Ensemble comparison ───────────────────────────────────────────
st.subheader("Ensemble vs FinBERT Signal Comparison")
fb_raw = load_finbert_articles(filters["start_str"], filters["end_str"])

if not fb_raw.empty and not df.empty:
    fb_trend = daily_sentiment_trend(fb_raw)
    ens_trend = daily_sentiment_trend(df)

    fb_trend["date"] = pd.to_datetime(fb_trend["date"])
    ens_trend["date"] = pd.to_datetime(ens_trend["date"])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        name="FinBERT Net Score",
        x=fb_trend["date"], y=fb_trend["net_score"],
        mode="lines", line=dict(color="#3498db", width=2),
    ))
    fig2.add_trace(go.Scatter(
        name="Ensemble Net Score",
        x=ens_trend["date"], y=ens_trend["net_score"],
        mode="lines", line=dict(color="#9b59b6", width=2, dash="dot"),
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.4)
    fig2.update_layout(
        yaxis_title="Net Score",
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Comparison unavailable — pipeline ensemble not yet run.")

# ── Full article table ────────────────────────────────────────────────────────
st.divider()
with st.expander(f"📋 Article Table ({len(df):,})", expanded=False):
    tbl = df[["published_at", "source_name", "title", "label", "score",
              "finbert_label", "keyword_label"]].copy()
    tbl["published_at"] = tbl["published_at"].dt.strftime("%Y-%m-%d %H:%M")
    tbl["score"] = tbl["score"].map(lambda x: f"{x:.1%}")
    st.dataframe(tbl.rename(columns={
        "published_at": "Published", "source_name": "Source", "title": "Title",
        "label": "Ensemble", "score": "Score",
        "finbert_label": "FinBERT", "keyword_label": "Keyword",
    }), use_container_width=True, hide_index=True)
