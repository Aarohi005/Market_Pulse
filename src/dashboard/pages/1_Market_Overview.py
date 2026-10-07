"""
Page 1 — Market Overview

Shows:
- Current overall sentiment + confidence (from latest LLM analysis)
- Ensemble KPI cards
- Top positive & negative drivers
- Latest LLM-generated market commentary
- Most important recent articles
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.dashboard.shared import (
    CONFIDENCE_COLORS,
    LABEL_COLORS,
    apply_filters,
    load_articles,
    load_llm_overview,
    render_kpi_row,
    render_sidebar,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
filters = render_sidebar()

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = load_articles(filters["start_str"], filters["end_str"])
llm_overview = load_llm_overview()

st.title("🌐 Market Overview")
st.caption(f"Sentiment intelligence · {filters['start_str']} → {filters['end_str']}")

if df_raw.empty:
    st.warning("No scored articles found. Run the pipeline first: `python scripts/run_pipeline.py`")
    st.stop()

df = apply_filters(df_raw.copy(), filters)
if df.empty:
    st.info("No articles match the current filters.")
    st.stop()

# ── LLM sentiment banner ──────────────────────────────────────────────────────
if llm_overview:
    col_sent, col_conf, col_info = st.columns([1, 1, 3])
    sentiment = (llm_overview.get("sentiment_label") or "unknown").capitalize()
    confidence = llm_overview.get("confidence") or "LOW"
    sent_color = LABEL_COLORS.get(sentiment.lower(), "#95a5a6")
    conf_color = CONFIDENCE_COLORS.get(confidence, "#e74c3c")

    col_sent.markdown(
        f"<div style='background:{sent_color};border-radius:8px;padding:12px;text-align:center;"
        f"color:white;font-size:1.4em;font-weight:bold'>{sentiment}</div>",
        unsafe_allow_html=True,
    )
    col_conf.markdown(
        f"<div style='background:{conf_color};border-radius:8px;padding:12px;text-align:center;"
        f"color:white;font-size:1.4em;font-weight:bold'>Confidence: {confidence}</div>",
        unsafe_allow_html=True,
    )
    if llm_overview.get("raw_text"):
        col_info.markdown(f"**LLM Commentary:** {llm_overview['raw_text'][:400]}…")

    st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
render_kpi_row(df)
st.divider()

# ── Sentiment gauge ───────────────────────────────────────────────────────────
col_gauge, col_drivers = st.columns([1, 2])

with col_gauge:
    st.subheader("Net Sentiment Score")
    total = len(df)
    pos = (df["label"] == "positive").sum()
    neg = (df["label"] == "negative").sum()
    net = (pos - neg) / max(total, 1)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(net * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Net Score (-100 to +100)"},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"color": LABEL_COLORS["positive"] if net >= 0 else LABEL_COLORS["negative"]},
            "steps": [
                {"range": [-100, -33], "color": "#fde8e8"},
                {"range": [-33, 33],   "color": "#f5f5f5"},
                {"range": [33, 100],   "color": "#e8fde8"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": net * 100},
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_drivers:
    st.subheader("Key Drivers (from LLM)")
    if llm_overview and llm_overview.get("key_signals"):
        try:
            drivers = json.loads(llm_overview["key_signals"])
            for i, d in enumerate(drivers[:6], 1):
                st.markdown(f"**{i}.** {d}")
        except (json.JSONDecodeError, TypeError):
            st.info("No structured drivers available yet.")
    else:
        st.info("Run the pipeline with an LLM provider configured to see AI-generated drivers.")

# ── Top headlines ─────────────────────────────────────────────────────────────
st.divider()
col_pos, col_neg = st.columns(2)

with col_pos:
    st.subheader("🟢 Most Positive Headlines")
    pos_df = df[df["label"] == "positive"].nlargest(8, "score")
    for _, row in pos_df.iterrows():
        pub = row["published_at"].strftime("%b %d") if pd.notna(row["published_at"]) else "—"
        st.markdown(
            f"[{row['title'][:90]}]({row['url']})  \n"
            f"<small>{row['source_name']} · {pub} · {row['score']:.0%}</small>",
            unsafe_allow_html=True,
        )
        st.divider()

with col_neg:
    st.subheader("🔴 Most Negative Headlines")
    neg_df = df[df["label"] == "negative"].nlargest(8, "score")
    for _, row in neg_df.iterrows():
        pub = row["published_at"].strftime("%b %d") if pd.notna(row["published_at"]) else "—"
        st.markdown(
            f"[{row['title'][:90]}]({row['url']})  \n"
            f"<small>{row['source_name']} · {pub} · {row['score']:.0%}</small>",
            unsafe_allow_html=True,
        )
        st.divider()

# ── Ensemble signal breakdown ─────────────────────────────────────────────────
st.divider()
st.subheader("Ensemble Signal Breakdown")

signal_cols = st.columns(3)
for col, (sig_label, sig_col, conf_col) in zip(
    signal_cols,
    [("FinBERT", "finbert_label", "finbert_score"),
     ("Keyword", "keyword_label", "keyword_score"),
     ("Ensemble", "label", "score")],
):
    with col:
        if sig_col in df.columns:
            counts = df[sig_col].value_counts()
            for lbl in ["positive", "negative", "neutral"]:
                n = counts.get(lbl, 0)
                col.metric(f"{sig_label} {lbl.capitalize()}", n)
        else:
            col.info(f"{sig_label} not available")
