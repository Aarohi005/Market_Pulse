"""
Page 3 — Source Analysis

Shows:
- Article volume by source (pie + bar)
- Average sentiment score per source
- Source quality ranking
- Sentiment breakdown per source
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.analytics.aggregator import source_breakdown
from src.dashboard.shared import (
    LABEL_COLORS,
    apply_filters,
    load_articles,
    render_sidebar,
)
from src.retrieval.reranker import SOURCE_QUALITY

filters = render_sidebar()
df_raw = load_articles(filters["start_str"], filters["end_str"])

st.title("📰 Source Analysis")

if df_raw.empty:
    st.warning("No data. Run the pipeline first.")
    st.stop()

df = apply_filters(df_raw.copy(), filters)
if df.empty:
    st.info("No articles match current filters.")
    st.stop()

src_df = source_breakdown(df)
if src_df.empty:
    st.info("No source data.")
    st.stop()

# ── Volume + sentiment ────────────────────────────────────────────────────────
col_pie, col_bar = st.columns(2)

with col_pie:
    st.subheader("Article Volume by Source")
    fig_pie = px.pie(
        src_df, names="source_name", values="count",
        hole=0.35, color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    st.subheader("Avg Sentiment Score by Source")
    src_sorted = src_df.sort_values("avg_score", ascending=True).tail(15)
    colors = [LABEL_COLORS["positive"] if v >= 0.5 else LABEL_COLORS["negative"]
              for v in src_sorted["avg_score"]]
    fig_bar = go.Figure(go.Bar(
        y=src_sorted["source_name"],
        x=src_sorted["avg_score"],
        orientation="h",
        marker_color=colors,
    ))
    fig_bar.update_layout(
        xaxis_title="Avg Confidence Score (0–1)",
        height=350, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Source quality ranking ────────────────────────────────────────────────────
st.divider()
st.subheader("Source Trust Ranking")
st.caption("Quality scores are fixed tiers based on editorial standards. Upgrade to API-sourced trust signals for dynamic ranking.")

quality_rows = []
for source in src_df["source_name"].tolist():
    quality = SOURCE_QUALITY.get(source, SOURCE_QUALITY.get("default", 0.55))
    quality_rows.append({"Source": source, "Quality Score": quality,
                          "Articles": src_df[src_df["source_name"] == source]["count"].values[0]})

quality_df = pd.DataFrame(quality_rows).sort_values("Quality Score", ascending=False)
st.dataframe(quality_df, use_container_width=True, hide_index=True)

# ── Sentiment heatmap per source ──────────────────────────────────────────────
st.divider()
st.subheader("Sentiment Breakdown per Source")

pivot = (
    df.groupby(["source_name", "label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
for col in ("positive", "negative", "neutral"):
    if col not in pivot.columns:
        pivot[col] = 0
pivot["total"] = pivot[["positive", "negative", "neutral"]].sum(axis=1)
pivot["pos_pct"] = (pivot["positive"] / pivot["total"] * 100).round(1)
pivot["neg_pct"] = (pivot["negative"] / pivot["total"] * 100).round(1)
pivot = pivot.sort_values("total", ascending=False)

fig_heat = go.Figure()
for lbl, color in LABEL_COLORS.items():
    if lbl in pivot.columns:
        fig_heat.add_bar(
            name=lbl.capitalize(),
            x=pivot["source_name"],
            y=pivot[lbl],
            marker_color=color,
        )
fig_heat.update_layout(
    barmode="stack",
    xaxis_title="Source",
    yaxis_title="Article Count",
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_heat, use_container_width=True)
