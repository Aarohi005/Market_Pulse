"""
Page 4 — Topic Analysis

Shows:
- Topic cluster sentiment chart
- Cluster drill-down (articles per cluster)
- Topic enum distribution
- Classification breakdown by asset class + region
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.dashboard.shared import (
    LABEL_COLORS,
    apply_filters,
    load_articles,
    load_topic_clusters,
    render_sidebar,
)

filters = render_sidebar()
df_raw = load_articles(filters["start_str"], filters["end_str"])
clusters_df = load_topic_clusters()

st.title("🏷️ Topic Analysis")

if df_raw.empty:
    st.warning("No data. Run the pipeline first.")
    st.stop()

df = apply_filters(df_raw.copy(), filters)

# ── Topic enum distribution ───────────────────────────────────────────────────
if "topic" in df.columns:
    st.subheader("Sentiment by Topic")
    topic_sent = (
        df.groupby(["topic", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("positive", "negative", "neutral"):
        if col not in topic_sent.columns:
            topic_sent[col] = 0
    topic_sent = topic_sent[topic_sent["topic"].notna()]
    topic_sent["total"] = topic_sent[["positive", "negative", "neutral"]].sum(axis=1)
    topic_sent = topic_sent.sort_values("total", ascending=False).head(14)

    fig_topic = go.Figure()
    for lbl, color in LABEL_COLORS.items():
        if lbl in topic_sent.columns:
            fig_topic.add_bar(
                name=lbl.capitalize(),
                x=topic_sent["topic"],
                y=topic_sent[lbl],
                marker_color=color,
            )
    fig_topic.update_layout(
        barmode="stack",
        xaxis_title="Topic",
        yaxis_title="Article Count",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_topic, use_container_width=True)
else:
    st.info("Topic classification not yet run. Ensure the pipeline classification stage has completed.")

# ── Asset class + Region breakdowns ──────────────────────────────────────────
col_ac, col_reg = st.columns(2)

with col_ac:
    st.subheader("Asset Class Distribution")
    if "asset_class" in df.columns:
        ac_counts = df["asset_class"].value_counts().reset_index()
        ac_counts.columns = ["asset_class", "count"]
        fig_ac = px.pie(
            ac_counts, names="asset_class", values="count",
            hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_ac.update_traces(textinfo="percent+label")
        fig_ac.update_layout(height=300, margin=dict(l=0, r=0, t=5, b=0))
        st.plotly_chart(fig_ac, use_container_width=True)
    else:
        st.info("Asset class data not available.")

with col_reg:
    st.subheader("Region Distribution")
    if "region" in df.columns:
        reg_counts = df["region"].value_counts().reset_index()
        reg_counts.columns = ["region", "count"]
        fig_reg = px.bar(
            reg_counts, x="region", y="count",
            color_discrete_sequence=["#3498db"],
        )
        fig_reg.update_layout(height=300, margin=dict(l=0, r=0, t=5, b=0),
                              xaxis_title="Region", yaxis_title="Articles")
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Region data not available.")

# ── Topic clusters ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Topic Clusters (K-means)")

if clusters_df.empty:
    st.info(
        "No clusters found. Run the pipeline with clustering enabled: "
        "`python scripts/run_pipeline.py` (requires articles with embeddings)."
    )
else:
    fig_clust = go.Figure()
    colors = [
        LABEL_COLORS.get(row["dominant_sentiment"].lower(), "#95a5a6")
        for _, row in clusters_df.iterrows()
    ]
    fig_clust.add_trace(go.Bar(
        x=clusters_df["cluster_label"],
        y=clusters_df["size"],
        marker_color=colors,
        text=clusters_df["dominant_sentiment"],
        textposition="outside",
    ))
    fig_clust.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Articles",
        height=360,
        margin=dict(l=0, r=0, t=10, b=60),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_clust, use_container_width=True)

    # Cluster detail table
    with st.expander("Cluster Details", expanded=False):
        display = clusters_df[[
            "cluster_label", "topic_enum", "size",
            "dominant_sentiment", "avg_sentiment_score",
        ]].rename(columns={
            "cluster_label": "Cluster", "topic_enum": "Topic",
            "size": "Articles", "dominant_sentiment": "Dominant Sentiment",
            "avg_sentiment_score": "Avg Score",
        })
        display["Avg Score"] = display["Avg Score"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)
