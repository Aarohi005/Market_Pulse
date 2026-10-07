"""
Page 5 — Confidence Analysis

Shows:
- Confidence distribution (pie)
- Confidence over time (trend)
- High-confidence vs low-confidence article lists
- Component score breakdown (waterfall)
- Supporting evidence for high-confidence calls
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
    CONFIDENCE_COLORS,
    LABEL_COLORS,
    apply_filters,
    load_articles,
    load_confidence_scores,
    render_sidebar,
)

filters = render_sidebar()
df_raw = load_articles(filters["start_str"], filters["end_str"])
conf_df = load_confidence_scores()

st.title("🎯 Confidence Analysis")
st.caption("Confidence is a hybrid score from 7 inputs: retrieval similarity, reranking, article count, source diversity, source quality, sentiment agreement, LLM confidence.")

if df_raw.empty:
    st.warning("No data. Run the pipeline first.")
    st.stop()

df = apply_filters(df_raw.copy(), filters)

# ── Confidence distribution from ensemble scores ──────────────────────────────
st.subheader("Model Confidence in Ensemble Scores")

if "score" in df.columns:
    # Bin ensemble scores into HIGH/MEDIUM/LOW confidence buckets
    def _conf_bucket(score: float) -> str:
        if score >= 0.80:
            return "HIGH"
        if score >= 0.50:
            return "MEDIUM"
        return "LOW"

    df["conf_bucket"] = df["score"].map(_conf_bucket)

    col_pie, col_time = st.columns(2)

    with col_pie:
        dist = df["conf_bucket"].value_counts().reset_index()
        dist.columns = ["Confidence", "Count"]
        fig_pie = px.pie(
            dist, names="Confidence", values="Count",
            color="Confidence",
            color_discrete_map=CONFIDENCE_COLORS,
            hole=0.35,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_time:
        if "date" in df.columns:
            conf_time = (
                df.groupby(["date", "conf_bucket"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            conf_time["date"] = pd.to_datetime(conf_time["date"])
            fig_time = go.Figure()
            for level, color in CONFIDENCE_COLORS.items():
                if level in conf_time.columns:
                    fig_time.add_bar(
                        name=level,
                        x=conf_time["date"],
                        y=conf_time[level],
                        marker_color=color,
                    )
            fig_time.update_layout(
                barmode="stack",
                title="Confidence Over Time",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_time, use_container_width=True)

# ── Hybrid confidence scores (from DB) ───────────────────────────────────────
st.divider()
st.subheader("Hybrid Confidence Score Components")

if not conf_df.empty:
    # Show average component scores as a horizontal bar
    component_cols = [
        "retrieval_similarity", "reranking_score", "source_diversity",
        "source_quality", "sentiment_agreement",
    ]
    available = [c for c in component_cols if c in conf_df.columns]
    if available:
        avg_components = conf_df[available].mean().reset_index()
        avg_components.columns = ["Component", "Score"]
        avg_components["Score"] = avg_components["Score"].fillna(0).clip(0, 1)

        fig_comp = go.Figure(go.Bar(
            y=avg_components["Component"],
            x=avg_components["Score"],
            orientation="h",
            marker_color="#3498db",
            text=avg_components["Score"].map(lambda x: f"{x:.2f}"),
            textposition="outside",
        ))
        fig_comp.update_layout(
            xaxis=dict(range=[0, 1], title="Avg Normalised Score"),
            height=250,
            margin=dict(l=0, r=60, t=10, b=0),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Distribution of final scores
    if "final_score" in conf_df.columns:
        fig_hist = px.histogram(
            conf_df, x="final_score", nbins=20,
            color_discrete_sequence=["#9b59b6"],
            title="Distribution of Final Confidence Scores (0–100)",
        )
        fig_hist.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info(
        "No hybrid confidence scores found. "
        "Configure an LLM provider and re-run the pipeline to generate confidence scores."
    )

# ── High vs Low confidence headlines ─────────────────────────────────────────
st.divider()
col_high, col_low = st.columns(2)

with col_high:
    st.subheader("🟢 High Confidence Calls")
    if "conf_bucket" in df.columns:
        high_df = (
            df[df["conf_bucket"] == "HIGH"]
            .sort_values("score", ascending=False)
            .head(8)
        )
        for _, row in high_df.iterrows():
            sent_color = LABEL_COLORS.get(row["label"], "#95a5a6")
            pub = row["published_at"].strftime("%b %d") if pd.notna(row["published_at"]) else "—"
            st.markdown(
                f"<span style='color:{sent_color}'>●</span> "
                f"[{row['title'][:80]}]({row['url']})  \n"
                f"<small>{row['source_name']} · {pub} · {row['score']:.0%}</small>",
                unsafe_allow_html=True,
            )

with col_low:
    st.subheader("🔴 Low Confidence Calls")
    if "conf_bucket" in df.columns:
        low_df = (
            df[df["conf_bucket"] == "LOW"]
            .sort_values("score", ascending=True)
            .head(8)
        )
        for _, row in low_df.iterrows():
            sent_color = LABEL_COLORS.get(row["label"], "#95a5a6")
            pub = row["published_at"].strftime("%b %d") if pd.notna(row["published_at"]) else "—"
            st.markdown(
                f"<span style='color:{sent_color}'>●</span> "
                f"[{row['title'][:80]}]({row['url']})  \n"
                f"<small>{row['source_name']} · {pub} · {row['score']:.0%}</small>",
                unsafe_allow_html=True,
            )

# ── Explainability note ───────────────────────────────────────────────────────
st.divider()
st.info(
    "**Why confidence matters:** A HIGH confidence score means multiple independent "
    "signals agree (strong retrieval match, multiple quality sources, consistent FinBERT + "
    "keyword + LLM signals). LOW confidence means signals conflict or insufficient context "
    "was retrieved. Always treat LOW confidence calls with extra scepticism."
)
