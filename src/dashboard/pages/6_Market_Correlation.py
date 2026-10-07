"""
Page 6 — Market Correlation

Shows:
- Sentiment vs market price overlay
- Pearson correlation + p-value
- Rolling correlation (7-day window)
- Lagged correlation (-3 to +3 days)
- Lead/lag interpretation

Statistical caveat: Pearson correlation on non-stationary financial time series
is a simplification. For rigorous analysis, use Granger causality (statsmodels).
The rolling correlation gives a more reliable view of the relationship over time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.analytics.aggregator import daily_sentiment_trend, rolling_net_score
from src.analytics.correlation import CorrelationAnalyzer
from src.dashboard.shared import (
    apply_filters,
    load_articles,
    load_market,
    render_sidebar,
)
from src.config.settings import settings

filters = render_sidebar()
df_raw = load_articles(filters["start_str"], filters["end_str"])

st.title("📉 Market Correlation")
st.caption(
    "**Statistical note:** Pearson correlation on raw sentiment vs. returns "
    "is exploratory. Non-stationarity, autocorrelation, and short time windows "
    "can produce spurious results. Use for qualitative insight, not trading signals."
)

if df_raw.empty:
    st.warning("No data. Run the pipeline first.")
    st.stop()

df = apply_filters(df_raw.copy(), filters)

# ── Compute daily sentiment ───────────────────────────────────────────────────
trend_df = rolling_net_score(df, window=filters["rolling_window"])
if trend_df.empty:
    st.info("Insufficient data for correlation analysis (need ≥ 10 days).")
    st.stop()

trend_df["date"] = pd.to_datetime(trend_df["date"])
trend_df = trend_df.set_index("date")

# ── Symbol selector ───────────────────────────────────────────────────────────
available = [s for s in settings.market_symbols if s != "(none)"]
if not available:
    st.warning("No market symbols configured. Set MARKET_SYMBOLS in .env")
    st.stop()

symbol = st.selectbox("Select Market Index", options=available)

# ── Load market data ──────────────────────────────────────────────────────────
mkt_df = load_market(symbol, filters["start_str"], filters["end_str"])

if mkt_df.empty:
    st.info(
        f"No market data for {symbol}. "
        "Run the pipeline to refresh market cache: `python scripts/run_pipeline.py`"
    )
    st.stop()

mkt_df["date"] = pd.to_datetime(mkt_df["date"])
mkt_df = mkt_df.set_index("date")

# ── Aligned DataFrames ────────────────────────────────────────────────────────
analyzer = CorrelationAnalyzer()
sentiment_col = trend_df[["net_score"]].rename(columns={"net_score": "sentiment_score"})
market_col = mkt_df[["close"]]

try:
    result = analyzer.compute_daily_correlation(
        sentiment_df=sentiment_col,
        market_df=market_col,
        window=filters["rolling_window"],
    )
except Exception as exc:
    st.error(f"Correlation computation failed: {exc}")
    st.stop()

# ── KPI metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pearson r", f"{result.pearson_r:.3f}")
c2.metric("P-value", f"{result.p_value:.3f}",
          delta="Significant" if result.p_value < 0.05 else "Not significant",
          delta_color="normal")
c3.metric("Data points", f"{len(result.rolling_correlation)}")
c4.metric("Interpretation", result.interpretation[:30])

st.divider()

# ── Price + sentiment overlay ─────────────────────────────────────────────────
st.subheader(f"{symbol} Price vs Net Sentiment Score")

# Align
aligned = sentiment_col.join(market_col, how="inner")
if aligned.empty:
    st.info("No overlapping dates between sentiment and market data.")
    st.stop()

fig_overlay = make_subplots(specs=[[{"secondary_y": True}]])
fig_overlay.add_trace(
    go.Scatter(x=aligned.index, y=aligned["close"],
               name=f"{symbol} Close", line=dict(color="#3498db")),
    secondary_y=False,
)
fig_overlay.add_trace(
    go.Scatter(x=aligned.index, y=aligned["sentiment_score"],
               name="Net Sentiment", line=dict(color="#f39c12", dash="dot")),
    secondary_y=True,
)
fig_overlay.update_yaxes(title_text=f"{symbol} Price (USD)", secondary_y=False)
fig_overlay.update_yaxes(title_text="Net Sentiment (-1 to +1)", secondary_y=True,
                          range=[-1.2, 1.2])
fig_overlay.add_hline(y=0, secondary_y=True, line_dash="dash",
                       line_color="grey", opacity=0.4)
fig_overlay.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                           legend=dict(orientation="h"))
st.plotly_chart(fig_overlay, use_container_width=True)

# ── Rolling correlation ───────────────────────────────────────────────────────
st.subheader(f"Rolling {filters['rolling_window']}-Day Correlation")

roll_df = result.rolling_correlation.dropna().reset_index()
roll_df.columns = ["date", "rolling_r"]
roll_df["date"] = pd.to_datetime(roll_df["date"])

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    x=roll_df["date"], y=roll_df["rolling_r"],
    fill="tozeroy",
    fillcolor="rgba(52,152,219,0.15)",
    line=dict(color="#3498db"),
    name="Rolling Pearson r",
))
fig_roll.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
fig_roll.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.4,
                   annotation_text="Moderate +")
fig_roll.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.4,
                   annotation_text="Moderate -")
fig_roll.update_layout(
    yaxis=dict(range=[-1, 1], title="Pearson r"),
    height=280,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_roll, use_container_width=True)

# ── Lagged correlation ────────────────────────────────────────────────────────
st.subheader("Lagged Correlation — Does Sentiment Lead or Follow?")

if result.lagged_correlations:
    lags = sorted(result.lagged_correlations.keys())
    lag_vals = [result.lagged_correlations[l] for l in lags]
    best_lag = lags[int(np.argmax(np.abs(lag_vals)))]

    fig_lag = go.Figure(go.Bar(
        x=[f"Lag {l:+d}d" for l in lags],
        y=lag_vals,
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in lag_vals],
        text=[f"{v:.3f}" for v in lag_vals],
        textposition="outside",
    ))
    fig_lag.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig_lag.update_layout(
        xaxis_title="Lag (negative = sentiment leads market)",
        yaxis_title="Pearson r",
        height=280,
        margin=dict(l=0, r=0, t=10, b=60),
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    direction = "leads" if best_lag < 0 else "lags" if best_lag > 0 else "is contemporaneous with"
    st.caption(
        f"Strongest correlation at lag **{best_lag:+d} days** (r={result.lagged_correlations[best_lag]:.3f}). "
        f"This suggests sentiment **{direction}** {symbol} by **{abs(best_lag)} day(s)**. "
        f"This is indicative only — small datasets produce unstable lag estimates."
    )

# ── Multi-symbol summary ──────────────────────────────────────────────────────
st.divider()
st.subheader("All Symbols Summary")

summary_rows = []
for sym in available:
    mdf = load_market(sym, filters["start_str"], filters["end_str"])
    if mdf.empty:
        continue
    mdf = mdf.set_index(pd.to_datetime(mdf["date"]))["close"].to_frame()
    try:
        r = analyzer.compute_daily_correlation(sentiment_col, mdf, window=7)
        summary_rows.append({
            "Symbol": sym,
            "Pearson r": round(r.pearson_r, 3),
            "P-value": round(r.p_value, 3),
            "Significant": "✓" if r.p_value < 0.05 else "—",
            "Interpretation": r.interpretation,
        })
    except Exception:
        pass

if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
