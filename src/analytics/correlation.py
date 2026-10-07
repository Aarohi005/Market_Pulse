"""
Sentiment–market correlation analysis.

Computes Pearson correlation, rolling correlation, and lagged correlation
between daily aggregated sentiment scores and daily market returns.
Handles data alignment, missing values, and insufficient data gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Minimum overlapping days required to produce a meaningful correlation
_MIN_OVERLAP_DAYS = 10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    """Stores the full correlation analysis for one symbol.

    Attributes
    ----------
    symbol:
        Ticker symbol this result relates to (e.g. ``"AAPL"``).
    pearson_r:
        Overall Pearson correlation coefficient in [-1, 1].
        ``float('nan')`` when insufficient data is available.
    p_value:
        Two-tailed p-value for the Pearson r test.
        ``float('nan')`` when insufficient data is available.
    rolling_correlation:
        Pandas Series indexed by date containing the rolling window
        correlation.  Empty Series when insufficient data.
    lagged_correlations:
        Dictionary mapping lag (in days) to Pearson r for that lag.
        Lags range from -3 (sentiment leads market by 3 days) to
        +3 (market leads sentiment by 3 days).
    interpretation:
        Human-readable summary of the correlation finding.
    """

    symbol: str
    pearson_r: float
    p_value: float
    rolling_correlation: pd.Series
    lagged_correlations: Dict[int, float] = field(default_factory=dict)
    interpretation: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (rolling_correlation as list)."""
        return {
            "symbol": self.symbol,
            "pearson_r": self.pearson_r,
            "p_value": self.p_value,
            "rolling_correlation": self.rolling_correlation.to_dict(),
            "lagged_correlations": self.lagged_correlations,
            "interpretation": self.interpretation,
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CorrelationAnalyzer:
    """Computes correlation between daily sentiment scores and market returns.

    The analyzer expects DataFrames with a ``DatetimeIndex`` (or a ``date``
    column) and specific column names:

    * ``sentiment_df`` must have a ``"sentiment_score"`` column containing
      daily mean sentiment values in [-1, 1].
    * ``market_df`` must have a ``"return"`` or ``"close"`` column.  When a
      ``"close"`` column is present, daily log-returns are computed
      automatically.

    Parameters
    ----------
    lag_range:
        Number of days in each direction to test for lagged correlation.
        Defaults to ``3`` (tests lags -3, -2, -1, 0, +1, +2, +3).

    Examples
    --------
    >>> analyzer = CorrelationAnalyzer()
    >>> result = analyzer.compute_daily_correlation(sentiment_df, market_df, window=7)
    >>> print(f"r={result.pearson_r:.3f}, p={result.p_value:.4f}")
    """

    def __init__(self, lag_range: int = 3) -> None:
        self.lag_range = lag_range

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def compute_daily_correlation(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        window: int = 7,
    ) -> CorrelationResult:
        """Compute full correlation analysis between sentiment and returns.

        Parameters
        ----------
        sentiment_df:
            DataFrame with daily sentiment scores.  Must contain a
            ``"sentiment_score"`` column.  Index or a ``"date"`` column
            used for alignment.
        market_df:
            DataFrame with daily market data.  Must contain a ``"return"``
            or ``"close"`` column.  Index or a ``"date"`` column used for
            alignment.
        symbol:
            Ticker symbol label embedded in the result.
        window:
            Rolling window size in trading days.  Defaults to ``7``.

        Returns
        -------
        CorrelationResult
            Full correlation analysis.  When fewer than
            ``_MIN_OVERLAP_DAYS`` overlapping days exist the numeric
            fields are set to ``NaN`` and the interpretation reflects the
            data gap.
        """
        # Prepare aligned series
        sent_series, ret_series = self._align(sentiment_df, market_df)

        overlap = sent_series.dropna().index.intersection(ret_series.dropna().index)
        n_overlap = len(overlap)

        if n_overlap < _MIN_OVERLAP_DAYS:
            logger.warning(
                "Insufficient overlapping data for %s (%d days, need %d).",
                symbol, n_overlap, _MIN_OVERLAP_DAYS,
            )
            return CorrelationResult(
                symbol=symbol,
                pearson_r=float("nan"),
                p_value=float("nan"),
                rolling_correlation=pd.Series(dtype=float),
                lagged_correlations={},
                interpretation=(
                    f"Insufficient data for {symbol}: only {n_overlap} overlapping "
                    f"days (minimum required: {_MIN_OVERLAP_DAYS})."
                ),
            )

        # Restrict both series to the overlapping date range
        s = sent_series.reindex(overlap)
        r = ret_series.reindex(overlap)

        # Drop any remaining NaN pairs
        valid = s.notna() & r.notna()
        s_clean = s[valid]
        r_clean = r[valid]

        # Overall Pearson r
        pearson_r, p_value = self._pearson(s_clean, r_clean)

        # Rolling correlation
        combined = pd.DataFrame({"sentiment": s, "returns": r})
        rolling_corr = (
            combined["sentiment"]
            .rolling(window=window, min_periods=max(2, window // 2))
            .corr(combined["returns"])
            .dropna()
        )

        # Lagged correlations
        lagged = self._compute_lagged(s, r)

        interpretation = self._interpret(symbol, pearson_r, p_value, lagged)

        return CorrelationResult(
            symbol=symbol,
            pearson_r=pearson_r,
            p_value=p_value,
            rolling_correlation=rolling_corr,
            lagged_correlations=lagged,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Multi-symbol convenience method
    # ------------------------------------------------------------------

    def compute_for_symbols(
        self,
        sentiment_df: pd.DataFrame,
        market_dfs: Dict[str, pd.DataFrame],
        window: int = 7,
    ) -> Dict[str, CorrelationResult]:
        """Compute correlations for multiple symbols in one call.

        Parameters
        ----------
        sentiment_df:
            DataFrame with a ``"sentiment_score"`` column **and** a
            ``"symbol"`` column.  Rows are filtered per symbol.
        market_dfs:
            Dictionary mapping ticker symbol (str) to its market
            DataFrame.
        window:
            Rolling window in trading days.

        Returns
        -------
        dict mapping symbol → CorrelationResult
            One result per symbol present in *market_dfs*.
        """
        results: Dict[str, CorrelationResult] = {}
        for symbol, market_df in market_dfs.items():
            if "symbol" in sentiment_df.columns:
                sym_sent = sentiment_df[sentiment_df["symbol"] == symbol].copy()
            else:
                sym_sent = sentiment_df.copy()

            if sym_sent.empty:
                logger.warning("No sentiment rows found for symbol %s.", symbol)
                results[symbol] = CorrelationResult(
                    symbol=symbol,
                    pearson_r=float("nan"),
                    p_value=float("nan"),
                    rolling_correlation=pd.Series(dtype=float),
                    lagged_correlations={},
                    interpretation=f"No sentiment data found for {symbol}.",
                )
                continue

            results[symbol] = self.compute_daily_correlation(
                sentiment_df=sym_sent,
                market_df=market_df,
                symbol=symbol,
                window=window,
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align(
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """Align sentiment and market DataFrames on a DatetimeIndex.

        Accepts either a DatetimeIndex or a ``"date"`` column.  Returns
        two aligned Series.
        """
        def to_date_index(df: pd.DataFrame, score_col: str) -> pd.Series:
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
            if score_col not in df.columns:
                raise KeyError(
                    f"Expected column '{score_col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
            return df[score_col].sort_index()

        sent_series = to_date_index(sentiment_df, "sentiment_score")

        # Determine return series from market_df
        mdf = market_df.copy()
        if "date" in mdf.columns:
            mdf["date"] = pd.to_datetime(mdf["date"])
            mdf = mdf.set_index("date")
        else:
            mdf.index = pd.to_datetime(mdf.index)
        mdf = mdf.sort_index()

        if "return" in mdf.columns:
            ret_series = mdf["return"]
        elif "close" in mdf.columns:
            # Compute log returns from close prices
            ret_series = np.log(mdf["close"] / mdf["close"].shift(1))
        else:
            raise KeyError(
                "market_df must contain a 'return' or 'close' column. "
                f"Available columns: {list(mdf.columns)}"
            )

        return sent_series, ret_series

    def _compute_lagged(
        self,
        sentiment: pd.Series,
        returns: pd.Series,
    ) -> Dict[int, float]:
        """Compute Pearson r for a range of integer lags.

        A positive lag means sentiment is shifted forward (market leads
        sentiment); a negative lag means sentiment leads the market.

        Parameters
        ----------
        sentiment:
            Daily sentiment score series with a DatetimeIndex.
        returns:
            Daily return series with a DatetimeIndex.

        Returns
        -------
        dict mapping int lag → float Pearson r
        """
        lagged: Dict[int, float] = {}
        for lag in range(-self.lag_range, self.lag_range + 1):
            shifted = sentiment.shift(lag)
            combined = pd.DataFrame({"s": shifted, "r": returns}).dropna()
            if len(combined) < _MIN_OVERLAP_DAYS:
                lagged[lag] = float("nan")
                continue
            r, _ = self._pearson(combined["s"], combined["r"])
            lagged[lag] = r
        return lagged

    @staticmethod
    def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
        """Compute Pearson r and p-value for two aligned series.

        Returns ``(nan, nan)`` on error or when data is insufficient.
        """
        try:
            r, p = stats.pearsonr(x.values, y.values)
            return float(r), float(p)
        except Exception as exc:
            logger.warning("Pearson computation failed: %s", exc)
            return float("nan"), float("nan")

    @staticmethod
    def _interpret(
        symbol: str,
        r: float,
        p: float,
        lagged: Dict[int, float],
    ) -> str:
        """Generate a human-readable interpretation string.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        r:
            Overall Pearson correlation coefficient.
        p:
            Two-tailed p-value.
        lagged:
            Lagged correlation dictionary.

        Returns
        -------
        str
            A concise English interpretation of the correlation results.
        """
        if np.isnan(r):
            return f"Correlation for {symbol} could not be computed (insufficient data)."

        significance = "statistically significant" if p < 0.05 else "not statistically significant"
        direction = (
            "positive" if r > 0.1
            else "negative" if r < -0.1
            else "negligible"
        )
        strength = (
            "strong" if abs(r) >= 0.6
            else "moderate" if abs(r) >= 0.3
            else "weak"
        )

        # Find the lag with the highest absolute correlation
        valid_lags = {lag: val for lag, val in lagged.items() if not np.isnan(val)}
        best_lag_str = ""
        if valid_lags:
            best_lag = max(valid_lags, key=lambda l: abs(valid_lags[l]))
            best_r = valid_lags[best_lag]
            if best_lag < 0:
                lead_str = f"sentiment leading market by {abs(best_lag)} day(s)"
            elif best_lag > 0:
                lead_str = f"market leading sentiment by {best_lag} day(s)"
            else:
                lead_str = "contemporaneous (lag=0)"
            best_lag_str = (
                f" The strongest lagged correlation is r={best_r:.3f} at {lead_str}."
            )

        return (
            f"{symbol}: Overall Pearson r={r:.3f} (p={p:.4f}), indicating a {strength} "
            f"{direction} correlation ({significance}).{best_lag_str}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"CorrelationAnalyzer(lag_range={self.lag_range})"
