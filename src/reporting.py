"""
Current monolithic reporting and analysis module.

This file contains report formatting, chart generation, validation summaries,
CSV export, console summaries, and several research-lab calculations. It is
preserved to keep existing outputs stable, but should eventually be split into
focused analysis, lab, and presentation modules.
"""
from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET_BINS = [-np.inf, -0.20, -0.10, 0.00, 0.10, 0.20, 0.40, np.inf]
_BUCKET_LABELS = [
    "< -20%",
    "-20% to -10%",
    "-10% to 0%",
    "0% to 10%",
    "10% to 20%",
    "20% to 40%",
    "> 40%",
]

_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]
_HIGH_SIGNAL_THRESHOLD = 0.40
_LOW_SIGNAL_THRESHOLD = -0.20
_MIN_VERDICT_VALID_20D = 20
_BROAD_SECTOR_GROUPS = {
    "growth_risk_on": [
        "technology",
        "communication_services",
        "consumer_discretionary",
    ],
    "defensive": [
        "consumer_staples",
        "utilities",
        "healthcare",
    ],
    "cyclical_value": [
        "energy",
        "financials",
        "industrials",
        "materials",
    ],
    "rate_sensitive": [
        "real_estate",
        "utilities",
        "financials",
    ],
}
_PERSONALITY_LABELS = [
    "secular_winner",
    "uptrend",
    "flat",
    "laggard",
    "deep_drawdown",
    "unknown",
]
_PERSONALITY_GROUPS = [
    "winner_or_uptrend",
    "flat_or_laggard",
    "deep_drawdown",
    "unknown",
]
_SIGNAL_CONDITIONS = [
    "High signal >= 40%",
    "Low signal <= -20%",
    "Neutral -10% to +10%",
]
_PATH_HIT_COLUMNS = [
    ("hit_plus_5_before_minus_5", "hit_plus_5_before_minus_5_rate"),
    ("hit_plus_10_before_minus_5", "hit_plus_10_before_minus_5_rate"),
    ("hit_plus_10_before_minus_10", "hit_plus_10_before_minus_10_rate"),
    ("hit_plus_15_before_minus_10", "hit_plus_15_before_minus_10_rate"),
]
_SIGNAL_LAB_CONDITIONS = [
    ("Markov high", "signal_markov_high"),
    ("Markov low", "signal_markov_low"),
    ("RSI/BB oversold", "signal_rsi_bb"),
    ("RSI/BB + Markov high", "signal_rsi_bb_markov_high"),
    ("RSI/BB + Markov low", "signal_rsi_bb_markov_low"),
    ("RSI/BB + Markov not bearish", "signal_rsi_bb_markov_not_bearish"),
    ("RSI/BB + deep drawdown", "signal_rsi_bb_deep_drawdown"),
    (
        "RSI/BB + deep drawdown + Markov high",
        "signal_rsi_bb_deep_drawdown_markov_high",
    ),
    (
        "RSI/BB + deep drawdown + Markov low",
        "signal_rsi_bb_deep_drawdown_markov_low",
    ),
]
_SIGNAL_LAB_HIT_COLUMNS = [
    ("hit_plus_5_before_minus_5", "hit_plus_5_before_minus_5_rate"),
    ("hit_plus_10_before_minus_5", "hit_plus_10_before_minus_5_rate"),
    ("hit_plus_10_before_minus_10", "hit_plus_10_before_minus_10_rate"),
]

_CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, sans-serif", size=12),
    margin=dict(l=55, r=30, t=50, b=50),
    height=380,
)

# ── Private formatting helpers ───────────────────────────────────────────────

def _pct(x: float, decimals: int = 2, colored: bool = True) -> str:
    """Format a fraction as a percentage string, optionally HTML color-coded."""
    if pd.isna(x):
        return '<span style="color:#aaa">—</span>' if colored else "—"
    pct = x * 100
    if colored:
        c = "#2ca02c" if pct > 0 else ("#d62728" if pct < 0 else "#555")
        return f'<span style="color:{c}">{pct:+.{decimals}f}%</span>'
    return f"{pct:.{decimals}f}%"


def _assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bucket"] = pd.cut(
        out["signal"],
        bins=_BUCKET_BINS,
        labels=_BUCKET_LABELS,
        right=True,
    )
    return out


def _valid_count(series: pd.Series) -> int:
    """Count non-missing observations in a return series."""
    return int(series.dropna().shape[0])


def positive_rate(series: pd.Series) -> float:
    """Fraction of non-missing returns that are positive."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float((valid > 0).mean())


def _numeric_series(subset: pd.DataFrame, col: str) -> pd.Series:
    if col not in subset.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(subset[col], errors="coerce")


def _mean_col(subset: pd.DataFrame, col: str) -> float:
    values = _numeric_series(subset, col).dropna()
    return float(values.mean()) if len(values) else np.nan


def _median_col(subset: pd.DataFrame, col: str) -> float:
    values = _numeric_series(subset, col).dropna()
    return float(values.median()) if len(values) else np.nan


def _excess_20d_stats(subset: pd.DataFrame) -> dict[str, Any]:
    """20D baseline-adjusted stats for any row subset."""
    sector_excess = _numeric_series(subset, "excess_vs_sector_etf_20d")
    return {
        "avg_excess_vs_spy_20d": _mean_col(subset, "excess_vs_spy_20d"),
        "avg_excess_vs_qqq_20d": _mean_col(subset, "excess_vs_qqq_20d"),
        "avg_excess_vs_sector_etf_20d": _mean_col(subset, "excess_vs_sector_etf_20d"),
        "median_excess_vs_sector_etf_20d": _median_col(subset, "excess_vs_sector_etf_20d"),
        "valid_excess_vs_sector_etf_20d_rows": _valid_count(sector_excess),
        "excess_vs_sector_etf_20d_win_rate": positive_rate(sector_excess),
    }


def _signal_stats(subset: pd.DataFrame) -> dict:
    """Compute forward-return stats for an arbitrary subset of rows."""
    stats: dict[str, Any] = {"rows": len(subset)}
    for d in [5, 10, 20]:
        col = f"future_{d}d_return"
        if col not in subset.columns:
            continue
        valid = subset[col].dropna()
        n = len(valid)
        stats[f"valid_{d}d_rows"] = n
        stats[f"avg_{d}d"] = float(valid.mean()) if n > 0 else np.nan
        stats[f"med_{d}d"] = float(valid.median()) if n > 0 else np.nan
        stats[f"win_{d}d"] = positive_rate(subset[col])
    stats.update(_excess_20d_stats(subset))
    return stats


def _return_stats(subset: pd.DataFrame, days: list[int]) -> dict[str, Any]:
    """Return count, mean, median, and positive-rate stats for selected horizons."""
    stats: dict[str, Any] = {"rows": len(subset)}
    for d in days:
        col = f"future_{d}d_return"
        if col not in subset.columns:
            stats[f"valid_{d}d_rows"] = 0
            stats[f"avg_{d}d_return"] = np.nan
            stats[f"median_{d}d_return"] = np.nan
            stats[f"{d}d_win_rate"] = np.nan
            continue

        valid = subset[col].dropna()
        stats[f"valid_{d}d_rows"] = len(valid)
        stats[f"avg_{d}d_return"] = float(valid.mean()) if len(valid) else np.nan
        stats[f"median_{d}d_return"] = float(valid.median()) if len(valid) else np.nan
        stats[f"{d}d_win_rate"] = positive_rate(subset[col])
    stats.update(_excess_20d_stats(subset))
    return stats


def _warning_html(message: str) -> str:
    return (
        '<p class="filter-note" style="color:#b35c00">'
        f"Warning: {_html.escape(message)}</p>"
    )


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns)


def _fmt_count(x: Any) -> str:
    return f"{int(x):,}" if not pd.isna(x) else "0"


def _normalize_ticker_groups(
    ticker_groups: dict[str, list[str]] | None,
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Keep configured groups that have at least one ticker in the dataset."""
    if not ticker_groups:
        return {}

    available = set(df["ticker"].dropna().astype(str).str.upper())
    groups: dict[str, list[str]] = {}
    for group, tickers in ticker_groups.items():
        normalized: list[str] = []
        for raw in tickers:
            ticker = str(raw).strip().upper()
            if ticker in available and ticker not in normalized:
                normalized.append(ticker)
        if normalized:
            groups[str(group)] = normalized
    return groups


# ── Public summary functions ─────────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame) -> str:
    """Human-readable console summary of the signal dataset."""
    state_counts = (
        df["state"].map({1: "Bull", 0: "Sideways", -1: "Bear"}).value_counts()
    )
    lines = [
        f"Tickers    : {df['ticker'].nunique()}",
        f"Total rows : {len(df):,}",
        f"Date range : {df['date'].min().date()} -> {df['date'].max().date()}",
        f"State dist : {state_counts.to_dict()}",
        f"Signal mean: {df['signal'].mean():.4f}",
        f"Signal std : {df['signal'].std():.4f}",
        f"Signal min : {df['signal'].min():.4f}",
        f"Signal max : {df['signal'].max():.4f}",
    ]
    return "\n".join(lines)


def signal_summary_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """Mean signal and count grouped by regime state (used in notebooks)."""
    labels = {1: "Bull", 0: "Sideways", -1: "Bear"}
    result = (
        df.groupby("state")["signal"]
        .agg(n_bars="count", mean="mean", std="std", min="min", max="max")
        .rename(index=labels)
    )
    result.index.name = "state"
    return result


def build_ticker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rich per-ticker summary: dates, row count, signal stats, forward return stats."""
    rows: list[dict[str, Any]] = []
    for ticker in sorted(df["ticker"].dropna().unique()):
        subset = df[df["ticker"] == ticker]
        row = {
            "ticker": ticker,
            "first_date": subset["date"].min(),
            "last_date": subset["date"].max(),
            "row_count": _valid_count(subset["signal"]),
            "valid_5d_rows": _valid_count(_numeric_series(subset, "future_5d_return")),
            "valid_10d_rows": _valid_count(_numeric_series(subset, "future_10d_return")),
            "valid_20d_rows": _valid_count(_numeric_series(subset, "future_20d_return")),
            "avg_signal": _mean_col(subset, "signal"),
            "min_signal": _numeric_series(subset, "signal").min(),
            "max_signal": _numeric_series(subset, "signal").max(),
            "avg_5d_return": _mean_col(subset, "future_5d_return"),
            "avg_10d_return": _mean_col(subset, "future_10d_return"),
            "avg_20d_return": _mean_col(subset, "future_20d_return"),
            "pos_10d_rate": positive_rate(_numeric_series(subset, "future_10d_return")),
            "pos_20d_rate": positive_rate(_numeric_series(subset, "future_20d_return")),
            **_excess_20d_stats(subset),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def build_signal_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows into 7 signal buckets and aggregate forward-return stats.
    All 7 buckets are always present; empty buckets show row_count=0.
    """
    bucketed = _assign_buckets(df)
    rows: list[dict[str, Any]] = []
    for bucket in _BUCKET_LABELS:
        subset = bucketed[bucketed["bucket"] == bucket]
        rows.append(
            {
                "bucket": bucket,
                "row_count": len(subset),
                "valid_5d_rows": _valid_count(_numeric_series(subset, "future_5d_return")),
                "valid_10d_rows": _valid_count(_numeric_series(subset, "future_10d_return")),
                "valid_20d_rows": _valid_count(_numeric_series(subset, "future_20d_return")),
                "avg_5d_return": _mean_col(subset, "future_5d_return"),
                "avg_10d_return": _mean_col(subset, "future_10d_return"),
                "median_10d_return": _median_col(subset, "future_10d_return"),
                "positive_10d_rate": positive_rate(_numeric_series(subset, "future_10d_return")),
                "avg_20d_return": _mean_col(subset, "future_20d_return"),
                "median_20d_return": _median_col(subset, "future_20d_return"),
                "positive_20d_rate": positive_rate(_numeric_series(subset, "future_20d_return")),
                "avg_sample_count": _mean_col(subset, "sample_count"),
                **_excess_20d_stats(subset),
            }
        )
    return pd.DataFrame(rows).set_index("bucket")


def build_filter_comparison_table(
    df: pd.DataFrame,
    filter_col: str,
    label_true: str = "Filter ON",
    label_false: str = "Filter OFF",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    For each signal threshold, compare forward-return stats across three conditions:
        All rows, filter=True, filter=False.

    Returns a DataFrame with MultiIndex (threshold_label, condition).
    """
    if thresholds is None:
        thresholds = _THRESHOLDS

    rows: list[dict] = []
    for thr in thresholds:
        label = f"{thr * 100:.0f}%"
        subset = df[df["signal"] >= thr]
        rows.append({"threshold": label, "condition": "All", **_signal_stats(subset)})

        if filter_col in df.columns:
            healthy = subset[subset[filter_col] == 1]
            weak = subset[subset[filter_col] == 0]
            rows.append({"threshold": label, "condition": label_true, **_signal_stats(healthy)})
            rows.append({"threshold": label, "condition": label_false, **_signal_stats(weak)})

    result = pd.DataFrame(rows)
    result.set_index(["threshold", "condition"], inplace=True)
    return result


def build_combined_conditions_table(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    For each signal threshold, compute stats for four filter combinations:
        1. Markov only
        2. Markov + Market Healthy (if available)
        3. Markov + Stock Trend Healthy (if available)
        4. Markov + Market + Stock (if both available)

    Returns a DataFrame with MultiIndex (threshold_label, condition).
    """
    if thresholds is None:
        thresholds = _THRESHOLDS

    has_market = "market_healthy_either" in df.columns
    has_stock = "stock_trend_healthy" in df.columns

    rows: list[dict] = []
    for thr in thresholds:
        label = f"{thr * 100:.0f}%"
        base = df[df["signal"] >= thr]

        rows.append({"threshold": label, "condition": "Markov only", **_signal_stats(base)})

        if has_market:
            mkt = base[base["market_healthy_either"] == 1]
            rows.append({"threshold": label, "condition": "Markov + Market", **_signal_stats(mkt)})

        if has_stock:
            trend = base[base["stock_trend_healthy"] == 1]
            rows.append({"threshold": label, "condition": "Markov + Trend", **_signal_stats(trend)})

        if has_market and has_stock:
            both = base[
                (base["market_healthy_either"] == 1) & (base["stock_trend_healthy"] == 1)
            ]
            rows.append({"threshold": label, "condition": "Markov + Market + Trend", **_signal_stats(both)})

    result = pd.DataFrame(rows)
    result.set_index(["threshold", "condition"], inplace=True)
    return result


def build_per_ticker_signal_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    """Signal-bucket forward-return stats for every ticker and every bucket."""
    if not _has_columns(df, ["ticker", "signal"]):
        return pd.DataFrame()

    bucketed = _assign_buckets(df)
    rows: list[dict[str, Any]] = []
    for ticker in sorted(bucketed["ticker"].dropna().unique()):
        ticker_df = bucketed[bucketed["ticker"] == ticker]
        for bucket in _BUCKET_LABELS:
            subset = ticker_df[ticker_df["bucket"] == bucket]
            row = {
                "ticker": ticker,
                "signal_bucket": bucket,
                **_return_stats(subset, [5, 10, 20]),
            }
            row["avg_sample_count"] = (
                float(subset["sample_count"].mean())
                if "sample_count" in subset.columns and not subset.empty
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _ticker_verdict(
    high_avg: float,
    low_avg: float,
    high_valid: int,
    low_valid: int,
) -> str:
    """Heuristic research label for per-ticker signal behavior."""
    if (
        high_valid < _MIN_VERDICT_VALID_20D
        or low_valid < _MIN_VERDICT_VALID_20D
        or pd.isna(high_avg)
        or pd.isna(low_avg)
    ):
        return "Weak/noisy"

    spread = high_avg - low_avg
    if low_avg > high_avg:
        return "Mean reversion / inverted"
    if high_avg > 0 and (low_avg < 0 or spread >= 0.02):
        return "Good directional"
    if high_avg > 0 and low_avg > 0:
        return "Momentum only"
    return "Weak/noisy"


def build_ticker_signal_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """High-vs-low signal behavior by ticker with market-filtered high-signal stats."""
    if not _has_columns(df, ["ticker", "signal"]):
        return pd.DataFrame()

    has_market = "market_healthy_either" in df.columns
    rows: list[dict[str, Any]] = []

    for ticker in sorted(df["ticker"].dropna().unique()):
        ticker_df = df[df["ticker"] == ticker]
        high = ticker_df[ticker_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        low = ticker_df[ticker_df["signal"] <= _LOW_SIGNAL_THRESHOLD]
        mkt_high = (
            high[high["market_healthy_either"] == 1]
            if has_market
            else high.iloc[0:0]
        )

        high_stats = _return_stats(high, [20])
        low_stats = _return_stats(low, [20])
        mkt_stats = _return_stats(mkt_high, [20])

        high_avg = high_stats["avg_20d_return"]
        low_avg = low_stats["avg_20d_return"]
        spread = (
            high_avg - low_avg
            if not pd.isna(high_avg) and not pd.isna(low_avg)
            else np.nan
        )

        rows.append(
            {
                "ticker": ticker,
                "total_rows": len(ticker_df),
                "high_signal_rows": high_stats["rows"],
                "high_signal_valid_20d_rows": high_stats["valid_20d_rows"],
                "high_signal_avg_20d_return": high_avg,
                "high_signal_median_20d_return": high_stats["median_20d_return"],
                "high_signal_20d_win_rate": high_stats["20d_win_rate"],
                "high_signal_avg_excess_vs_spy_20d": high_stats["avg_excess_vs_spy_20d"],
                "high_signal_avg_excess_vs_qqq_20d": high_stats["avg_excess_vs_qqq_20d"],
                "high_signal_avg_excess_vs_sector_etf_20d": high_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "high_signal_median_excess_vs_sector_etf_20d": high_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "high_signal_excess_vs_sector_etf_20d_win_rate": high_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
                "low_signal_rows": low_stats["rows"],
                "low_signal_valid_20d_rows": low_stats["valid_20d_rows"],
                "low_signal_avg_20d_return": low_avg,
                "low_signal_median_20d_return": low_stats["median_20d_return"],
                "low_signal_20d_win_rate": low_stats["20d_win_rate"],
                "low_signal_avg_excess_vs_spy_20d": low_stats["avg_excess_vs_spy_20d"],
                "low_signal_avg_excess_vs_qqq_20d": low_stats["avg_excess_vs_qqq_20d"],
                "low_signal_avg_excess_vs_sector_etf_20d": low_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "low_signal_median_excess_vs_sector_etf_20d": low_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "low_signal_excess_vs_sector_etf_20d_win_rate": low_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
                "directional_spread": spread,
                "market_high_signal_rows": mkt_stats["rows"] if has_market else np.nan,
                "market_high_signal_valid_20d_rows": (
                    mkt_stats["valid_20d_rows"] if has_market else np.nan
                ),
                "market_high_signal_avg_20d_return": (
                    mkt_stats["avg_20d_return"] if has_market else np.nan
                ),
                "market_high_signal_median_20d_return": (
                    mkt_stats["median_20d_return"] if has_market else np.nan
                ),
                "market_high_signal_20d_win_rate": (
                    mkt_stats["20d_win_rate"] if has_market else np.nan
                ),
                "market_high_signal_avg_excess_vs_sector_etf_20d": (
                    mkt_stats["avg_excess_vs_sector_etf_20d"] if has_market else np.nan
                ),
                "verdict": _ticker_verdict(
                    high_avg,
                    low_avg,
                    high_stats["valid_20d_rows"],
                    low_stats["valid_20d_rows"],
                ),
            }
        )

    return pd.DataFrame(rows)


def build_market_filtered_ticker_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare high-signal ticker stats with and without the market regime filter."""
    if not _has_columns(df, ["ticker", "signal", "market_healthy_either"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for ticker in sorted(df["ticker"].dropna().unique()):
        ticker_df = df[df["ticker"] == ticker]
        high = ticker_df[ticker_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        conditions = [
            ("High signal only", high),
            ("High signal + market healthy", high[high["market_healthy_either"] == 1]),
            ("High signal + market weak", high[high["market_healthy_either"] == 0]),
        ]
        for condition, subset in conditions:
            rows.append(
                {
                    "ticker": ticker,
                    "condition": condition,
                    **_return_stats(subset, [10, 20]),
                }
            )
    return pd.DataFrame(rows)


def build_group_signal_bucket_table(
    df: pd.DataFrame,
    ticker_groups: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """Signal-bucket stats for manually configured ticker groups."""
    groups = _normalize_ticker_groups(ticker_groups, df)
    if not groups or not _has_columns(df, ["ticker", "signal"]):
        return pd.DataFrame()

    bucketed = _assign_buckets(df)
    rows: list[dict[str, Any]] = []
    for group, tickers in groups.items():
        group_df = bucketed[bucketed["ticker"].isin(tickers)]
        for bucket in _BUCKET_LABELS:
            subset = group_df[group_df["bucket"] == bucket]
            rows.append(
                {
                    "group": group,
                    "signal_bucket": bucket,
                    **_return_stats(subset, [10, 20]),
                }
            )
    return pd.DataFrame(rows)


def build_group_high_signal_summary(
    df: pd.DataFrame,
    ticker_groups: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """High-signal 20D stats by configured ticker group."""
    groups = _normalize_ticker_groups(ticker_groups, df)
    if not groups or not _has_columns(df, ["ticker", "signal"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group, tickers in groups.items():
        subset = df[
            (df["ticker"].isin(tickers))
            & (df["signal"] >= _HIGH_SIGNAL_THRESHOLD)
        ]
        stats = _return_stats(subset, [20])
        rows.append(
            {
                "group": group,
                "high_signal_rows": stats["rows"],
                "high_signal_valid_20d_rows": stats["valid_20d_rows"],
                "high_signal_avg_20d_return": stats["avg_20d_return"],
                "high_signal_median_20d_return": stats["median_20d_return"],
                "high_signal_20d_win_rate": stats["20d_win_rate"],
            }
        )
    return pd.DataFrame(rows)


def build_sector_signal_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-level signal-bucket stats using the same buckets as the global report."""
    if not _has_columns(df, ["sector", "signal"]):
        return pd.DataFrame()

    bucketed = _assign_buckets(df)
    rows: list[dict[str, Any]] = []
    for sector in sorted(bucketed["sector"].dropna().unique()):
        sector_df = bucketed[bucketed["sector"] == sector]
        for bucket in _BUCKET_LABELS:
            subset = sector_df[sector_df["bucket"] == bucket]
            rows.append(
                {
                    "sector": sector,
                    "signal_bucket": bucket,
                    **_return_stats(subset, [10, 20]),
                }
            )
    return pd.DataFrame(rows)


def build_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """High-signal summary by sector."""
    if not _has_columns(df, ["sector", "signal"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        high = sector_df[sector_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        high_stats = _return_stats(high, [10, 20])
        rows.append(
            {
                "sector": sector,
                "rows": len(sector_df),
                "valid_20d_rows": _valid_count(sector_df.get("future_20d_return", pd.Series(dtype=float))),
                "avg_signal": float(sector_df["signal"].mean()) if not sector_df.empty else np.nan,
                "high_signal_rows": high_stats["rows"],
                "high_signal_valid_10d_rows": high_stats["valid_10d_rows"],
                "high_signal_avg_10d_return": high_stats["avg_10d_return"],
                "high_signal_median_10d_return": high_stats["median_10d_return"],
                "high_signal_10d_win_rate": high_stats["10d_win_rate"],
                "high_signal_valid_20d_rows": high_stats["valid_20d_rows"],
                "high_signal_avg_20d_return": high_stats["avg_20d_return"],
                "high_signal_median_20d_return": high_stats["median_20d_return"],
                "high_signal_20d_win_rate": high_stats["20d_win_rate"],
                "high_signal_avg_excess_vs_spy_20d": high_stats["avg_excess_vs_spy_20d"],
                "high_signal_avg_excess_vs_qqq_20d": high_stats["avg_excess_vs_qqq_20d"],
                "high_signal_avg_excess_vs_sector_etf_20d": high_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "high_signal_median_excess_vs_sector_etf_20d": high_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "high_signal_excess_vs_sector_etf_20d_win_rate": high_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_sector_market_regime_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare high-signal sector performance across market-regime conditions."""
    if not _has_columns(df, ["sector", "signal", "market_healthy_either"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        high = sector_df[sector_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        conditions = [
            ("High signal only", high),
            ("High signal + market healthy", high[high["market_healthy_either"] == 1]),
            ("High signal + market weak", high[high["market_healthy_either"] == 0]),
        ]
        for condition, subset in conditions:
            rows.append(
                {
                    "sector": sector,
                    "condition": condition,
                    **_return_stats(subset, [10, 20]),
                }
            )
    return pd.DataFrame(rows)


def build_sector_etf_vs_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Compare sector ETF rows against stock rows for each market condition."""
    required = ["sector", "signal", "instrument_type", "market_healthy_either"]
    if not _has_columns(df, required):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        high = sector_df[sector_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        conditions = [
            ("High signal only", high),
            ("High signal + market healthy", high[high["market_healthy_either"] == 1]),
            ("High signal + market weak", high[high["market_healthy_either"] == 0]),
        ]
        for condition, condition_df in conditions:
            for instrument_type, label in [
                ("sector_etf", "Sector ETF"),
                ("stock", "Stocks"),
            ]:
                subset = condition_df[condition_df["instrument_type"] == instrument_type]
                rows.append(
                    {
                        "sector": sector,
                        "condition": condition,
                        "instrument": label,
                        **_return_stats(subset, [20]),
                    }
                )
    return pd.DataFrame(rows)


def build_broad_group_regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Broad defensive/growth/cyclical/rate-sensitive regime analysis."""
    if not _has_columns(df, ["sector", "signal", "market_healthy_either"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group, sectors in _BROAD_SECTOR_GROUPS.items():
        group_df = df[df["sector"].isin(sectors)]
        high = group_df[group_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        conditions = [
            ("High signal only", high),
            ("Market healthy", high[high["market_healthy_either"] == 1]),
            ("Market weak", high[high["market_healthy_either"] == 0]),
        ]
        for condition, subset in conditions:
            rows.append(
                {
                    "broad_group": group,
                    "condition": condition,
                    **_return_stats(subset, [20]),
                }
            )
    return pd.DataFrame(rows)


def _edge_20d_stats(
    subset: pd.DataFrame,
    universe_avg_20d: float | None = None,
) -> dict[str, Any]:
    """Raw and baseline-adjusted 20D stats for edge analysis."""
    raw_avg = _mean_col(subset, "future_20d_return")
    universe_baseline = _mean_col(subset, "universe_avg_future_20d_return")
    if pd.isna(universe_baseline) and universe_avg_20d is not None:
        universe_baseline = universe_avg_20d

    excess_universe = _mean_col(subset, "excess_vs_universe_avg_20d")
    if pd.isna(excess_universe) and not pd.isna(raw_avg) and not pd.isna(universe_baseline):
        excess_universe = raw_avg - universe_baseline

    return {
        "rows": len(subset),
        "valid_20d_rows": _valid_count(_numeric_series(subset, "future_20d_return")),
        "raw_avg_20d_return": raw_avg,
        "raw_median_20d_return": _median_col(subset, "future_20d_return"),
        "raw_20d_win_rate": positive_rate(_numeric_series(subset, "future_20d_return")),
        "universe_baseline_avg_20d_return": universe_baseline,
        "excess_vs_universe_avg_20d": excess_universe,
        "avg_excess_vs_spy_20d": _mean_col(subset, "excess_vs_spy_20d"),
        "avg_excess_vs_qqq_20d": _mean_col(subset, "excess_vs_qqq_20d"),
        "avg_excess_vs_sector_etf_20d": _mean_col(subset, "excess_vs_sector_etf_20d"),
        "median_excess_vs_sector_etf_20d": _median_col(subset, "excess_vs_sector_etf_20d"),
        "valid_excess_vs_sector_etf_20d_rows": _valid_count(
            _numeric_series(subset, "excess_vs_sector_etf_20d")
        ),
        "excess_vs_sector_etf_20d_win_rate": positive_rate(
            _numeric_series(subset, "excess_vs_sector_etf_20d")
        ),
    }


def build_global_signal_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Global signal-bucket edge table versus universe, SPY, QQQ, and sector ETF."""
    if not _has_columns(df, ["signal"]):
        return pd.DataFrame()

    bucketed = _assign_buckets(df)
    universe_avg_20d = _mean_col(df, "future_20d_return")
    rows: list[dict[str, Any]] = []
    for bucket in _BUCKET_LABELS:
        subset = bucketed[bucketed["bucket"] == bucket]
        rows.append(
            {
                "signal_bucket": bucket,
                **_edge_20d_stats(subset, universe_avg_20d),
            }
        )
    return pd.DataFrame(rows)


def build_high_vs_low_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare high, low, and neutral signal regimes against baselines."""
    if not _has_columns(df, ["signal"]):
        return pd.DataFrame()

    universe_avg_20d = _mean_col(df, "future_20d_return")
    conditions = [
        ("High signal >= 40%", df[df["signal"] >= _HIGH_SIGNAL_THRESHOLD]),
        ("Low signal <= -20%", df[df["signal"] <= _LOW_SIGNAL_THRESHOLD]),
        ("Neutral -10% to +10%", df[df["signal"].between(-0.10, 0.10, inclusive="both")]),
    ]
    rows = [
        {"condition": label, **_edge_20d_stats(subset, universe_avg_20d)}
        for label, subset in conditions
    ]
    return pd.DataFrame(rows)


def build_sector_signal_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-level excess-return table by high/low and market-regime conditions."""
    if not _has_columns(df, ["sector", "signal"]):
        return pd.DataFrame()

    has_market = "market_healthy_either" in df.columns
    universe_avg_20d = _mean_col(df, "future_20d_return")
    rows: list[dict[str, Any]] = []
    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        high = sector_df[sector_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        low = sector_df[sector_df["signal"] <= _LOW_SIGNAL_THRESHOLD]
        conditions: list[tuple[str, pd.DataFrame]] = [
            ("High signal >= 40%", high),
            ("Low signal <= -20%", low),
        ]
        if has_market:
            conditions.extend(
                [
                    ("High signal + market healthy", high[high["market_healthy_either"] == 1]),
                    ("High signal + market weak", high[high["market_healthy_either"] == 0]),
                ]
            )
        for condition, subset in conditions:
            rows.append(
                {
                    "sector": sector,
                    "condition": condition,
                    **_edge_20d_stats(subset, universe_avg_20d),
                }
            )
    return pd.DataFrame(rows)


def build_ticker_signal_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Ticker-level high/low excess-return rankings versus each sector ETF."""
    if not _has_columns(df, ["ticker", "signal"]):
        return pd.DataFrame()

    universe_avg_20d = _mean_col(df, "future_20d_return")
    rows: list[dict[str, Any]] = []
    for ticker in sorted(df["ticker"].dropna().unique()):
        ticker_df = df[df["ticker"] == ticker]
        high = ticker_df[ticker_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        low = ticker_df[ticker_df["signal"] <= _LOW_SIGNAL_THRESHOLD]
        high_stats = _edge_20d_stats(high, universe_avg_20d)
        low_stats = _edge_20d_stats(low, universe_avg_20d)
        rows.append(
            {
                "ticker": ticker,
                "sector": ticker_df["sector"].iloc[0] if "sector" in ticker_df.columns and not ticker_df.empty else "",
                "instrument_type": (
                    ticker_df["instrument_type"].iloc[0]
                    if "instrument_type" in ticker_df.columns and not ticker_df.empty
                    else ""
                ),
                "sector_etf": (
                    ticker_df["sector_etf"].iloc[0]
                    if "sector_etf" in ticker_df.columns and not ticker_df.empty
                    else ""
                ),
                "total_rows": len(ticker_df),
                "high_signal_rows": high_stats["rows"],
                "high_signal_valid_20d_rows": high_stats["valid_20d_rows"],
                "high_signal_raw_avg_20d_return": high_stats["raw_avg_20d_return"],
                "high_signal_avg_excess_vs_sector_etf_20d": high_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "high_signal_median_excess_vs_sector_etf_20d": high_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "high_signal_excess_vs_sector_etf_20d_win_rate": high_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
                "low_signal_rows": low_stats["rows"],
                "low_signal_valid_20d_rows": low_stats["valid_20d_rows"],
                "low_signal_raw_avg_20d_return": low_stats["raw_avg_20d_return"],
                "low_signal_avg_excess_vs_sector_etf_20d": low_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "low_signal_median_excess_vs_sector_etf_20d": low_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "low_signal_excess_vs_sector_etf_20d_win_rate": low_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_market_regime_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Sector high-signal excess stats split by SPY/QQQ market regime."""
    if not _has_columns(df, ["sector", "signal", "market_healthy_either"]):
        return pd.DataFrame()

    universe_avg_20d = _mean_col(df, "future_20d_return")
    rows: list[dict[str, Any]] = []
    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        high = sector_df[sector_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        for condition, subset in [
            ("High signal only", high),
            ("High signal + market healthy", high[high["market_healthy_either"] == 1]),
            ("High signal + market weak", high[high["market_healthy_either"] == 0]),
        ]:
            rows.append(
                {
                    "sector": sector,
                    "condition": condition,
                    **_edge_20d_stats(subset, universe_avg_20d),
                }
            )
    return pd.DataFrame(rows)


def _personality_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """Use stock rows for stock-personality analysis; ETFs self-baseline by design."""
    if "instrument_type" not in df.columns:
        return df
    return df[df["instrument_type"] != "sector_etf"].copy()


def _ordered_labels(series: pd.Series, preferred: list[str]) -> list[str]:
    available = [str(x) for x in series.dropna().unique()]
    labels = [label for label in preferred if label in available]
    labels.extend(sorted(label for label in available if label not in labels))
    return labels


def _signal_condition_subsets(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "signal" not in df.columns:
        empty = df.iloc[0:0]
        return [(label, empty) for label in _SIGNAL_CONDITIONS]
    return [
        ("High signal >= 40%", df[df["signal"] >= _HIGH_SIGNAL_THRESHOLD]),
        ("Low signal <= -20%", df[df["signal"] <= _LOW_SIGNAL_THRESHOLD]),
        ("Neutral -10% to +10%", df[df["signal"].between(-0.10, 0.10, inclusive="both")]),
    ]


def build_personality_signal_excess_summary(
    df: pd.DataFrame,
    label_col: str = "stock_personality",
) -> pd.DataFrame:
    """Signal-condition excess stats by stock personality label."""
    if not _has_columns(df, [label_col, "signal"]):
        return pd.DataFrame()

    base = _personality_base_df(df)
    preferred = _PERSONALITY_GROUPS if label_col == "stock_personality_group" else _PERSONALITY_LABELS
    labels = _ordered_labels(base[label_col], preferred)
    universe_avg_20d = _mean_col(base, "future_20d_return")
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_df = base[base[label_col] == label]
        for condition, subset in _signal_condition_subsets(label_df):
            rows.append(
                {
                    label_col: label,
                    "condition": condition,
                    **_edge_20d_stats(subset, universe_avg_20d),
                }
            )
    return pd.DataFrame(rows)


def _personality_behavior_label(high_avg: float, low_avg: float) -> str:
    if pd.isna(high_avg) or pd.isna(low_avg):
        return "No clear edge"
    clear_margin = 0.01
    if high_avg > low_avg + clear_margin:
        return "High-signal continuation"
    if low_avg > high_avg + clear_margin:
        return "Low-signal rebound"
    if high_avg > 0 and low_avg > 0:
        return "Both extreme states work"
    return "No clear edge"


def build_personality_high_vs_low_summary(
    df: pd.DataFrame,
    label_col: str = "stock_personality",
) -> pd.DataFrame:
    """High-vs-low signal behavior by personality label."""
    if not _has_columns(df, [label_col, "signal"]):
        return pd.DataFrame()

    base = _personality_base_df(df)
    preferred = _PERSONALITY_GROUPS if label_col == "stock_personality_group" else _PERSONALITY_LABELS
    labels = _ordered_labels(base[label_col], preferred)
    universe_avg_20d = _mean_col(base, "future_20d_return")
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_df = base[base[label_col] == label]
        high = label_df[label_df["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        low = label_df[label_df["signal"] <= _LOW_SIGNAL_THRESHOLD]
        high_stats = _edge_20d_stats(high, universe_avg_20d)
        low_stats = _edge_20d_stats(low, universe_avg_20d)
        high_avg = high_stats["avg_excess_vs_sector_etf_20d"]
        low_avg = low_stats["avg_excess_vs_sector_etf_20d"]
        rows.append(
            {
                label_col: label,
                "rows": len(label_df),
                "high_signal_rows": high_stats["rows"],
                "high_signal_valid_20d_rows": high_stats["valid_20d_rows"],
                "high_signal_avg_excess_vs_sector_etf_20d": high_avg,
                "high_signal_median_excess_vs_sector_etf_20d": high_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "high_signal_excess_vs_sector_etf_20d_win_rate": high_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
                "low_signal_rows": low_stats["rows"],
                "low_signal_valid_20d_rows": low_stats["valid_20d_rows"],
                "low_signal_avg_excess_vs_sector_etf_20d": low_avg,
                "low_signal_median_excess_vs_sector_etf_20d": low_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "low_signal_excess_vs_sector_etf_20d_win_rate": low_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
                "high_minus_low_excess_vs_sector_etf_20d": (
                    high_avg - low_avg
                    if not pd.isna(high_avg) and not pd.isna(low_avg)
                    else np.nan
                ),
                "behavior_label": _personality_behavior_label(high_avg, low_avg),
            }
        )
    return pd.DataFrame(rows)


def build_personality_bucket_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Personality x signal-bucket average excess vs sector ETF."""
    if not _has_columns(df, ["stock_personality", "signal"]):
        return pd.DataFrame()

    base = _assign_buckets(_personality_base_df(df))
    labels = _ordered_labels(base["stock_personality"], _PERSONALITY_LABELS)
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_df = base[base["stock_personality"] == label]
        for bucket in _BUCKET_LABELS:
            subset = label_df[label_df["bucket"] == bucket]
            rows.append(
                {
                    "stock_personality": label,
                    "signal_bucket": bucket,
                    "rows": len(subset),
                    "valid_20d_rows": _valid_count(_numeric_series(subset, "future_20d_return")),
                    "raw_avg_20d_return": _mean_col(subset, "future_20d_return"),
                    "avg_excess_vs_sector_etf_20d": _mean_col(
                        subset, "excess_vs_sector_etf_20d"
                    ),
                    "median_excess_vs_sector_etf_20d": _median_col(
                        subset, "excess_vs_sector_etf_20d"
                    ),
                    "excess_vs_sector_etf_20d_win_rate": positive_rate(
                        _numeric_series(subset, "excess_vs_sector_etf_20d")
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_ticker_personality_excess_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Ticker/personality high-vs-low excess summary."""
    if not _has_columns(df, ["ticker", "stock_personality", "signal"]):
        return pd.DataFrame()

    base = _personality_base_df(df)
    universe_avg_20d = _mean_col(base, "future_20d_return")
    rows: list[dict[str, Any]] = []
    grouped = base.groupby(["ticker", "stock_personality"], dropna=False)
    for (ticker, personality), subset in grouped:
        if pd.isna(ticker) or pd.isna(personality):
            continue
        high = subset[subset["signal"] >= _HIGH_SIGNAL_THRESHOLD]
        low = subset[subset["signal"] <= _LOW_SIGNAL_THRESHOLD]
        high_stats = _edge_20d_stats(high, universe_avg_20d)
        low_stats = _edge_20d_stats(low, universe_avg_20d)
        rows.append(
            {
                "ticker": ticker,
                "stock_personality": personality,
                "stock_personality_group": (
                    subset["stock_personality_group"].iloc[0]
                    if "stock_personality_group" in subset.columns and not subset.empty
                    else ""
                ),
                "sector": subset["sector"].iloc[0] if "sector" in subset.columns and not subset.empty else "",
                "instrument_type": (
                    subset["instrument_type"].iloc[0]
                    if "instrument_type" in subset.columns and not subset.empty
                    else ""
                ),
                "sector_etf": (
                    subset["sector_etf"].iloc[0]
                    if "sector_etf" in subset.columns and not subset.empty
                    else ""
                ),
                "rows": len(subset),
                "high_signal_rows": high_stats["rows"],
                "high_signal_valid_20d_rows": high_stats["valid_20d_rows"],
                "high_signal_raw_avg_20d_return": high_stats["raw_avg_20d_return"],
                "high_signal_avg_excess_vs_sector_etf_20d": high_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "high_signal_median_excess_vs_sector_etf_20d": high_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "high_signal_excess_vs_sector_etf_20d_win_rate": high_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
                "low_signal_rows": low_stats["rows"],
                "low_signal_valid_20d_rows": low_stats["valid_20d_rows"],
                "low_signal_raw_avg_20d_return": low_stats["raw_avg_20d_return"],
                "low_signal_avg_excess_vs_sector_etf_20d": low_stats[
                    "avg_excess_vs_sector_etf_20d"
                ],
                "low_signal_median_excess_vs_sector_etf_20d": low_stats[
                    "median_excess_vs_sector_etf_20d"
                ],
                "low_signal_excess_vs_sector_etf_20d_win_rate": low_stats[
                    "excess_vs_sector_etf_20d_win_rate"
                ],
            }
        )
    return pd.DataFrame(rows)


def _path_hit_rate(series: pd.Series) -> float:
    """Average 0/1 path-hit labels, dropping rows without a full path."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.mean())


def _path_stats(subset: pd.DataFrame) -> dict[str, Any]:
    """Forward path/risk stats for a row subset."""
    stats: dict[str, Any] = {
        "rows": len(subset),
        "valid_20d_rows": _valid_count(_numeric_series(subset, "future_20d_return")),
        "valid_path_20d_rows": _valid_count(
            _numeric_series(subset, "max_favorable_excursion_20d")
        ),
        "avg_20d_close_return": _mean_col(subset, "future_20d_return"),
        "avg_20d_mfe": _mean_col(subset, "max_favorable_excursion_20d"),
        "avg_20d_mae": _mean_col(subset, "max_adverse_excursion_20d"),
        "median_20d_mae": _median_col(subset, "max_adverse_excursion_20d"),
    }
    for source_col, output_col in _PATH_HIT_COLUMNS:
        stats[output_col] = _path_hit_rate(_numeric_series(subset, source_col))
    return stats


def build_path_by_personality_signal(
    df: pd.DataFrame,
    label_col: str = "stock_personality",
) -> pd.DataFrame:
    """Forward path stats by stock personality and signal condition."""
    required = [label_col, "signal", "max_favorable_excursion_20d"]
    if not _has_columns(df, required):
        return pd.DataFrame()

    base = _personality_base_df(df)
    preferred = _PERSONALITY_GROUPS if label_col == "stock_personality_group" else _PERSONALITY_LABELS
    labels = _ordered_labels(base[label_col], preferred)
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_df = base[base[label_col] == label]
        for condition, subset in _signal_condition_subsets(label_df):
            rows.append(
                {
                    label_col: label,
                    "condition": condition,
                    **_path_stats(subset),
                }
            )
    return pd.DataFrame(rows)


def build_path_by_sector_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Forward path stats by sector and signal condition."""
    if not _has_columns(df, ["sector", "signal", "max_favorable_excursion_20d"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sector in sorted(df["sector"].dropna().unique()):
        sector_df = df[df["sector"] == sector]
        for condition, subset in _signal_condition_subsets(sector_df):
            rows.append(
                {
                    "sector": sector,
                    "condition": condition,
                    **_path_stats(subset),
                }
            )
    return pd.DataFrame(rows)


def build_path_by_ticker_personality(df: pd.DataFrame) -> pd.DataFrame:
    """Forward path stats by ticker, personality, and signal condition."""
    required = ["ticker", "stock_personality", "signal", "max_favorable_excursion_20d"]
    if not _has_columns(df, required):
        return pd.DataFrame()

    base = _personality_base_df(df)
    rows: list[dict[str, Any]] = []
    grouped = base.groupby(["ticker", "stock_personality"], dropna=False)
    for (ticker, personality), subset in grouped:
        if pd.isna(ticker) or pd.isna(personality):
            continue
        for condition, condition_df in _signal_condition_subsets(subset):
            rows.append(
                {
                    "ticker": ticker,
                    "stock_personality": personality,
                    "stock_personality_group": (
                        subset["stock_personality_group"].iloc[0]
                        if "stock_personality_group" in subset.columns and not subset.empty
                        else ""
                    ),
                    "sector": (
                        subset["sector"].iloc[0]
                        if "sector" in subset.columns and not subset.empty
                        else ""
                    ),
                    "sector_etf": (
                        subset["sector_etf"].iloc[0]
                        if "sector_etf" in subset.columns and not subset.empty
                        else ""
                    ),
                    "condition": condition,
                    **_path_stats(condition_df),
                }
            )
    return pd.DataFrame(rows)


def build_path_focus_summary(path_personality_df: pd.DataFrame) -> pd.DataFrame:
    """Key forward-path cases called out in the report."""
    if path_personality_df.empty:
        return pd.DataFrame()

    focus_cases = [
        ("deep_drawdown + high signal", "deep_drawdown", "High signal >= 40%"),
        ("deep_drawdown + low signal", "deep_drawdown", "Low signal <= -20%"),
        ("secular_winner + high signal", "secular_winner", "High signal >= 40%"),
        ("laggard + low signal", "laggard", "Low signal <= -20%"),
    ]
    rows: list[pd.Series] = []
    for label, personality, condition in focus_cases:
        match = path_personality_df[
            (path_personality_df["stock_personality"] == personality)
            & (path_personality_df["condition"] == condition)
        ]
        if match.empty:
            row = pd.Series(
                {
                    "focus_case": label,
                    "stock_personality": personality,
                    "condition": condition,
                    **_path_stats(pd.DataFrame()),
                }
            )
        else:
            row = match.iloc[0].copy()
            row["focus_case"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def _boolean_mask(series: pd.Series) -> pd.Series:
    """Robust boolean mask for parquet or CSV-loaded signal labels."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) != 0
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _signal_lab_stats(subset: pd.DataFrame) -> dict[str, Any]:
    """Return, excess, and path stats for RSI/BB vs Markov comparisons."""
    stats: dict[str, Any] = {"rows": len(subset)}
    for d in [5, 10, 20]:
        col = f"future_{d}d_return"
        valid = _numeric_series(subset, col).dropna()
        stats[f"valid_{d}d_rows"] = len(valid)
        stats[f"avg_{d}d_return"] = float(valid.mean()) if len(valid) else np.nan
        stats[f"median_{d}d_return"] = float(valid.median()) if len(valid) else np.nan
        stats[f"{d}d_win_rate"] = positive_rate(_numeric_series(subset, col))

    sector_excess = _numeric_series(subset, "excess_vs_sector_etf_20d")
    stats.update(
        {
            "avg_excess_vs_sector_etf_20d": _mean_col(
                subset, "excess_vs_sector_etf_20d"
            ),
            "median_excess_vs_sector_etf_20d": _median_col(
                subset, "excess_vs_sector_etf_20d"
            ),
            "valid_excess_vs_sector_etf_20d_rows": _valid_count(sector_excess),
            "excess_vs_sector_etf_20d_win_rate": positive_rate(sector_excess),
            "avg_mfe_20d": _mean_col(subset, "max_favorable_excursion_20d"),
            "median_mfe_20d": _median_col(subset, "max_favorable_excursion_20d"),
            "avg_mae_20d": _mean_col(subset, "max_adverse_excursion_20d"),
            "median_mae_20d": _median_col(subset, "max_adverse_excursion_20d"),
        }
    )
    for source_col, output_col in _SIGNAL_LAB_HIT_COLUMNS:
        stats[output_col] = _path_hit_rate(_numeric_series(subset, source_col))
    return stats


def build_signal_lab_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Markov-only, RSI/BB-only, and combined signal labels."""
    rows: list[dict[str, Any]] = []
    for label, col in _SIGNAL_LAB_CONDITIONS:
        if col not in df.columns:
            continue
        subset = df[_boolean_mask(df[col])]
        rows.append(
            {
                "condition": label,
                "signal_column": col,
                **_signal_lab_stats(subset),
            }
        )
    return pd.DataFrame(rows)


def _rsi_bb_base(df: pd.DataFrame) -> pd.DataFrame:
    if "signal_rsi_bb" not in df.columns:
        return df.iloc[0:0]
    return df[_boolean_mask(df["signal_rsi_bb"])].copy()


def build_rsi_bb_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/BB oversold outcomes grouped by sector."""
    if "sector" not in df.columns:
        return pd.DataFrame()

    base = _rsi_bb_base(df)
    rows: list[dict[str, Any]] = []
    for sector in sorted(base["sector"].dropna().unique()):
        subset = base[base["sector"] == sector]
        rows.append({"sector": sector, **_signal_lab_stats(subset)})
    return pd.DataFrame(rows)


def build_rsi_bb_by_personality(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/BB oversold outcomes by personality and broad personality group."""
    base = _rsi_bb_base(df)
    if base.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group_col, group_type, preferred in [
        ("stock_personality", "stock_personality", _PERSONALITY_LABELS),
        ("stock_personality_group", "stock_personality_group", _PERSONALITY_GROUPS),
    ]:
        if group_col not in base.columns:
            continue
        for value in _ordered_labels(base[group_col], preferred):
            subset = base[base[group_col] == value]
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": value,
                    **_signal_lab_stats(subset),
                }
            )
    return pd.DataFrame(rows)


def build_rsi_bb_market_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/BB oversold outcomes by SPY/QQQ and sector ETF trend regimes."""
    base = _rsi_bb_base(df)
    if base.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    if "market_healthy_either" in base.columns:
        for value, label in [(1, "market healthy"), (0, "market weak")]:
            subset = base[base["market_healthy_either"] == value]
            rows.append(
                {
                    "breakdown": "market_healthy_either",
                    "bucket": label,
                    **_signal_lab_stats(subset),
                }
            )
    if "sector_etf_trend_healthy" in base.columns:
        for value, label in [(1, "sector ETF trend healthy"), (0, "sector ETF trend weak")]:
            subset = base[base["sector_etf_trend_healthy"] == value]
            rows.append(
                {
                    "breakdown": "sector_etf_trend_healthy",
                    "bucket": label,
                    **_signal_lab_stats(subset),
                }
            )
    return pd.DataFrame(rows)


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "avg_trade_return": np.nan,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "avg_bars_held": np.nan,
            "median_bars_held": np.nan,
            "avg_mae_before_exit": np.nan,
            "median_mae_before_exit": np.nan,
            "avg_mfe_before_exit": np.nan,
            "median_mfe_before_exit": np.nan,
        }

    t = pd.DataFrame(trades)
    returns = pd.to_numeric(t["trade_return"], errors="coerce")
    bars = pd.to_numeric(t["bars_held"], errors="coerce")
    mae = pd.to_numeric(t["mae_before_exit"], errors="coerce")
    mfe = pd.to_numeric(t["mfe_before_exit"], errors="coerce")
    return {
        "trade_count": len(t),
        "avg_trade_return": float(returns.mean()),
        "median_trade_return": float(returns.median()),
        "win_rate": positive_rate(returns),
        "avg_bars_held": float(bars.mean()),
        "median_bars_held": float(bars.median()),
        "avg_mae_before_exit": float(mae.mean()),
        "median_mae_before_exit": float(mae.median()),
        "avg_mfe_before_exit": float(mfe.mean()),
        "median_mfe_before_exit": float(mfe.median()),
    }


def _path_before_exit(
    frame: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
) -> tuple[float, float]:
    path = frame.iloc[entry_idx + 1 : exit_idx + 1]
    if path.empty:
        return np.nan, np.nan
    high_col = "high" if "high" in path.columns else "close"
    low_col = "low" if "low" in path.columns else "close"
    mfe = pd.to_numeric(path[high_col], errors="coerce").max() / entry_price - 1.0
    mae = pd.to_numeric(path[low_col], errors="coerce").min() / entry_price - 1.0
    return float(mfe), float(mae)


def build_rsi_bb_exit_rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Simple independent RSI/BB exit-rule simulations, one row per signal."""
    required = ["ticker", "date", "close", "rsi_14", "bb_basis_20", "signal_rsi_bb"]
    if not _has_columns(df, required):
        return pd.DataFrame()

    rule_trades: dict[str, list[dict[str, Any]]] = {
        "RSI >= 50": [],
        "RSI >= 70": [],
        "Close >= BB basis": [],
        "Exit after 20 bars": [],
        "TP +8% / SL -5%": [],
    }
    for _, ticker_df in df.groupby("ticker", dropna=False):
        frame = ticker_df.sort_values("date").reset_index(drop=True)
        signals = _boolean_mask(frame["signal_rsi_bb"])
        for entry_idx in list(np.flatnonzero(signals.to_numpy())):
            if entry_idx + 20 >= len(frame):
                continue
            entry = float(frame.loc[entry_idx, "close"])
            if pd.isna(entry) or entry <= 0:
                continue

            exit_specs = [
                ("RSI >= 50", lambda row: row["rsi_14"] >= 50),
                ("RSI >= 70", lambda row: row["rsi_14"] >= 70),
                ("Close >= BB basis", lambda row: row["close"] >= row["bb_basis_20"]),
            ]
            for rule_name, predicate in exit_specs:
                exit_idx = entry_idx + 20
                for j in range(entry_idx + 1, entry_idx + 21):
                    row = frame.loc[j]
                    if predicate(row):
                        exit_idx = j
                        break
                exit_price = float(frame.loc[exit_idx, "close"])
                mfe, mae = _path_before_exit(frame, entry_idx, exit_idx, entry)
                rule_trades[rule_name].append(
                    {
                        "trade_return": exit_price / entry - 1.0,
                        "bars_held": exit_idx - entry_idx,
                        "mfe_before_exit": mfe,
                        "mae_before_exit": mae,
                    }
                )

            exit_idx = entry_idx + 20
            exit_price = float(frame.loc[exit_idx, "close"])
            mfe, mae = _path_before_exit(frame, entry_idx, exit_idx, entry)
            rule_trades["Exit after 20 bars"].append(
                {
                    "trade_return": exit_price / entry - 1.0,
                    "bars_held": 20,
                    "mfe_before_exit": mfe,
                    "mae_before_exit": mae,
                }
            )

            target = entry * 1.08
            stop = entry * 0.95
            tp_sl_exit_idx = entry_idx + 20
            tp_sl_return = float(frame.loc[tp_sl_exit_idx, "close"] / entry - 1.0)
            for j in range(entry_idx + 1, entry_idx + 21):
                low_price = float(frame.loc[j, "low"] if "low" in frame.columns else frame.loc[j, "close"])
                high_price = float(frame.loc[j, "high"] if "high" in frame.columns else frame.loc[j, "close"])
                if low_price <= stop:
                    tp_sl_exit_idx = j
                    tp_sl_return = -0.05
                    break
                if high_price >= target:
                    tp_sl_exit_idx = j
                    tp_sl_return = 0.08
                    break
            mfe, mae = _path_before_exit(frame, entry_idx, tp_sl_exit_idx, entry)
            rule_trades["TP +8% / SL -5%"].append(
                {
                    "trade_return": tp_sl_return,
                    "bars_held": tp_sl_exit_idx - entry_idx,
                    "mfe_before_exit": mfe,
                    "mae_before_exit": mae,
                }
            )

    rows = [
        {"exit_rule": rule_name, **_trade_summary(trades)}
        for rule_name, trades in rule_trades.items()
    ]
    return pd.DataFrame(rows)


def build_rsi_bb_dca_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Simple RSI/BB add-permission simulation with up to three equal entries."""
    required = ["ticker", "date", "close", "rsi_14", "signal_rsi_bb"]
    if not _has_columns(df, required):
        return pd.DataFrame()

    trades: list[dict[str, Any]] = []
    for _, ticker_df in df.groupby("ticker", dropna=False):
        frame = ticker_df.sort_values("date").reset_index(drop=True)
        signals = _boolean_mask(frame["signal_rsi_bb"])
        for entry_idx in list(np.flatnonzero(signals.to_numpy())):
            first_entry = float(frame.loc[entry_idx, "close"])
            if pd.isna(first_entry) or first_entry <= 0:
                continue

            entries = [first_entry]
            last_entry = first_entry
            mfe = 0.0
            mae = 0.0
            for j in range(entry_idx + 1, len(frame)):
                entry_prices = np.array(entries, dtype=float)
                high_price = float(frame.loc[j, "high"] if "high" in frame.columns else frame.loc[j, "close"])
                low_price = float(frame.loc[j, "low"] if "low" in frame.columns else frame.loc[j, "close"])
                mfe = max(mfe, float(np.mean(high_price / entry_prices - 1.0)))
                mae = min(mae, float(np.mean(low_price / entry_prices - 1.0)))

                rsi_value = frame.loc[j, "rsi_14"]
                if not pd.isna(rsi_value) and rsi_value >= 70:
                    exit_price = float(frame.loc[j, "close"])
                    trade_return = float(np.mean(exit_price / entry_prices - 1.0))
                    trades.append(
                        {
                            "trade_return": trade_return,
                            "bars_held": j - entry_idx,
                            "entry_count": len(entries),
                            "mfe_before_exit": mfe,
                            "mae_before_exit": mae,
                        }
                    )
                    break

                close_price = float(frame.loc[j, "close"])
                if (
                    len(entries) < 3
                    and bool(signals.iloc[j])
                    and close_price <= last_entry * 0.985
                ):
                    entries.append(close_price)
                    last_entry = close_price

    if not trades:
        return pd.DataFrame(
            [
                {
                    "dca_rule": "RSI/BB up to 3 entries, exit RSI >= 70",
                    "trade_count": 0,
                    "avg_return": np.nan,
                    "median_return": np.nan,
                    "win_rate": np.nan,
                    "avg_entries": np.nan,
                    "max_entries": np.nan,
                    "avg_mae": np.nan,
                    "median_mae": np.nan,
                    "worst_trade_return": np.nan,
                    "worst_mae": np.nan,
                }
            ]
        )

    t = pd.DataFrame(trades)
    returns = pd.to_numeric(t["trade_return"], errors="coerce")
    entries = pd.to_numeric(t["entry_count"], errors="coerce")
    mae = pd.to_numeric(t["mae_before_exit"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "dca_rule": "RSI/BB up to 3 entries, exit RSI >= 70",
                "trade_count": len(t),
                "avg_return": float(returns.mean()),
                "median_return": float(returns.median()),
                "win_rate": positive_rate(returns),
                "avg_entries": float(entries.mean()),
                "max_entries": int(entries.max()),
                "avg_mae": float(mae.mean()),
                "median_mae": float(mae.median()),
                "worst_trade_return": float(returns.min()),
                "worst_mae": float(mae.min()),
            }
        ]
    )


def build_excess_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build compact CSV-ready excess-return summary tables."""
    return {
        "summary_global_signal_excess": build_global_signal_excess_summary(df),
        "summary_sector_signal_excess": build_sector_signal_excess_summary(df),
        "summary_ticker_signal_excess": build_ticker_signal_excess_summary(df),
        "summary_market_regime_excess": build_market_regime_excess_summary(df),
        "summary_high_vs_low_excess": build_high_vs_low_excess_summary(df),
        "summary_personality_signal_excess": build_personality_signal_excess_summary(df),
        "summary_personality_high_vs_low": build_personality_high_vs_low_summary(df),
        "summary_ticker_personality_excess": build_ticker_personality_excess_summary(df),
        "summary_path_by_personality_signal": build_path_by_personality_signal(df),
        "summary_path_by_sector_signal": build_path_by_sector_signal(df),
        "summary_path_by_ticker_personality": build_path_by_ticker_personality(df),
        "summary_signal_lab": build_signal_lab_summary(df),
        "summary_rsi_bb_by_sector": build_rsi_bb_by_sector(df),
        "summary_rsi_bb_by_personality": build_rsi_bb_by_personality(df),
        "summary_rsi_bb_exit_rules": build_rsi_bb_exit_rule_summary(df),
        "summary_rsi_bb_dca": build_rsi_bb_dca_summary(df),
    }


def export_excess_summary_csvs(
    df: pd.DataFrame,
    results_dir: str | Path,
) -> dict[str, Path]:
    """Write compact excess-return summary CSVs to data/results."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in build_excess_summary_tables(df).items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def build_excess_console_summary(df: pd.DataFrame) -> list[str]:
    """Console summary focused on baseline-adjusted edge."""
    universe_avg = _mean_col(df, "future_20d_return")
    high = df[df["signal"] >= _HIGH_SIGNAL_THRESHOLD] if "signal" in df.columns else df.iloc[0:0]
    low = df[df["signal"] <= _LOW_SIGNAL_THRESHOLD] if "signal" in df.columns else df.iloc[0:0]
    high_avg = _mean_col(high, "future_20d_return")
    low_avg = _mean_col(low, "future_20d_return")

    lines = [
        f"Unconditional universe avg 20D return: {universe_avg * 100:+.2f}%" if not pd.isna(universe_avg) else "Unconditional universe avg 20D return: n/a",
        f"High signal avg 20D return: {high_avg * 100:+.2f}%" if not pd.isna(high_avg) else "High signal avg 20D return: n/a",
        (
            f"High signal excess vs universe: {(high_avg - universe_avg) * 100:+.2f}%"
            if not pd.isna(high_avg) and not pd.isna(universe_avg)
            else "High signal excess vs universe: n/a"
        ),
        f"Low signal avg 20D return: {low_avg * 100:+.2f}%" if not pd.isna(low_avg) else "Low signal avg 20D return: n/a",
        (
            f"Low signal excess vs universe: {(low_avg - universe_avg) * 100:+.2f}%"
            if not pd.isna(low_avg) and not pd.isna(universe_avg)
            else "Low signal excess vs universe: n/a"
        ),
    ]

    sector_excess = build_sector_signal_excess_summary(df)
    if sector_excess.empty:
        lines.extend(
            [
                "Best sector by high signal excess vs sector ETF: n/a",
                "Best sector by low signal excess vs sector ETF: n/a",
            ]
        )
    else:
        high_sector = sector_excess[
            sector_excess["condition"] == "High signal >= 40%"
        ].dropna(subset=["avg_excess_vs_sector_etf_20d"])
        high_sector = high_sector[high_sector["valid_excess_vs_sector_etf_20d_rows"] > 0]
        low_sector = sector_excess[
            sector_excess["condition"] == "Low signal <= -20%"
        ].dropna(subset=["avg_excess_vs_sector_etf_20d"])
        low_sector = low_sector[low_sector["valid_excess_vs_sector_etf_20d_rows"] > 0]
        lines.append(
            "Best sector by high signal excess vs sector ETF: "
            + _format_console_metric(
                high_sector.loc[high_sector["avg_excess_vs_sector_etf_20d"].idxmax()]
                if not high_sector.empty else None,
                "sector",
                "avg_excess_vs_sector_etf_20d",
                "valid_excess_vs_sector_etf_20d_rows",
            )
        )
        lines.append(
            "Best sector by low signal excess vs sector ETF: "
            + _format_console_metric(
                low_sector.loc[low_sector["avg_excess_vs_sector_etf_20d"].idxmax()]
                if not low_sector.empty else None,
                "sector",
                "avg_excess_vs_sector_etf_20d",
                "valid_excess_vs_sector_etf_20d_rows",
            )
        )

    ticker_excess = build_ticker_signal_excess_summary(df)
    if ticker_excess.empty:
        lines.extend(
            [
                "Best ticker by high signal excess vs sector ETF: n/a",
                "Best ticker by low signal excess vs sector ETF: n/a",
            ]
        )
    else:
        high_ticker = ticker_excess.dropna(
            subset=["high_signal_avg_excess_vs_sector_etf_20d"]
        )
        high_ticker = high_ticker[high_ticker["high_signal_valid_20d_rows"] > 0]
        low_ticker = ticker_excess.dropna(
            subset=["low_signal_avg_excess_vs_sector_etf_20d"]
        )
        low_ticker = low_ticker[low_ticker["low_signal_valid_20d_rows"] > 0]
        lines.append(
            "Best ticker by high signal excess vs sector ETF: "
            + _format_console_metric(
                high_ticker.loc[
                    high_ticker["high_signal_avg_excess_vs_sector_etf_20d"].idxmax()
                ] if not high_ticker.empty else None,
                "ticker",
                "high_signal_avg_excess_vs_sector_etf_20d",
                "high_signal_valid_20d_rows",
            )
        )
        lines.append(
            "Best ticker by low signal excess vs sector ETF: "
            + _format_console_metric(
                low_ticker.loc[
                    low_ticker["low_signal_avg_excess_vs_sector_etf_20d"].idxmax()
                ] if not low_ticker.empty else None,
                "ticker",
                "low_signal_avg_excess_vs_sector_etf_20d",
                "low_signal_valid_20d_rows",
            )
        )

    return lines


def _format_ticker_personality_metric(
    row: pd.Series | None,
    value_col: str,
    valid_col: str,
) -> str:
    if row is None or pd.isna(row.get(value_col)):
        return "n/a"
    valid = row.get(valid_col)
    valid_str = f", valid 20D={int(valid):,}" if not pd.isna(valid) else ""
    return (
        f"{row.get('ticker')} / {row.get('stock_personality')} "
        f"({row[value_col] * 100:+.2f}%{valid_str})"
    )


def build_personality_console_summary(df: pd.DataFrame) -> list[str]:
    """Console summary for stock-personality signal behavior."""
    group_summary = build_personality_high_vs_low_summary(
        df, label_col="stock_personality_group"
    )
    label_summary = build_personality_high_vs_low_summary(df)
    ticker_summary = build_ticker_personality_excess_summary(df)

    lines: list[str] = []
    if group_summary.empty:
        return [
            "Best personality group for high-signal continuation: n/a",
            "Best personality group for low-signal rebound: n/a",
            "flat_or_laggard low-signal rebound: n/a",
            "secular_winner high-signal continuation: n/a",
            "Best ticker/personality by high-signal excess: n/a",
            "Best ticker/personality by low-signal excess: n/a",
        ]

    group_valid = group_summary.dropna(
        subset=["high_minus_low_excess_vs_sector_etf_20d"]
    )
    best_high_group = (
        group_valid.loc[group_valid["high_minus_low_excess_vs_sector_etf_20d"].idxmax()]
        if not group_valid.empty
        else None
    )
    best_low_group = (
        group_valid.loc[group_valid["high_minus_low_excess_vs_sector_etf_20d"].idxmin()]
        if not group_valid.empty
        else None
    )

    lines.append(
        "Best personality group for high-signal continuation: "
        + _format_console_metric(
            best_high_group,
            "stock_personality_group",
            "high_minus_low_excess_vs_sector_etf_20d",
        )
    )
    if best_low_group is None or pd.isna(best_low_group.get("high_minus_low_excess_vs_sector_etf_20d")):
        lines.append("Best personality group for low-signal rebound: n/a")
    else:
        rebound = -best_low_group["high_minus_low_excess_vs_sector_etf_20d"]
        lines.append(
            "Best personality group for low-signal rebound: "
            f"{best_low_group['stock_personality_group']} ({rebound * 100:+.2f}% low-minus-high)"
        )

    flat_rows = group_summary[
        group_summary["stock_personality_group"] == "flat_or_laggard"
    ]
    if flat_rows.empty:
        lines.append("flat_or_laggard low-signal rebound: n/a")
    else:
        row = flat_rows.iloc[0]
        high_avg = row["high_signal_avg_excess_vs_sector_etf_20d"]
        low_avg = row["low_signal_avg_excess_vs_sector_etf_20d"]
        improves = not pd.isna(high_avg) and not pd.isna(low_avg) and low_avg > high_avg
        lines.append(
            "flat_or_laggard low-signal rebound: "
            + ("yes" if improves else "no")
            + (
                f" (low {low_avg * 100:+.2f}% vs high {high_avg * 100:+.2f}%)"
                if not pd.isna(high_avg) and not pd.isna(low_avg)
                else ""
            )
        )

    winner_rows = label_summary[
        label_summary["stock_personality"] == "secular_winner"
    ]
    if winner_rows.empty:
        lines.append("secular_winner high-signal continuation: n/a")
    else:
        row = winner_rows.iloc[0]
        high_avg = row["high_signal_avg_excess_vs_sector_etf_20d"]
        low_avg = row["low_signal_avg_excess_vs_sector_etf_20d"]
        improves = not pd.isna(high_avg) and not pd.isna(low_avg) and high_avg > low_avg
        lines.append(
            "secular_winner high-signal continuation: "
            + ("yes" if improves else "no")
            + (
                f" (high {high_avg * 100:+.2f}% vs low {low_avg * 100:+.2f}%)"
                if not pd.isna(high_avg) and not pd.isna(low_avg)
                else ""
            )
        )

    if ticker_summary.empty:
        lines.extend(
            [
                "Best ticker/personality by high-signal excess: n/a",
                "Best ticker/personality by low-signal excess: n/a",
            ]
        )
        return lines

    high_ticker = ticker_summary.dropna(
        subset=["high_signal_avg_excess_vs_sector_etf_20d"]
    )
    high_ticker = high_ticker[
        high_ticker["high_signal_valid_20d_rows"] >= _MIN_VERDICT_VALID_20D
    ]
    low_ticker = ticker_summary.dropna(
        subset=["low_signal_avg_excess_vs_sector_etf_20d"]
    )
    low_ticker = low_ticker[
        low_ticker["low_signal_valid_20d_rows"] >= _MIN_VERDICT_VALID_20D
    ]

    lines.append(
        "Best ticker/personality by high-signal excess: "
        + _format_ticker_personality_metric(
            high_ticker.loc[
                high_ticker["high_signal_avg_excess_vs_sector_etf_20d"].idxmax()
            ] if not high_ticker.empty else None,
            "high_signal_avg_excess_vs_sector_etf_20d",
            "high_signal_valid_20d_rows",
        )
    )
    lines.append(
        "Best ticker/personality by low-signal excess: "
        + _format_ticker_personality_metric(
            low_ticker.loc[
                low_ticker["low_signal_avg_excess_vs_sector_etf_20d"].idxmax()
            ] if not low_ticker.empty else None,
            "low_signal_avg_excess_vs_sector_etf_20d",
            "low_signal_valid_20d_rows",
        )
    )
    return lines


def _format_path_combo(
    row: pd.Series | None,
    name_cols: list[str],
    value_col: str,
    valid_col: str = "valid_path_20d_rows",
    signed: bool = False,
) -> str:
    if row is None or pd.isna(row.get(value_col)):
        return "n/a"
    name = " / ".join(
        str(row.get(col))
        for col in name_cols
        if col in row and not pd.isna(row.get(col)) and str(row.get(col)) != ""
    )
    value = row[value_col] * 100
    value_str = f"{value:+.2f}%" if signed else f"{value:.1f}%"
    valid = row.get(valid_col)
    valid_str = f", valid path={int(valid):,}" if not pd.isna(valid) else ""
    return f"{name} ({value_str}{valid_str})"


def _format_deep_drawdown_adverse_line(
    path_personality: pd.DataFrame,
    condition: str,
    label: str,
) -> str:
    subset = path_personality[
        (path_personality["stock_personality"] == "deep_drawdown")
        & (path_personality["condition"] == condition)
    ]
    if subset.empty or pd.isna(subset.iloc[0].get("median_20d_mae")):
        return f"{label} acceptable adverse excursion: n/a"

    row = subset.iloc[0]
    median_mae = row["median_20d_mae"]
    threshold = -0.10
    acceptable = median_mae >= threshold
    valid = row.get("valid_path_20d_rows")
    valid_str = f", valid path={int(valid):,}" if not pd.isna(valid) else ""
    return (
        f"{label} acceptable adverse excursion: "
        f"{'yes' if acceptable else 'no'} "
        f"(median MAE {median_mae * 100:+.2f}%, threshold {threshold * 100:+.0f}%{valid_str})"
    )


def build_path_console_summary(df: pd.DataFrame) -> list[str]:
    """Console summary for forward path / tradability labels."""
    path_personality = build_path_by_personality_signal(df)
    path_ticker = build_path_by_ticker_personality(df)

    if path_personality.empty:
        return [
            "Best personality/signal by +10 before -5 hit rate: n/a",
            "Best ticker/personality/signal by +10 before -5 hit rate with at least 50 rows: n/a",
            "Worst personality/signal combo by median adverse excursion: n/a",
            "deep_drawdown + high signal acceptable adverse excursion: n/a",
            "deep_drawdown + low signal acceptable adverse excursion: n/a",
        ]

    hit_col = "hit_plus_10_before_minus_5_rate"
    valid_personality = path_personality.dropna(subset=[hit_col])
    valid_personality = valid_personality[valid_personality["valid_path_20d_rows"] > 0]
    best_personality = (
        valid_personality.loc[valid_personality[hit_col].idxmax()]
        if not valid_personality.empty
        else None
    )

    valid_ticker = path_ticker.dropna(subset=[hit_col]) if not path_ticker.empty else path_ticker
    if not valid_ticker.empty:
        valid_ticker = valid_ticker[valid_ticker["valid_path_20d_rows"] >= 50]
    best_ticker = (
        valid_ticker.loc[valid_ticker[hit_col].idxmax()]
        if not valid_ticker.empty
        else None
    )

    adverse = path_personality.dropna(subset=["median_20d_mae"])
    adverse = adverse[adverse["valid_path_20d_rows"] > 0]
    worst_adverse = (
        adverse.loc[adverse["median_20d_mae"].idxmin()]
        if not adverse.empty
        else None
    )

    return [
        "Best personality/signal by +10 before -5 hit rate: "
        + _format_path_combo(
            best_personality,
            ["stock_personality", "condition"],
            hit_col,
            signed=False,
        ),
        "Best ticker/personality/signal by +10 before -5 hit rate with at least 50 rows: "
        + _format_path_combo(
            best_ticker,
            ["ticker", "stock_personality", "condition"],
            hit_col,
            signed=False,
        ),
        "Worst personality/signal combo by median adverse excursion: "
        + _format_path_combo(
            worst_adverse,
            ["stock_personality", "condition"],
            "median_20d_mae",
            signed=True,
        ),
        _format_deep_drawdown_adverse_line(
            path_personality,
            "High signal >= 40%",
            "deep_drawdown + high signal",
        ),
        _format_deep_drawdown_adverse_line(
            path_personality,
            "Low signal <= -20%",
            "deep_drawdown + low signal",
        ),
    ]


def _format_signal_lab_metric(row: pd.Series | None) -> str:
    if row is None:
        return "n/a"
    parts = [f"rows={int(row.get('rows', 0)):,}"]
    for label, col, signed in [
        ("avg20", "avg_20d_return", True),
        ("win20", "20d_win_rate", False),
        ("exSecETF20", "avg_excess_vs_sector_etf_20d", True),
        ("medMAE20", "median_mae_20d", True),
    ]:
        value = row.get(col)
        if pd.isna(value):
            parts.append(f"{label}=n/a")
        else:
            fmt = f"{value * 100:+.2f}%" if signed else f"{value * 100:.1f}%"
            parts.append(f"{label}={fmt}")
    return ", ".join(parts)


def _signal_lab_row(table: pd.DataFrame, condition: str) -> pd.Series | None:
    if table.empty or "condition" not in table.columns:
        return None
    match = table[table["condition"] == condition]
    return match.iloc[0] if not match.empty else None


def build_rsi_bb_console_summary(df: pd.DataFrame) -> list[str]:
    """Console summary for the RSI/Bollinger vs Markov research lab."""
    lab = build_signal_lab_summary(df)
    exits = build_rsi_bb_exit_rule_summary(df)

    if lab.empty:
        return [
            "RSI/BB standalone: n/a",
            "RSI/BB + Markov high: n/a",
            "RSI/BB + Markov low: n/a",
            "RSI/BB + Markov not bearish: n/a",
            "Best RSI/BB filter by 20D excess vs sector ETF: n/a",
            "Best RSI/BB exit rule by average return: n/a",
            "Best RSI/BB exit rule by win rate: n/a",
            "Markov improves RSI/BB standalone win rate: n/a",
            "Markov reduces RSI/BB median MAE: n/a",
            "Markov improves RSI/BB excess return: n/a",
        ]

    standalone = _signal_lab_row(lab, "RSI/BB oversold")
    markov_high = _signal_lab_row(lab, "RSI/BB + Markov high")
    markov_low = _signal_lab_row(lab, "RSI/BB + Markov low")
    not_bearish = _signal_lab_row(lab, "RSI/BB + Markov not bearish")

    lines = [
        "RSI/BB standalone: " + _format_signal_lab_metric(standalone),
        "RSI/BB + Markov high: " + _format_signal_lab_metric(markov_high),
        "RSI/BB + Markov low: " + _format_signal_lab_metric(markov_low),
        "RSI/BB + Markov not bearish: " + _format_signal_lab_metric(not_bearish),
    ]

    rsi_filters = lab[lab["condition"].astype(str).str.startswith("RSI/BB")]
    rsi_filters = (
        rsi_filters.dropna(subset=["avg_excess_vs_sector_etf_20d"])
        if "avg_excess_vs_sector_etf_20d" in rsi_filters.columns
        else pd.DataFrame()
    )
    best_filter = (
        rsi_filters.loc[rsi_filters["avg_excess_vs_sector_etf_20d"].idxmax()]
        if not rsi_filters.empty
        else None
    )
    lines.append(
        "Best RSI/BB filter by 20D excess vs sector ETF: "
        + _format_console_metric(
            best_filter,
            "condition",
            "avg_excess_vs_sector_etf_20d",
            "valid_excess_vs_sector_etf_20d_rows",
        )
    )

    def _format_exit_metric(row: pd.Series | None, value_col: str, signed: bool = True) -> str:
        if row is None or pd.isna(row.get(value_col)):
            return "n/a"
        value = row[value_col] * 100
        value_str = f"{value:+.2f}%" if signed else f"{value:.1f}%"
        trades = row.get("trade_count")
        trade_str = f", trades={int(trades):,}" if not pd.isna(trades) else ""
        return f"{row['exit_rule']} ({value_str}{trade_str})"

    if exits.empty:
        lines.extend(
            [
                "Best RSI/BB exit rule by average return: n/a",
                "Best RSI/BB exit rule by win rate: n/a",
            ]
        )
    else:
        avg_valid = exits.dropna(subset=["avg_trade_return"])
        win_valid = exits.dropna(subset=["win_rate"])
        lines.append(
            "Best RSI/BB exit rule by average return: "
            + _format_exit_metric(
                avg_valid.loc[avg_valid["avg_trade_return"].idxmax()]
                if not avg_valid.empty
                else None,
                "avg_trade_return",
            )
        )
        lines.append(
            "Best RSI/BB exit rule by win rate: "
            + _format_exit_metric(
                win_valid.loc[win_valid["win_rate"].idxmax()]
                if not win_valid.empty
                else None,
                "win_rate",
                signed=False,
            )
        )

    def _compare_candidate(
        candidate: pd.Series | None,
        baseline: pd.Series | None,
        metric: str,
        rate: bool = False,
    ) -> str:
        if candidate is None or baseline is None:
            return "n/a"
        rows = candidate.get("rows", 0)
        if pd.isna(rows) or int(rows) == 0:
            return f"{candidate.get('condition', 'candidate')} n/a (rows=0)"
        a = candidate.get(metric)
        b = baseline.get(metric)
        if pd.isna(a) or pd.isna(b):
            return f"{candidate.get('condition', 'candidate')} n/a"
        improves = a > b
        if rate:
            return (
                f"{candidate.get('condition')} {'yes' if improves else 'no'} "
                f"({a * 100:.1f}% vs {b * 100:.1f}%, rows={int(rows):,})"
            )
        return (
            f"{candidate.get('condition')} {'yes' if improves else 'no'} "
            f"({a * 100:+.2f}% vs {b * 100:+.2f}%, rows={int(rows):,})"
        )

    def _compare_markov_filters(metric: str, rate: bool = False) -> str:
        return "; ".join(
            [
                _compare_candidate(markov_high, standalone, metric, rate=rate),
                _compare_candidate(not_bearish, standalone, metric, rate=rate),
            ]
        )

    lines.append(
        "Markov improves RSI/BB standalone win rate: "
        + _compare_markov_filters("20d_win_rate", rate=True)
    )
    lines.append(
        "Markov reduces RSI/BB median MAE: "
        + _compare_markov_filters("median_mae_20d")
    )
    lines.append(
        "Markov improves RSI/BB excess return: "
        + _compare_markov_filters("avg_excess_vs_sector_etf_20d")
    )
    return lines


def _format_console_metric(
    row: pd.Series | None,
    name_col: str,
    value_col: str,
    valid_col: str | None = None,
    signed: bool = True,
) -> str:
    if row is None or pd.isna(row.get(value_col)):
        return "n/a"
    value = row[value_col] * 100
    value_str = f"{value:+.2f}%" if signed else f"{value:.1f}%"
    n_str = ""
    if valid_col and valid_col in row and not pd.isna(row[valid_col]):
        n_str = f", valid 20D={int(row[valid_col]):,}"
    return f"{row[name_col]} ({value_str}{n_str})"


def build_research_console_summary(
    df: pd.DataFrame,
    ticker_groups: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Small console summary for the per-ticker and group research layer."""
    summary: dict[str, str] = {}

    quality = build_ticker_signal_quality_summary(df)
    if quality.empty:
        return {
            "best_ticker_high_avg": "n/a",
            "best_ticker_high_win": "n/a",
            "worst_ticker_high_avg": "n/a",
            "low_better_tickers": "n/a",
            "best_ticker_market_avg": "n/a",
            "best_group_high_avg": "n/a",
            "best_group_high_win": "n/a",
        }

    valid_high_avg = quality.dropna(subset=["high_signal_avg_20d_return"])
    valid_high_avg = valid_high_avg[valid_high_avg["high_signal_valid_20d_rows"] > 0]
    best_avg = (
        valid_high_avg.loc[valid_high_avg["high_signal_avg_20d_return"].idxmax()]
        if not valid_high_avg.empty
        else None
    )
    worst_avg = (
        valid_high_avg.loc[valid_high_avg["high_signal_avg_20d_return"].idxmin()]
        if not valid_high_avg.empty
        else None
    )

    valid_high_win = quality.dropna(subset=["high_signal_20d_win_rate"])
    valid_high_win = valid_high_win[valid_high_win["high_signal_valid_20d_rows"] > 0]
    best_win = (
        valid_high_win.loc[valid_high_win["high_signal_20d_win_rate"].idxmax()]
        if not valid_high_win.empty
        else None
    )

    inverted = quality[
        quality["low_signal_avg_20d_return"].notna()
        & quality["high_signal_avg_20d_return"].notna()
        & (quality["low_signal_avg_20d_return"] > quality["high_signal_avg_20d_return"])
    ]["ticker"].tolist()

    market = quality.dropna(subset=["market_high_signal_avg_20d_return"])
    market = market[market["market_high_signal_valid_20d_rows"].fillna(0) > 0]
    best_market = (
        market.loc[market["market_high_signal_avg_20d_return"].idxmax()]
        if not market.empty
        else None
    )

    group_summary = build_group_high_signal_summary(df, ticker_groups)
    valid_group_avg = (
        group_summary.dropna(subset=["high_signal_avg_20d_return"])
        if not group_summary.empty
        else pd.DataFrame()
    )
    valid_group_win = (
        group_summary.dropna(subset=["high_signal_20d_win_rate"])
        if not group_summary.empty
        else pd.DataFrame()
    )
    best_group_avg = (
        valid_group_avg.loc[valid_group_avg["high_signal_avg_20d_return"].idxmax()]
        if not valid_group_avg.empty
        else None
    )
    best_group_win = (
        valid_group_win.loc[valid_group_win["high_signal_20d_win_rate"].idxmax()]
        if not valid_group_win.empty
        else None
    )

    summary["best_ticker_high_avg"] = _format_console_metric(
        best_avg,
        "ticker",
        "high_signal_avg_20d_return",
        "high_signal_valid_20d_rows",
    )
    summary["best_ticker_high_win"] = _format_console_metric(
        best_win,
        "ticker",
        "high_signal_20d_win_rate",
        "high_signal_valid_20d_rows",
        signed=False,
    )
    summary["worst_ticker_high_avg"] = _format_console_metric(
        worst_avg,
        "ticker",
        "high_signal_avg_20d_return",
        "high_signal_valid_20d_rows",
    )
    summary["low_better_tickers"] = ", ".join(inverted) if inverted else "none"
    summary["best_ticker_market_avg"] = _format_console_metric(
        best_market,
        "ticker",
        "market_high_signal_avg_20d_return",
        "market_high_signal_valid_20d_rows",
    )
    summary["best_group_high_avg"] = _format_console_metric(
        best_group_avg,
        "group",
        "high_signal_avg_20d_return",
        "high_signal_valid_20d_rows",
    )
    summary["best_group_high_win"] = _format_console_metric(
        best_group_win,
        "group",
        "high_signal_20d_win_rate",
        "high_signal_valid_20d_rows",
        signed=False,
    )
    return summary


def _best_high_signal_ticker_by_type(
    df: pd.DataFrame,
    instrument_type: str,
) -> pd.Series | None:
    if not _has_columns(df, ["ticker", "signal", "instrument_type"]):
        return None

    subset = df[
        (df["instrument_type"] == instrument_type)
        & (df["signal"] >= _HIGH_SIGNAL_THRESHOLD)
    ]
    if subset.empty:
        return None

    rows: list[dict[str, Any]] = []
    for ticker in sorted(subset["ticker"].dropna().unique()):
        t = subset[subset["ticker"] == ticker]
        stats = _return_stats(t, [20])
        rows.append(
            {
                "ticker": ticker,
                "sector": t["sector"].iloc[0] if "sector" in t.columns and not t.empty else "",
                "avg_20d_return": stats["avg_20d_return"],
                "valid_20d_rows": stats["valid_20d_rows"],
            }
        )
    result = pd.DataFrame(rows).dropna(subset=["avg_20d_return"])
    result = result[result["valid_20d_rows"] > 0]
    if result.empty:
        return None
    return result.loc[result["avg_20d_return"].idxmax()]


def build_sector_console_summary(
    df: pd.DataFrame,
    active_universe: str,
    requested_symbols: list[str],
    processed_symbols: list[str],
    skipped_symbols: list[str],
) -> list[str]:
    """Console summary for the active structured sector universe."""
    lines = [
        f"Active universe: {active_universe}",
        f"Requested symbols: {len(requested_symbols)}",
        f"Successfully processed symbols: {len(processed_symbols)}",
        "Failed/skipped symbols: " + (", ".join(skipped_symbols) if skipped_symbols else "none"),
    ]

    sector_summary = build_sector_summary(df)
    valid_sector_avg = (
        sector_summary.dropna(subset=["high_signal_avg_20d_return"])
        if not sector_summary.empty
        else pd.DataFrame()
    )
    valid_sector_avg = (
        valid_sector_avg[valid_sector_avg["high_signal_valid_20d_rows"] > 0]
        if not valid_sector_avg.empty
        else valid_sector_avg
    )
    best_sector_avg = (
        valid_sector_avg.loc[valid_sector_avg["high_signal_avg_20d_return"].idxmax()]
        if not valid_sector_avg.empty
        else None
    )

    valid_sector_win = (
        sector_summary.dropna(subset=["high_signal_20d_win_rate"])
        if not sector_summary.empty
        else pd.DataFrame()
    )
    valid_sector_win = (
        valid_sector_win[valid_sector_win["high_signal_valid_20d_rows"] > 0]
        if not valid_sector_win.empty
        else valid_sector_win
    )
    best_sector_win = (
        valid_sector_win.loc[valid_sector_win["high_signal_20d_win_rate"].idxmax()]
        if not valid_sector_win.empty
        else None
    )

    lines.append(
        "Best sector by high-signal avg 20D return: "
        + _format_console_metric(
            best_sector_avg,
            "sector",
            "high_signal_avg_20d_return",
            "high_signal_valid_20d_rows",
        )
    )
    lines.append(
        "Best sector by high-signal 20D win rate: "
        + _format_console_metric(
            best_sector_win,
            "sector",
            "high_signal_20d_win_rate",
            "high_signal_valid_20d_rows",
            signed=False,
        )
    )

    market = build_sector_market_regime_comparison(df)
    weak_better: list[str] = []
    if not market.empty:
        avg_pivot = market.pivot(
            index="sector",
            columns="condition",
            values="avg_20d_return",
        )
        if {
            "High signal + market healthy",
            "High signal + market weak",
        }.issubset(set(avg_pivot.columns)):
            mask = (
                avg_pivot["High signal + market weak"]
                > avg_pivot["High signal + market healthy"]
            )
            weak_better = avg_pivot.index[mask.fillna(False)].tolist()
    lines.append(
        "Sectors where market weak outperforms market healthy: "
        + (", ".join(weak_better) if weak_better else "none")
    )

    best_stock = _best_high_signal_ticker_by_type(df, "stock")
    best_etf = _best_high_signal_ticker_by_type(df, "sector_etf")
    lines.append(
        "Best individual stock by high-signal avg 20D return: "
        + _format_console_metric(best_stock, "ticker", "avg_20d_return", "valid_20d_rows")
    )
    lines.append(
        "Best sector ETF by high-signal avg 20D return: "
        + _format_console_metric(best_etf, "ticker", "avg_20d_return", "valid_20d_rows")
    )
    return lines


def print_filter_summary(df: pd.DataFrame) -> None:
    """
    Print a short console summary of the best threshold+filter combinations
    by avg 20D return and 20D win rate.
    """
    combined = build_combined_conditions_table(df).reset_index()
    valid = combined.dropna(subset=["avg_20d", "win_20d"])

    if valid.empty:
        print("[Filter Summary] No valid combinations found.")
        return

    # Prefer valid 20D rows when available.
    count_col = "valid_20d_rows" if "valid_20d_rows" in valid.columns else "rows"
    candidates = valid[valid[count_col] >= 30] if (valid[count_col] >= 30).any() else valid

    best_ret = candidates.loc[candidates["avg_20d"].idxmax()]
    best_win = candidates.loc[candidates["win_20d"].idxmax()]

    sep = "=" * 62
    print(f"\n{sep}")
    print("  MARKOV FILTER COMBINATION SUMMARY")
    print(sep)

    print(f"\n  Best avg 20D return:")
    print(f"    Threshold  : signal >= {best_ret['threshold']}")
    print(f"    Condition  : {best_ret['condition']}")
    print(f"    Avg 20D    : {best_ret['avg_20d'] * 100:+.2f}%")
    print(f"    Rows       : {int(best_ret['rows']):,}")
    if "valid_20d_rows" in best_ret:
        print(f"    Valid 20D  : {int(best_ret['valid_20d_rows']):,}")

    print(f"\n  Best 20D win rate:")
    print(f"    Threshold  : signal >= {best_win['threshold']}")
    print(f"    Condition  : {best_win['condition']}")
    print(f"    Win rate   : {best_win['win_20d'] * 100:.1f}%")
    print(f"    Rows       : {int(best_win['rows']):,}")
    if "valid_20d_rows" in best_win:
        print(f"    Valid 20D  : {int(best_win['valid_20d_rows']):,}")

    # Full table
    print(f"\n  All combinations — 20D return:")
    display_cols = ["threshold", "condition", "rows"]
    if "valid_20d_rows" in combined.columns:
        display_cols.append("valid_20d_rows")
    display_cols.extend(["avg_20d", "win_20d"])
    disp = combined[display_cols].copy()
    if "valid_20d_rows" in disp.columns:
        disp["valid_20d_rows"] = disp["valid_20d_rows"].map(
            lambda x: f"{int(x):,}" if not pd.isna(x) else "0"
        )
    disp["avg_20d"] = disp["avg_20d"].map(
        lambda x: f"{x * 100:+.2f}%" if not pd.isna(x) else "—"
    )
    disp["win_20d"] = disp["win_20d"].map(
        lambda x: f"{x * 100:.1f}%" if not pd.isna(x) else "—"
    )
    disp.columns = ["Threshold", "Condition", "Rows"] + (
        ["Valid 20D"] if "valid_20d_rows" in combined.columns else []
    ) + ["Avg 20D", "Win 20D"]
    print(disp.to_string(index=False, col_space=12))
    print(f"{sep}\n")


# ── Private chart functions ──────────────────────────────────────────────────

def _chart_returns_by_bucket(bucket_df: pd.DataFrame) -> str:
    labels = bucket_df.index.tolist()
    fig = go.Figure()
    for col, name, color in [
        ("avg_5d_return", "Avg 5D", "#4c78a8"),
        ("avg_10d_return", "Avg 10D", "#f58518"),
        ("avg_20d_return", "Avg 20D", "#72b7b2"),
    ]:
        vals = (bucket_df[col] * 100).tolist()
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name, marker_color=color,
            text=[f"{v:+.2f}%" if not np.isnan(v) else "" for v in vals],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Average Forward Return by Signal Bucket",
        yaxis_title="Return (%)", xaxis_title="Signal Bucket",
        yaxis_ticksuffix="%", **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_hit_rate_by_bucket(bucket_df: pd.DataFrame) -> str:
    labels = bucket_df.index.tolist()
    fig = go.Figure()
    for col, name, color in [
        ("positive_10d_rate", "Positive 10D Rate", "#f58518"),
        ("positive_20d_rate", "Positive 20D Rate", "#72b7b2"),
    ]:
        vals = (bucket_df[col] * 100).tolist()
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name, marker_color=color,
            text=[f"{v:.1f}%" if not np.isnan(v) else "" for v in vals],
            textposition="outside",
        ))
    fig.add_shape(
        type="line", x0=-0.5, x1=len(labels) - 0.5, xref="x",
        y0=50, y1=50, yref="y",
        line=dict(dash="dot", color="gray", width=1.5),
    )
    fig.update_layout(
        barmode="group",
        title="Positive-Return Rate by Signal Bucket",
        yaxis_title="% Positive Returns", xaxis_title="Signal Bucket",
        yaxis_ticksuffix="%", yaxis_range=[0, 105], **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_ticker_at_high_signal(df: pd.DataFrame, threshold: float = 0.10) -> str:
    high = df[df["signal"] >= threshold]
    if high.empty:
        return f"<p style='color:#888'>No rows with signal ≥ {threshold*100:.0f}%.</p>"
    by_ticker = (
        high.groupby("ticker")["future_20d_return"]
        .mean().dropna().sort_values() * 100
    )
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in by_ticker.values]
    fig = go.Figure(go.Bar(
        x=by_ticker.index.tolist(), y=by_ticker.values.tolist(),
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in by_ticker.values], textposition="outside",
    ))
    fig.update_layout(
        title=f"Avg 20D Return by Ticker (Signal ≥ {threshold*100:.0f}%)",
        yaxis_title="Avg 20D Return (%)", yaxis_ticksuffix="%",
        xaxis_title="Ticker", **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_scatter(df: pd.DataFrame, max_points: int = 8_000) -> str:
    valid = df[["ticker", "signal", "future_20d_return"]].dropna().copy()
    if len(valid) > max_points:
        valid = valid.sample(max_points, random_state=42)
    xs = valid["signal"] * 100
    ys = valid["future_20d_return"] * 100
    fig = px.scatter(
        valid, x=xs, y=ys, color="ticker", opacity=0.35,
        labels={"x": "Signal (%)", "y": "20D Return (%)"},
        title="Signal vs 20-Day Forward Return",
    )
    if len(xs) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b, mode="lines", name="OLS (all)",
            line=dict(color="black", width=2, dash="dash"),
        ))
    fig.update_layout(**_CHART_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_box_by_bucket(df: pd.DataFrame) -> str:
    bucketed = _assign_buckets(df)[["bucket", "future_20d_return"]].dropna()
    fig = go.Figure()
    for label in _BUCKET_LABELS:
        vals = bucketed.loc[bucketed["bucket"] == label, "future_20d_return"] * 100
        if vals.empty:
            continue
        fig.add_trace(go.Box(y=vals, name=label, boxmean="sd"))
    fig.update_layout(
        title="20D Return Distribution by Signal Bucket",
        yaxis_title="20D Return (%)", yaxis_ticksuffix="%",
        showlegend=False, **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_ticker_timeline(df: pd.DataFrame, ticker: str) -> str:
    t = df[df["ticker"] == ticker].sort_values("date")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=t["date"], y=t["close"], name="Close",
                   line=dict(color="#4c78a8", width=1.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t["date"], y=t["signal"] * 100, name="Signal (%)",
                   line=dict(color="#f58518", width=1),
                   fill="tozeroy", fillcolor="rgba(245,133,24,0.07)"),
        secondary_y=True,
    )
    x_range = [t["date"].iloc[0], t["date"].iloc[-1]]
    for level, dash in [(0, "solid"), (10, "dot"), (-10, "dot")]:
        fig.add_trace(
            go.Scatter(x=x_range, y=[level, level], mode="lines",
                       line=dict(color="rgba(150,150,150,0.45)", width=1, dash=dash),
                       showlegend=False, hoverinfo="skip"),
            secondary_y=True,
        )
    fig.update_layout(
        title=f"{ticker} — Close & Markov Signal",
        xaxis_title="Date", height=360, template="plotly_white",
        font=dict(family="system-ui, sans-serif", size=11),
        margin=dict(l=55, r=65, t=50, b=40),
        legend=dict(orientation="h", y=1.06),
    )
    fig.update_yaxes(title_text="Close ($)", secondary_y=False)
    fig.update_yaxes(title_text="Signal (%)", ticksuffix="%", secondary_y=True)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_filter_metric(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    yaxis_title: str = "(%)",
) -> str:
    """Line+marker chart: metric by signal threshold, one line per condition."""
    flat = df.reset_index()
    fig = go.Figure()
    for cond in flat["condition"].unique():
        c_data = flat[flat["condition"] == cond].sort_values("threshold")
        y_raw = c_data[metric_col]
        y = y_raw * 100  # fractions → percentages
        if y.dropna().empty:
            continue
        fig.add_trace(go.Scatter(
            x=c_data["threshold"], y=y,
            mode="lines+markers", name=cond,
            marker=dict(size=8),
        ))
    fig.update_layout(
        title=title, xaxis_title="Signal Threshold",
        yaxis_title=yaxis_title, yaxis_ticksuffix="%",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_sample_counts(df: pd.DataFrame) -> str:
    """Grouped bar chart: sample count by condition and threshold."""
    flat = df.reset_index()
    fig = go.Figure()
    for cond in flat["condition"].unique():
        c_data = flat[flat["condition"] == cond].sort_values("threshold")
        fig.add_trace(go.Bar(
            x=c_data["threshold"], y=c_data["rows"], name=cond,
        ))
    fig.update_layout(
        title="Sample Count by Condition & Signal Threshold",
        barmode="group", yaxis_title="Rows", xaxis_title="Signal Threshold",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_ticker_bucket_heatmap(bucket_df: pd.DataFrame) -> str:
    if bucket_df.empty:
        return _warning_html("Per-ticker bucket data is unavailable.")

    tickers = sorted(bucket_df["ticker"].unique())
    values = (
        bucket_df.pivot(index="ticker", columns="signal_bucket", values="avg_20d_return")
        .reindex(index=tickers, columns=_BUCKET_LABELS)
    )
    counts = (
        bucket_df.pivot(index="ticker", columns="signal_bucket", values="valid_20d_rows")
        .reindex(index=tickers, columns=_BUCKET_LABELS)
    )

    text = []
    for ticker in tickers:
        row = []
        for bucket in _BUCKET_LABELS:
            val = values.loc[ticker, bucket]
            n = counts.loc[ticker, bucket]
            if pd.isna(val):
                row.append(f"n={int(n) if not pd.isna(n) else 0}")
            else:
                row.append(f"{val * 100:+.1f}%<br>n={int(n)}")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        x=_BUCKET_LABELS,
        y=tickers,
        z=values.to_numpy(dtype=float) * 100,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Avg 20D %"),
    ))
    fig.update_layout(
        title="Avg 20D Return by Ticker and Signal Bucket",
        xaxis_title="Signal Bucket",
        yaxis_title="Ticker",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_quality_metric(
    quality_df: pd.DataFrame,
    metric_col: str,
    title: str,
    yaxis_title: str,
    percent_decimals: int = 2,
    signed: bool = True,
) -> str:
    valid = quality_df.dropna(subset=[metric_col]).copy()
    if valid.empty:
        return _warning_html(f"No valid values for {title}.")
    valid = valid.sort_values(metric_col)
    y = valid[metric_col] * 100
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]
    text = [
        (f"{v:+.{percent_decimals}f}%" if signed else f"{v:.{percent_decimals}f}%")
        for v in y
    ]
    fig = go.Figure(go.Bar(
        x=valid["ticker"],
        y=y,
        marker_color=colors,
        text=text,
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title=yaxis_title,
        yaxis_ticksuffix="%",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_ticker_bucket_sample_counts(bucket_df: pd.DataFrame) -> str:
    if bucket_df.empty:
        return _warning_html("Per-ticker bucket sample counts are unavailable.")

    tickers = sorted(bucket_df["ticker"].unique())
    fig = go.Figure()
    for bucket in _BUCKET_LABELS:
        subset = (
            bucket_df[bucket_df["signal_bucket"] == bucket]
            .set_index("ticker")
            .reindex(tickers)
        )
        fig.add_trace(go.Bar(
            x=tickers,
            y=subset["valid_20d_rows"].fillna(0),
            name=bucket,
        ))
    fig.update_layout(
        title="Valid 20D Rows by Ticker and Signal Bucket",
        xaxis_title="Ticker",
        yaxis_title="Valid 20D Rows",
        barmode="stack",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_group_bucket_heatmap(group_df: pd.DataFrame) -> str:
    if group_df.empty:
        return _warning_html("Group bucket data is unavailable.")

    groups = sorted(group_df["group"].unique())
    values = (
        group_df.pivot(index="group", columns="signal_bucket", values="avg_20d_return")
        .reindex(index=groups, columns=_BUCKET_LABELS)
    )
    counts = (
        group_df.pivot(index="group", columns="signal_bucket", values="valid_20d_rows")
        .reindex(index=groups, columns=_BUCKET_LABELS)
    )

    text = []
    for group in groups:
        row = []
        for bucket in _BUCKET_LABELS:
            val = values.loc[group, bucket]
            n = counts.loc[group, bucket]
            if pd.isna(val):
                row.append(f"n={int(n) if not pd.isna(n) else 0}")
            else:
                row.append(f"{val * 100:+.1f}%<br>n={int(n)}")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        x=_BUCKET_LABELS,
        y=groups,
        z=values.to_numpy(dtype=float) * 100,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Avg 20D %"),
    ))
    fig.update_layout(
        title="Avg 20D Return by Group and Signal Bucket",
        xaxis_title="Signal Bucket",
        yaxis_title="Group",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_group_high_signal_metric(
    group_df: pd.DataFrame,
    metric_col: str,
    title: str,
    yaxis_title: str,
    signed: bool = True,
) -> str:
    valid = group_df.dropna(subset=[metric_col]).copy()
    if valid.empty:
        return _warning_html(f"No valid values for {title}.")
    valid = valid.sort_values(metric_col)
    y = valid[metric_col] * 100
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]
    text = [f"{v:+.2f}%" if signed else f"{v:.1f}%" for v in y]
    fig = go.Figure(go.Bar(
        x=valid["group"],
        y=y,
        marker_color=colors,
        text=text,
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title=yaxis_title,
        yaxis_ticksuffix="%",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Interpretation ────────────────────────────────────────────────────────────

def _chart_sector_bucket_heatmap(sector_bucket_df: pd.DataFrame) -> str:
    if sector_bucket_df.empty:
        return _warning_html("Sector bucket data is unavailable.")

    sectors = sorted(sector_bucket_df["sector"].unique())
    values = (
        sector_bucket_df.pivot(index="sector", columns="signal_bucket", values="avg_20d_return")
        .reindex(index=sectors, columns=_BUCKET_LABELS)
    )
    counts = (
        sector_bucket_df.pivot(index="sector", columns="signal_bucket", values="valid_20d_rows")
        .reindex(index=sectors, columns=_BUCKET_LABELS)
    )
    text = []
    for sector in sectors:
        row = []
        for bucket in _BUCKET_LABELS:
            val = values.loc[sector, bucket]
            n = counts.loc[sector, bucket]
            if pd.isna(val):
                row.append(f"n={int(n) if not pd.isna(n) else 0}")
            else:
                row.append(f"{val * 100:+.1f}%<br>n={int(n)}")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        x=_BUCKET_LABELS,
        y=sectors,
        z=values.to_numpy(dtype=float) * 100,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Avg 20D %"),
    ))
    fig.update_layout(
        title="Avg 20D Return by Sector and Signal Bucket",
        xaxis_title="Signal Bucket",
        yaxis_title="Sector",
        height=520,
        template="plotly_white",
        font=dict(family="system-ui, sans-serif", size=12),
        margin=dict(l=110, r=30, t=50, b=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_sector_high_signal_metric(
    sector_df: pd.DataFrame,
    metric_col: str,
    title: str,
    yaxis_title: str,
    signed: bool = True,
) -> str:
    if sector_df.empty or metric_col not in sector_df.columns:
        return _warning_html(f"No valid values for {title}.")
    valid = sector_df.dropna(subset=[metric_col]).copy()
    if valid.empty:
        return _warning_html(f"No valid values for {title}.")
    valid = valid.sort_values(metric_col)
    y = valid[metric_col] * 100
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]
    text = [f"{v:+.2f}%" if signed else f"{v:.1f}%" for v in y]
    fig = go.Figure(go.Bar(
        x=valid["sector"],
        y=y,
        marker_color=colors,
        text=text,
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Sector",
        yaxis_title=yaxis_title,
        yaxis_ticksuffix="%",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_sector_market_comparison_metric(
    comparison_df: pd.DataFrame,
    metric_col: str,
    title: str,
    yaxis_title: str,
    signed: bool = True,
) -> str:
    if comparison_df.empty or metric_col not in comparison_df.columns:
        return _warning_html("Sector market-regime comparison is unavailable.")

    sectors = sorted(comparison_df["sector"].unique())
    conditions = ["High signal + market healthy", "High signal + market weak"]
    fig = go.Figure()
    for condition in conditions:
        data = (
            comparison_df[comparison_df["condition"] == condition]
            .set_index("sector")
            .reindex(sectors)
        )
        y = data[metric_col] * 100
        fig.add_trace(go.Bar(
            x=sectors,
            y=y,
            name=condition.replace("High signal + ", ""),
            text=[
                (f"{v:+.2f}%" if signed else f"{v:.1f}%")
                if not pd.isna(v)
                else ""
                for v in y
            ],
            textposition="outside",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Sector",
        yaxis_title=yaxis_title,
        yaxis_ticksuffix="%",
        barmode="group",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_broad_group_regime_heatmap(group_df: pd.DataFrame) -> str:
    if group_df.empty:
        return _warning_html("Broad group regime analysis is unavailable.")

    conditions = ["High signal only", "Market healthy", "Market weak"]
    groups = list(_BROAD_SECTOR_GROUPS.keys())
    values = (
        group_df.pivot(index="broad_group", columns="condition", values="avg_20d_return")
        .reindex(index=groups, columns=conditions)
    )
    counts = (
        group_df.pivot(index="broad_group", columns="condition", values="valid_20d_rows")
        .reindex(index=groups, columns=conditions)
    )
    text = []
    for group in groups:
        row = []
        for condition in conditions:
            val = values.loc[group, condition]
            n = counts.loc[group, condition]
            if pd.isna(val):
                row.append(f"n={int(n) if not pd.isna(n) else 0}")
            else:
                row.append(f"{val * 100:+.1f}%<br>n={int(n)}")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        x=conditions,
        y=groups,
        z=values.to_numpy(dtype=float) * 100,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Avg 20D %"),
    ))
    fig.update_layout(
        title="Broad Group Avg 20D Return by Market Condition",
        xaxis_title="Market Condition",
        yaxis_title="Broad Group",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_personality_high_low_metric(
    high_low_df: pd.DataFrame,
    metric_col: str,
    title: str,
    yaxis_title: str,
    signed: bool = True,
) -> str:
    if high_low_df.empty or metric_col not in high_low_df.columns:
        return _warning_html(f"No valid values for {title}.")
    valid = high_low_df.dropna(subset=[metric_col]).copy()
    if valid.empty:
        return _warning_html(f"No valid values for {title}.")
    valid["stock_personality"] = pd.Categorical(
        valid["stock_personality"], categories=_PERSONALITY_LABELS, ordered=True
    )
    valid = valid.sort_values("stock_personality")
    y = valid[metric_col] * 100
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]
    text = [f"{v:+.2f}%" if signed else f"{v:.1f}%" for v in y]
    fig = go.Figure(go.Bar(
        x=valid["stock_personality"].astype(str),
        y=y,
        marker_color=colors,
        text=text,
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Stock Personality",
        yaxis_title=yaxis_title,
        yaxis_ticksuffix="%",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_personality_bucket_heatmap(bucket_df: pd.DataFrame) -> str:
    if bucket_df.empty:
        return _warning_html("Personality bucket data is unavailable.")

    labels = [label for label in _PERSONALITY_LABELS if label in set(bucket_df["stock_personality"])]
    values = (
        bucket_df.pivot(
            index="stock_personality",
            columns="signal_bucket",
            values="avg_excess_vs_sector_etf_20d",
        )
        .reindex(index=labels, columns=_BUCKET_LABELS)
    )
    counts = (
        bucket_df.pivot(
            index="stock_personality",
            columns="signal_bucket",
            values="valid_20d_rows",
        )
        .reindex(index=labels, columns=_BUCKET_LABELS)
    )

    text = []
    for label in labels:
        row = []
        for bucket in _BUCKET_LABELS:
            val = values.loc[label, bucket]
            n = counts.loc[label, bucket]
            if pd.isna(val):
                row.append(f"n={int(n) if not pd.isna(n) else 0}")
            else:
                row.append(f"{val * 100:+.1f}%<br>n={int(n)}")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        x=_BUCKET_LABELS,
        y=labels,
        z=values.to_numpy(dtype=float) * 100,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Ex SecETF 20D %"),
    ))
    fig.update_layout(
        title="Avg Excess vs Sector ETF by Personality and Signal Bucket",
        xaxis_title="Signal Bucket",
        yaxis_title="Stock Personality",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_personality_raw_vs_excess(signal_df: pd.DataFrame) -> str:
    if signal_df.empty:
        return _warning_html("Personality raw-vs-excess chart is unavailable.")

    high = signal_df[signal_df["condition"] == "High signal >= 40%"].copy()
    high = high.dropna(subset=["raw_avg_20d_return", "avg_excess_vs_sector_etf_20d"])
    if high.empty:
        return _warning_html("No high-signal personality rows for raw-vs-excess chart.")
    high["stock_personality"] = pd.Categorical(
        high["stock_personality"], categories=_PERSONALITY_LABELS, ordered=True
    )
    high = high.sort_values("stock_personality")
    labels = high["stock_personality"].astype(str).tolist()

    fig = go.Figure()
    for col, name, color in [
        ("raw_avg_20d_return", "Raw Avg 20D", "#4c78a8"),
        ("avg_excess_vs_sector_etf_20d", "Excess vs Sector ETF", "#f58518"),
    ]:
        y = high[col] * 100
        fig.add_trace(go.Bar(
            x=labels,
            y=y,
            name=name,
            marker_color=color,
            text=[f"{v:+.2f}%" for v in y],
            textposition="outside",
        ))
    fig.update_layout(
        title="High-Signal Raw Return vs Sector-ETF Excess by Personality",
        xaxis_title="Stock Personality",
        yaxis_title="20D Return (%)",
        yaxis_ticksuffix="%",
        barmode="group",
        **_CHART_LAYOUT,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _generate_interpretation(
    df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    ticker_df: pd.DataFrame,
) -> str:
    items: list[str] = []

    # 1. Monotonic trend in bucket returns
    returns_20 = bucket_df["avg_20d_return"].dropna()
    valid_labels = [l for l in _BUCKET_LABELS if l in returns_20.index]
    if len(valid_labels) >= 3:
        indices = [_BUCKET_LABELS.index(l) for l in valid_labels]
        vals = [returns_20[l] for l in valid_labels]
        corr = float(np.corrcoef(indices, vals)[0, 1])
        if corr > 0.6:
            verdict = '<span style="color:#2ca02c">progressively better</span>'
            note = "consistent with predictive value"
        elif corr > 0.2:
            verdict = '<span style="color:#f58518">weakly positive trend</span>'
            note = "mixed — more tickers or a longer history would help"
        else:
            verdict = '<span style="color:#d62728">no clear improvement</span>'
            note = "signal may lack predictive value at this horizon for this universe"
        items.append(
            f"<li><strong>Signal monotonicity (20D):</strong> Higher buckets show "
            f"{verdict} avg returns (rank correlation: {corr:+.2f}). {note.capitalize()}.</li>"
        )

    # 2. Best bucket
    if not bucket_df["avg_20d_return"].dropna().empty:
        best_bkt = bucket_df["avg_20d_return"].idxmax()
        best_val = bucket_df.loc[best_bkt, "avg_20d_return"] * 100
        best_n = int(bucket_df.loc[best_bkt, "row_count"])
        items.append(
            f"<li><strong>Best bucket (avg 20D):</strong> "
            f"<code>{_html.escape(str(best_bkt))}</code> → {best_val:+.2f}% "
            f"(n={best_n:,} rows).</li>"
        )

    # 3. Ticker performance at high signal
    high = df[df["signal"] >= 0.10]
    if not high.empty:
        by_t = high.groupby("ticker")["future_20d_return"].mean().dropna()
        if len(by_t) >= 2:
            best_t, worst_t = by_t.idxmax(), by_t.idxmin()
            items.append(
                f"<li><strong>Ticker performance when signal ≥ 10%:</strong> "
                f"Best: <strong>{best_t}</strong> ({by_t[best_t]*100:+.2f}%), "
                f"Worst: <strong>{worst_t}</strong> ({by_t[worst_t]*100:+.2f}%).</li>"
            )
        hit_10 = positive_rate(high["future_10d_return"]) * 100
        hit_20 = positive_rate(high["future_20d_return"]) * 100
        valid_10 = _valid_count(high["future_10d_return"])
        valid_20 = _valid_count(high["future_20d_return"])
        items.append(
            f"<li><strong>Hit rate when signal ≥ 10%:</strong> "
            f"{hit_10:.1f}% positive 10D, {hit_20:.1f}% positive 20D "
            f"(rows={len(high):,}; valid 10D={valid_10:,}, valid 20D={valid_20:,}).</li>"
        )
    else:
        items.append("<li>No rows with signal ≥ 10%. Consider reviewing the universe or parameters.</li>")

    # 4. Filter combination findings
    has_market = "market_healthy_either" in df.columns
    has_stock = "stock_trend_healthy" in df.columns
    if has_market or has_stock:
        combined = build_combined_conditions_table(df).reset_index()
        count_col = "valid_20d_rows" if "valid_20d_rows" in combined.columns else "rows"
        candidates = combined[combined[count_col] >= 30].dropna(subset=["avg_20d"])
        if not candidates.empty:
            best = candidates.loc[candidates["avg_20d"].idxmax()]
            # Compare against Markov-only at the same threshold
            baseline_rows = combined[
                (combined["threshold"] == best["threshold"]) &
                (combined["condition"] == "Markov only")
            ]
            improvement_str = ""
            if not baseline_rows.empty:
                base_val = baseline_rows.iloc[0]["avg_20d"]
                if not pd.isna(base_val):
                    delta = (best["avg_20d"] - base_val) * 100
                    improvement_str = f", {delta:+.2f}% vs Markov-only at same threshold"
            valid_str = ""
            if "valid_20d_rows" in best:
                valid_str = f"; valid 20D={int(best['valid_20d_rows']):,}"
            items.append(
                f"<li><strong>Best filter combination (20D):</strong> "
                f"Signal ≥ {best['threshold']}, {best['condition']}: "
                f"avg 20D = {best['avg_20d']*100:+.2f}%{improvement_str} "
                f"(rows={int(best['rows']):,}{valid_str}).</li>"
            )

    # 5. Standing disclaimer
    items.append(
        "<li style='color:#888;font-size:0.9em;margin-top:6px'>"
        "<strong>Reminder:</strong> This is <em>signal validation</em>, not a trading backtest. "
        "Forward returns are raw price returns with no entries, exits, commissions, or position sizing. "
        "A positive correlation does not guarantee profitability when traded.</li>"
    )

    return "<ul style='padding-left:20px;line-height:1.8'>" + "\n".join(items) + "</ul>"


# ── Table formatters ─────────────────────────────────────────────────────────

def _format_ticker_table(df: pd.DataFrame) -> str:
    fmt = df.copy()
    fmt["first_date"] = pd.to_datetime(fmt["first_date"]).dt.strftime("%Y-%m-%d")
    fmt["last_date"] = pd.to_datetime(fmt["last_date"]).dt.strftime("%Y-%m-%d")
    for col in ["row_count", "valid_5d_rows", "valid_10d_rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].map("{:,}".format)
    for col in ["avg_signal", "min_signal", "max_signal",
                "avg_5d_return", "avg_10d_return", "avg_20d_return",
                "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
                "avg_excess_vs_sector_etf_20d",
                "median_excess_vs_sector_etf_20d"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["pos_10d_rate", "pos_20d_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))
    ordered = [
        "first_date", "last_date", "row_count",
        "valid_5d_rows", "valid_10d_rows", "valid_20d_rows",
        "avg_signal", "min_signal", "max_signal",
        "avg_5d_return", "avg_10d_return", "avg_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
        "excess_vs_sector_etf_20d_win_rate",
        "pos_10d_rate", "pos_20d_rate",
    ]
    labels = {
        "first_date": "First Date", "last_date": "Last Date", "row_count": "Rows",
        "valid_5d_rows": "Valid 5D", "valid_10d_rows": "Valid 10D",
        "valid_20d_rows": "Valid 20D", "avg_signal": "Avg Signal",
        "min_signal": "Min Signal", "max_signal": "Max Signal",
        "avg_5d_return": "Avg 5D Ret", "avg_10d_return": "Avg 10D Ret",
        "avg_20d_return": "Avg Raw 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
        "pos_10d_rate": "+10D Rate", "pos_20d_rate": "+20D Rate",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    return fmt.to_html(escape=False, classes="data-table", border=0)


def _format_bucket_table(df: pd.DataFrame) -> str:
    fmt = df.copy()
    for col in ["row_count", "valid_5d_rows", "valid_10d_rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(
            lambda x: f"{int(x):,}" if not pd.isna(x) else "0"
        )
    fmt["avg_sample_count"] = fmt["avg_sample_count"].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "—"
    )
    for col in ["avg_5d_return", "avg_10d_return", "median_10d_return",
                "avg_20d_return", "median_20d_return",
                "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
                "avg_excess_vs_sector_etf_20d",
                "median_excess_vs_sector_etf_20d"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["positive_10d_rate", "positive_20d_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))
    ordered = [
        "row_count", "valid_5d_rows", "valid_10d_rows", "valid_20d_rows",
        "avg_5d_return", "avg_10d_return", "median_10d_return",
        "positive_10d_rate", "avg_20d_return", "median_20d_return",
        "positive_20d_rate", "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d", "median_excess_vs_sector_etf_20d",
        "excess_vs_sector_etf_20d_win_rate", "avg_sample_count",
    ]
    labels = {
        "row_count": "Rows", "valid_5d_rows": "Valid 5D",
        "valid_10d_rows": "Valid 10D", "valid_20d_rows": "Valid 20D",
        "avg_5d_return": "Avg 5D Ret", "avg_10d_return": "Avg 10D Ret",
        "median_10d_return": "Med 10D Ret", "positive_10d_rate": "+10D Rate",
        "avg_20d_return": "Avg Raw 20D", "median_20d_return": "Med 20D Ret",
        "positive_20d_rate": "+20D Rate",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
        "avg_sample_count": "Avg Sample N",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    return fmt.to_html(escape=False, classes="data-table", border=0)


def _table_html_grouped(flat: pd.DataFrame, group_col: str) -> str:
    """
    Build an HTML table that draws a thick top border between each group.
    group_col must be a column in flat (after reset_index).
    """
    cols = flat.columns.tolist()
    headers = "".join(f"<th>{_html.escape(str(c))}</th>" for c in cols)

    prev_group: Any = object()  # sentinel — never equal to a real value
    body_rows: list[str] = []
    for _, row in flat.iterrows():
        current = row.get(group_col)
        is_new_group = current != prev_group
        prev_group = current
        top = "border-top:2.5px solid #b0b8c6" if is_new_group else "border-top:1px solid #eef0f3"
        cells = "".join(f'<td style="{top}">{v}</td>' for v in row.values)
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        f'<table class="data-table" border="0">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table>'
    )


def _format_comparison_table(df: pd.DataFrame) -> str:
    """Format a filter comparison table (MultiIndex: threshold × condition) for HTML."""
    flat = df.reset_index().copy()

    flat["rows"] = flat["rows"].apply(
        lambda x: f"{int(x):,}" if not pd.isna(x) else "—"
    )
    for d in [5, 10, 20]:
        col = f"valid_{d}d_rows"
        if col in flat.columns:
            flat[col] = flat[col].apply(
                lambda x: f"{int(x):,}" if not pd.isna(x) else "0"
            )
    for prefix in ["avg", "med"]:
        for d in [5, 10, 20]:
            col = f"{prefix}_{d}d"
            if col in flat.columns:
                flat[col] = flat[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in [
        "avg_excess_vs_spy_20d",
        "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in flat.columns:
            flat[col] = flat[col].apply(lambda x: _pct(x, 2, colored=True))
    for d in [5, 10, 20]:
        col = f"win_{d}d"
        if col in flat.columns:
            flat[col] = flat[col].apply(lambda x: _pct(x, 1, colored=False))
    if "excess_vs_sector_etf_20d_win_rate" in flat.columns:
        flat["excess_vs_sector_etf_20d_win_rate"] = flat[
            "excess_vs_sector_etf_20d_win_rate"
        ].apply(lambda x: _pct(x, 1, colored=False))

    col_display = {
        "threshold": "Threshold", "condition": "Condition", "rows": "Rows",
        "valid_5d_rows": "Valid 5D",
        "valid_10d_rows": "Valid 10D",
        "valid_20d_rows": "Valid 20D",
        "avg_5d": "Avg 5D", "med_5d": "Med 5D", "win_5d": "Win 5D",
        "avg_10d": "Avg 10D", "med_10d": "Med 10D", "win_10d": "Win 10D",
        "avg_20d": "Avg 20D", "med_20d": "Med 20D", "win_20d": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    }
    present = [c for c in col_display if c in flat.columns]
    flat = flat[present].rename(columns={c: col_display[c] for c in present})

    return _table_html_grouped(flat, "Threshold")


def _format_per_ticker_bucket_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Per-ticker bucket table is unavailable.")

    fmt = df.copy()
    fmt["signal_bucket"] = pd.Categorical(
        fmt["signal_bucket"], categories=_BUCKET_LABELS, ordered=True
    )
    fmt = fmt.sort_values(["ticker", "signal_bucket"])
    for col in ["rows", "valid_5d_rows", "valid_10d_rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_5d_return", "median_5d_return",
        "avg_10d_return", "median_10d_return",
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["5d_win_rate", "10d_win_rate", "20d_win_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))
    fmt["avg_sample_count"] = fmt["avg_sample_count"].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "-"
    )

    fmt = fmt[
        [
            "ticker", "signal_bucket", "rows",
            "valid_5d_rows", "valid_10d_rows", "valid_20d_rows",
            "avg_5d_return", "median_5d_return", "5d_win_rate",
            "avg_10d_return", "median_10d_return", "10d_win_rate",
            "avg_20d_return", "median_20d_return", "20d_win_rate",
            "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
            "avg_excess_vs_sector_etf_20d",
            "median_excess_vs_sector_etf_20d",
            "excess_vs_sector_etf_20d_win_rate",
            "avg_sample_count",
        ]
    ].rename(columns={
        "ticker": "Ticker",
        "signal_bucket": "Signal Bucket",
        "rows": "Rows",
        "valid_5d_rows": "Valid 5D",
        "valid_10d_rows": "Valid 10D",
        "valid_20d_rows": "Valid 20D",
        "avg_5d_return": "Avg 5D",
        "median_5d_return": "Med 5D",
        "5d_win_rate": "Win 5D",
        "avg_10d_return": "Avg 10D",
        "median_10d_return": "Med 10D",
        "10d_win_rate": "Win 10D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
        "avg_sample_count": "Avg Sample N",
    })
    return _table_html_grouped(fmt, "Ticker")


def _format_ticker_quality_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Per-ticker quality summary is unavailable.")

    fmt = df.copy().sort_values("ticker")
    for col in [
        "total_rows", "high_signal_rows", "high_signal_valid_20d_rows",
        "low_signal_rows", "low_signal_valid_20d_rows", "market_high_signal_rows",
        "market_high_signal_valid_20d_rows",
    ]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "high_signal_avg_20d_return", "high_signal_median_20d_return",
        "high_signal_avg_excess_vs_spy_20d",
        "high_signal_avg_excess_vs_qqq_20d",
        "high_signal_avg_excess_vs_sector_etf_20d",
        "high_signal_median_excess_vs_sector_etf_20d",
        "low_signal_avg_20d_return", "low_signal_median_20d_return",
        "low_signal_avg_excess_vs_spy_20d",
        "low_signal_avg_excess_vs_qqq_20d",
        "low_signal_avg_excess_vs_sector_etf_20d",
        "low_signal_median_excess_vs_sector_etf_20d",
        "directional_spread", "market_high_signal_avg_20d_return",
        "market_high_signal_median_20d_return",
        "market_high_signal_avg_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in [
        "high_signal_20d_win_rate", "low_signal_20d_win_rate",
        "market_high_signal_20d_win_rate",
        "high_signal_excess_vs_sector_etf_20d_win_rate",
        "low_signal_excess_vs_sector_etf_20d_win_rate",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    fmt = fmt[
        [
            "ticker", "total_rows",
            "high_signal_rows", "high_signal_valid_20d_rows",
            "high_signal_avg_20d_return", "high_signal_median_20d_return",
            "high_signal_20d_win_rate",
            "high_signal_avg_excess_vs_spy_20d",
            "high_signal_avg_excess_vs_qqq_20d",
            "high_signal_avg_excess_vs_sector_etf_20d",
            "high_signal_median_excess_vs_sector_etf_20d",
            "high_signal_excess_vs_sector_etf_20d_win_rate",
            "low_signal_rows", "low_signal_valid_20d_rows",
            "low_signal_avg_20d_return", "low_signal_median_20d_return",
            "low_signal_20d_win_rate",
            "low_signal_avg_excess_vs_spy_20d",
            "low_signal_avg_excess_vs_qqq_20d",
            "low_signal_avg_excess_vs_sector_etf_20d",
            "low_signal_median_excess_vs_sector_etf_20d",
            "low_signal_excess_vs_sector_etf_20d_win_rate",
            "directional_spread",
            "market_high_signal_rows", "market_high_signal_valid_20d_rows",
            "market_high_signal_avg_20d_return",
            "market_high_signal_median_20d_return",
            "market_high_signal_20d_win_rate",
            "market_high_signal_avg_excess_vs_sector_etf_20d",
            "verdict",
        ]
    ].rename(columns={
        "ticker": "Ticker",
        "total_rows": "Rows",
        "high_signal_rows": "High Rows",
        "high_signal_valid_20d_rows": "High Valid 20D",
        "high_signal_avg_20d_return": "High Avg 20D",
        "high_signal_median_20d_return": "High Med 20D",
        "high_signal_20d_win_rate": "High Win 20D",
        "high_signal_avg_excess_vs_spy_20d": "High Ex SPY 20D",
        "high_signal_avg_excess_vs_qqq_20d": "High Ex QQQ 20D",
        "high_signal_avg_excess_vs_sector_etf_20d": "High Ex SecETF 20D",
        "high_signal_median_excess_vs_sector_etf_20d": "High Med Ex SecETF 20D",
        "high_signal_excess_vs_sector_etf_20d_win_rate": "High Ex Win 20D",
        "low_signal_rows": "Low Rows",
        "low_signal_valid_20d_rows": "Low Valid 20D",
        "low_signal_avg_20d_return": "Low Avg 20D",
        "low_signal_median_20d_return": "Low Med 20D",
        "low_signal_20d_win_rate": "Low Win 20D",
        "low_signal_avg_excess_vs_spy_20d": "Low Ex SPY 20D",
        "low_signal_avg_excess_vs_qqq_20d": "Low Ex QQQ 20D",
        "low_signal_avg_excess_vs_sector_etf_20d": "Low Ex SecETF 20D",
        "low_signal_median_excess_vs_sector_etf_20d": "Low Med Ex SecETF 20D",
        "low_signal_excess_vs_sector_etf_20d_win_rate": "Low Ex Win 20D",
        "directional_spread": "Spread",
        "market_high_signal_rows": "Mkt High Rows",
        "market_high_signal_valid_20d_rows": "Mkt High Valid 20D",
        "market_high_signal_avg_20d_return": "Mkt High Avg 20D",
        "market_high_signal_median_20d_return": "Mkt High Med 20D",
        "market_high_signal_20d_win_rate": "Mkt High Win 20D",
        "market_high_signal_avg_excess_vs_sector_etf_20d": "Mkt High Ex SecETF 20D",
        "verdict": "Verdict",
    })
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_market_ticker_comparison_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Market-filtered per-ticker comparison is unavailable.")

    fmt = df.copy()
    condition_order = [
        "High signal only",
        "High signal + market healthy",
        "High signal + market weak",
    ]
    fmt["condition"] = pd.Categorical(fmt["condition"], condition_order, ordered=True)
    fmt = fmt.sort_values(["ticker", "condition"])
    for col in ["rows", "valid_10d_rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_10d_return", "median_10d_return",
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["10d_win_rate", "20d_win_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    fmt = fmt[
        [
            "ticker", "condition", "rows",
            "valid_10d_rows", "avg_10d_return", "median_10d_return", "10d_win_rate",
            "valid_20d_rows", "avg_20d_return", "median_20d_return", "20d_win_rate",
            "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
            "avg_excess_vs_sector_etf_20d",
            "median_excess_vs_sector_etf_20d",
            "excess_vs_sector_etf_20d_win_rate",
        ]
    ].rename(columns={
        "ticker": "Ticker",
        "condition": "Condition",
        "rows": "Rows",
        "valid_10d_rows": "Valid 10D",
        "avg_10d_return": "Avg 10D",
        "median_10d_return": "Med 10D",
        "10d_win_rate": "Win 10D",
        "valid_20d_rows": "Valid 20D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    })
    return _table_html_grouped(fmt, "Ticker")


def _format_group_bucket_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Group-level bucket table is unavailable.")

    fmt = df.copy()
    fmt["signal_bucket"] = pd.Categorical(
        fmt["signal_bucket"], categories=_BUCKET_LABELS, ordered=True
    )
    fmt = fmt.sort_values(["group", "signal_bucket"])
    for col in ["rows", "valid_10d_rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_10d_return", "median_10d_return",
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["10d_win_rate", "20d_win_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    fmt = fmt[
        [
            "group", "signal_bucket", "rows",
            "valid_10d_rows", "avg_10d_return", "median_10d_return", "10d_win_rate",
            "valid_20d_rows", "avg_20d_return", "median_20d_return", "20d_win_rate",
            "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
            "avg_excess_vs_sector_etf_20d",
            "median_excess_vs_sector_etf_20d",
            "excess_vs_sector_etf_20d_win_rate",
        ]
    ].rename(columns={
        "group": "Group",
        "signal_bucket": "Signal Bucket",
        "rows": "Rows",
        "valid_10d_rows": "Valid 10D",
        "avg_10d_return": "Avg 10D",
        "median_10d_return": "Med 10D",
        "10d_win_rate": "Win 10D",
        "valid_20d_rows": "Valid 20D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    })
    return _table_html_grouped(fmt, "Group")


# ── CSS ───────────────────────────────────────────────────────────────────────

def _format_sector_summary_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Sector summary is unavailable.")

    fmt = df.copy().sort_values("sector")
    for col in [
        "rows", "valid_20d_rows", "high_signal_rows",
        "high_signal_valid_10d_rows", "high_signal_valid_20d_rows",
    ]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_signal", "high_signal_avg_10d_return",
        "high_signal_median_10d_return", "high_signal_avg_20d_return",
        "high_signal_median_20d_return",
        "high_signal_avg_excess_vs_spy_20d",
        "high_signal_avg_excess_vs_qqq_20d",
        "high_signal_avg_excess_vs_sector_etf_20d",
        "high_signal_median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in [
        "high_signal_10d_win_rate", "high_signal_20d_win_rate",
        "high_signal_excess_vs_sector_etf_20d_win_rate",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    fmt = fmt[
        [
            "sector", "rows", "valid_20d_rows", "avg_signal",
            "high_signal_rows", "high_signal_valid_10d_rows",
            "high_signal_avg_10d_return", "high_signal_median_10d_return",
            "high_signal_10d_win_rate", "high_signal_valid_20d_rows",
            "high_signal_avg_20d_return", "high_signal_median_20d_return",
            "high_signal_20d_win_rate",
            "high_signal_avg_excess_vs_spy_20d",
            "high_signal_avg_excess_vs_qqq_20d",
            "high_signal_avg_excess_vs_sector_etf_20d",
            "high_signal_median_excess_vs_sector_etf_20d",
            "high_signal_excess_vs_sector_etf_20d_win_rate",
        ]
    ].rename(columns={
        "sector": "Sector",
        "rows": "Rows",
        "valid_20d_rows": "Valid 20D",
        "avg_signal": "Avg Signal",
        "high_signal_rows": "High Rows",
        "high_signal_valid_10d_rows": "High Valid 10D",
        "high_signal_avg_10d_return": "High Avg 10D",
        "high_signal_median_10d_return": "High Med 10D",
        "high_signal_10d_win_rate": "High Win 10D",
        "high_signal_valid_20d_rows": "High Valid 20D",
        "high_signal_avg_20d_return": "High Avg 20D",
        "high_signal_median_20d_return": "High Med 20D",
        "high_signal_20d_win_rate": "High Win 20D",
        "high_signal_avg_excess_vs_spy_20d": "High Ex SPY 20D",
        "high_signal_avg_excess_vs_qqq_20d": "High Ex QQQ 20D",
        "high_signal_avg_excess_vs_sector_etf_20d": "High Ex SecETF 20D",
        "high_signal_median_excess_vs_sector_etf_20d": "High Med Ex SecETF 20D",
        "high_signal_excess_vs_sector_etf_20d_win_rate": "High Ex Win 20D",
    })
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_sector_market_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Sector market-regime comparison is unavailable.")

    fmt = df.copy()
    order = [
        "High signal only",
        "High signal + market healthy",
        "High signal + market weak",
    ]
    fmt["condition"] = pd.Categorical(fmt["condition"], order, ordered=True)
    fmt = fmt.sort_values(["sector", "condition"])
    for col in ["rows", "valid_10d_rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_10d_return", "median_10d_return",
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["10d_win_rate", "20d_win_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    fmt = fmt[
        [
            "sector", "condition", "rows", "valid_10d_rows",
            "avg_10d_return", "median_10d_return", "10d_win_rate",
            "valid_20d_rows", "avg_20d_return", "median_20d_return",
            "20d_win_rate",
            "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
            "avg_excess_vs_sector_etf_20d",
            "median_excess_vs_sector_etf_20d",
            "excess_vs_sector_etf_20d_win_rate",
        ]
    ].rename(columns={
        "sector": "Sector",
        "condition": "Condition",
        "rows": "Rows",
        "valid_10d_rows": "Valid 10D",
        "avg_10d_return": "Avg 10D",
        "median_10d_return": "Med 10D",
        "10d_win_rate": "Win 10D",
        "valid_20d_rows": "Valid 20D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    })
    return _table_html_grouped(fmt, "Sector")


def _format_sector_etf_vs_stocks_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Sector ETF vs stocks comparison is unavailable.")

    fmt = df.copy().sort_values(["sector", "condition", "instrument"])
    for col in ["rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["20d_win_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))
    fmt = fmt[
        [
            "sector", "condition", "instrument", "rows", "valid_20d_rows",
            "avg_20d_return", "median_20d_return", "20d_win_rate",
            "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
            "avg_excess_vs_sector_etf_20d",
            "median_excess_vs_sector_etf_20d",
            "excess_vs_sector_etf_20d_win_rate",
        ]
    ].rename(columns={
        "sector": "Sector",
        "condition": "Condition",
        "instrument": "Instrument",
        "rows": "Rows",
        "valid_20d_rows": "Valid 20D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    })
    return _table_html_grouped(fmt, "Sector")


def _format_broad_group_regime_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Broad group regime analysis is unavailable.")

    fmt = df.copy()
    condition_order = ["High signal only", "Market healthy", "Market weak"]
    fmt["condition"] = pd.Categorical(fmt["condition"], condition_order, ordered=True)
    fmt = fmt.sort_values(["broad_group", "condition"])
    for col in ["rows", "valid_20d_rows"]:
        fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in ["20d_win_rate", "excess_vs_sector_etf_20d_win_rate"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))
    fmt = fmt[
        [
            "broad_group", "condition", "rows", "valid_20d_rows",
            "avg_20d_return", "median_20d_return", "20d_win_rate",
            "avg_excess_vs_spy_20d", "avg_excess_vs_qqq_20d",
            "avg_excess_vs_sector_etf_20d",
            "median_excess_vs_sector_etf_20d",
            "excess_vs_sector_etf_20d_win_rate",
        ]
    ].rename(columns={
        "broad_group": "Broad Group",
        "condition": "Condition",
        "rows": "Rows",
        "valid_20d_rows": "Valid 20D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_spy_20d": "Avg Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Avg Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Avg Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    })
    return _table_html_grouped(fmt, "Broad Group")


def _format_edge_summary_table(
    df: pd.DataFrame,
    id_cols: list[str],
    group_col: str | None = None,
) -> str:
    if df.empty:
        return _warning_html("Baseline-adjusted edge table is unavailable.")

    fmt = df.copy()
    for col in ["rows", "valid_20d_rows", "valid_excess_vs_sector_etf_20d_rows"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)
    pct_cols = [
        "raw_avg_20d_return",
        "raw_median_20d_return",
        "universe_baseline_avg_20d_return",
        "excess_vs_universe_avg_20d",
        "avg_excess_vs_spy_20d",
        "avg_excess_vs_qqq_20d",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
    ]
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    if "excess_vs_sector_etf_20d_win_rate" in fmt.columns:
        fmt["excess_vs_sector_etf_20d_win_rate"] = fmt[
            "excess_vs_sector_etf_20d_win_rate"
        ].apply(lambda x: _pct(x, 1, colored=False))
    if "raw_20d_win_rate" in fmt.columns:
        fmt["raw_20d_win_rate"] = fmt["raw_20d_win_rate"].apply(
            lambda x: _pct(x, 1, colored=False)
        )

    ordered = id_cols + [
        "rows", "valid_20d_rows",
        "raw_avg_20d_return", "raw_median_20d_return", "raw_20d_win_rate",
        "universe_baseline_avg_20d_return",
        "excess_vs_universe_avg_20d", "avg_excess_vs_spy_20d",
        "avg_excess_vs_qqq_20d", "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
        "valid_excess_vs_sector_etf_20d_rows",
        "excess_vs_sector_etf_20d_win_rate",
    ]
    labels = {
        "signal_bucket": "Signal Bucket",
        "condition": "Condition",
        "sector": "Sector",
        "stock_personality": "Personality",
        "stock_personality_group": "Personality Group",
        "rows": "Rows",
        "valid_20d_rows": "Valid 20D",
        "raw_avg_20d_return": "Raw Avg 20D",
        "raw_median_20d_return": "Raw Med 20D",
        "raw_20d_win_rate": "Raw Win 20D",
        "universe_baseline_avg_20d_return": "Universe Baseline 20D",
        "excess_vs_universe_avg_20d": "Ex Universe 20D",
        "avg_excess_vs_spy_20d": "Ex SPY 20D",
        "avg_excess_vs_qqq_20d": "Ex QQQ 20D",
        "avg_excess_vs_sector_etf_20d": "Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "valid_excess_vs_sector_etf_20d_rows": "Valid Ex SecETF",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win 20D",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    if group_col and group_col in labels:
        return _table_html_grouped(fmt, labels[group_col])
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_path_summary_table(
    df: pd.DataFrame,
    id_cols: list[str],
    group_col: str | None = None,
) -> str:
    if df.empty:
        return _warning_html(
            "Forward path table is unavailable. Regenerate the signal dataset to add path labels."
        )

    fmt = df.copy()
    if "stock_personality" in fmt.columns:
        fmt["stock_personality"] = pd.Categorical(
            fmt["stock_personality"], categories=_PERSONALITY_LABELS, ordered=True
        )
    if "condition" in fmt.columns:
        fmt["condition"] = pd.Categorical(
            fmt["condition"], categories=_SIGNAL_CONDITIONS, ordered=True
        )
    sort_cols = [c for c in id_cols if c in fmt.columns]
    if sort_cols:
        fmt = fmt.sort_values(sort_cols)

    for col in ["rows", "valid_20d_rows", "valid_path_20d_rows"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_20d_close_return",
        "avg_20d_mfe",
        "avg_20d_mae",
        "median_20d_mae",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for _, rate_col in _PATH_HIT_COLUMNS:
        if rate_col in fmt.columns:
            fmt[rate_col] = fmt[rate_col].apply(lambda x: _pct(x, 1, colored=False))

    ordered = id_cols + [
        "rows", "valid_20d_rows", "valid_path_20d_rows",
        "avg_20d_close_return", "avg_20d_mfe", "avg_20d_mae",
        "median_20d_mae", "hit_plus_5_before_minus_5_rate",
        "hit_plus_10_before_minus_5_rate", "hit_plus_10_before_minus_10_rate",
        "hit_plus_15_before_minus_10_rate",
    ]
    labels = {
        "focus_case": "Focus Case",
        "ticker": "Ticker",
        "sector": "Sector",
        "stock_personality": "Personality",
        "stock_personality_group": "Personality Group",
        "condition": "Signal Condition",
        "rows": "Rows",
        "valid_20d_rows": "Valid Close 20D",
        "valid_path_20d_rows": "Valid Path 20D",
        "avg_20d_close_return": "Avg 20D Close",
        "avg_20d_mfe": "Avg 20D MFE",
        "avg_20d_mae": "Avg 20D MAE",
        "median_20d_mae": "Med 20D MAE",
        "hit_plus_5_before_minus_5_rate": "+5 Before -5",
        "hit_plus_10_before_minus_5_rate": "+10 Before -5",
        "hit_plus_10_before_minus_10_rate": "+10 Before -10",
        "hit_plus_15_before_minus_10_rate": "+15 Before -10",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    if group_col and group_col in labels:
        return _table_html_grouped(fmt, labels[group_col])
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_signal_lab_summary_table(
    df: pd.DataFrame,
    id_cols: list[str],
    group_col: str | None = None,
) -> str:
    if df.empty:
        return _warning_html(
            "RSI/Bollinger signal lab table is unavailable. Regenerate the signal dataset."
        )

    fmt = df.copy()
    for col in [
        "rows", "valid_5d_rows", "valid_10d_rows", "valid_20d_rows",
        "valid_excess_vs_sector_etf_20d_rows",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)

    pct_cols = [
        "avg_5d_return", "median_5d_return",
        "avg_10d_return", "median_10d_return",
        "avg_20d_return", "median_20d_return",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
        "avg_mfe_20d", "median_mfe_20d",
        "avg_mae_20d", "median_mae_20d",
    ]
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    rate_cols = [
        "5d_win_rate", "10d_win_rate", "20d_win_rate",
        "excess_vs_sector_etf_20d_win_rate",
        "hit_plus_5_before_minus_5_rate",
        "hit_plus_10_before_minus_5_rate",
        "hit_plus_10_before_minus_10_rate",
    ]
    for col in rate_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    ordered = id_cols + [
        "rows", "valid_5d_rows", "avg_5d_return", "median_5d_return", "5d_win_rate",
        "valid_10d_rows", "avg_10d_return", "median_10d_return", "10d_win_rate",
        "valid_20d_rows", "avg_20d_return", "median_20d_return", "20d_win_rate",
        "avg_excess_vs_sector_etf_20d",
        "median_excess_vs_sector_etf_20d",
        "excess_vs_sector_etf_20d_win_rate",
        "avg_mfe_20d", "median_mfe_20d",
        "avg_mae_20d", "median_mae_20d",
        "hit_plus_5_before_minus_5_rate",
        "hit_plus_10_before_minus_5_rate",
        "hit_plus_10_before_minus_10_rate",
    ]
    labels = {
        "condition": "Condition",
        "sector": "Sector",
        "group_type": "Group Type",
        "group_value": "Group",
        "breakdown": "Breakdown",
        "bucket": "Bucket",
        "rows": "Rows",
        "valid_5d_rows": "Valid 5D",
        "valid_10d_rows": "Valid 10D",
        "valid_20d_rows": "Valid 20D",
        "avg_5d_return": "Avg 5D",
        "median_5d_return": "Med 5D",
        "5d_win_rate": "Win 5D",
        "avg_10d_return": "Avg 10D",
        "median_10d_return": "Med 10D",
        "10d_win_rate": "Win 10D",
        "avg_20d_return": "Avg 20D",
        "median_20d_return": "Med 20D",
        "20d_win_rate": "Win 20D",
        "avg_excess_vs_sector_etf_20d": "Ex SecETF 20D",
        "median_excess_vs_sector_etf_20d": "Med Ex SecETF 20D",
        "excess_vs_sector_etf_20d_win_rate": "Ex SecETF Win",
        "avg_mfe_20d": "Avg MFE 20D",
        "median_mfe_20d": "Med MFE 20D",
        "avg_mae_20d": "Avg MAE 20D",
        "median_mae_20d": "Med MAE 20D",
        "hit_plus_5_before_minus_5_rate": "+5 Before -5",
        "hit_plus_10_before_minus_5_rate": "+10 Before -5",
        "hit_plus_10_before_minus_10_rate": "+10 Before -10",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    if group_col and group_col in labels:
        return _table_html_grouped(fmt, labels[group_col])
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_exit_rule_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("RSI/Bollinger exit-rule simulation is unavailable.")

    fmt = df.copy()
    for col in ["trade_count"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_trade_return", "median_trade_return",
        "avg_mae_before_exit", "median_mae_before_exit",
        "avg_mfe_before_exit", "median_mfe_before_exit",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    if "win_rate" in fmt.columns:
        fmt["win_rate"] = fmt["win_rate"].apply(lambda x: _pct(x, 1, colored=False))
    for col in ["avg_bars_held", "median_bars_held"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "n/a")

    ordered = [
        "exit_rule", "trade_count", "avg_trade_return", "median_trade_return",
        "win_rate", "avg_bars_held", "median_bars_held",
        "avg_mae_before_exit", "median_mae_before_exit",
        "avg_mfe_before_exit", "median_mfe_before_exit",
    ]
    labels = {
        "exit_rule": "Exit Rule",
        "trade_count": "Trades",
        "avg_trade_return": "Avg Return",
        "median_trade_return": "Med Return",
        "win_rate": "Win Rate",
        "avg_bars_held": "Avg Bars",
        "median_bars_held": "Med Bars",
        "avg_mae_before_exit": "Avg MAE",
        "median_mae_before_exit": "Med MAE",
        "avg_mfe_before_exit": "Avg MFE",
        "median_mfe_before_exit": "Med MFE",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_dca_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("RSI/Bollinger DCA simulation is unavailable.")

    fmt = df.copy()
    for col in ["trade_count", "max_entries"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "avg_return", "median_return", "avg_mae", "median_mae",
        "worst_trade_return", "worst_mae",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    if "win_rate" in fmt.columns:
        fmt["win_rate"] = fmt["win_rate"].apply(lambda x: _pct(x, 1, colored=False))
    if "avg_entries" in fmt.columns:
        fmt["avg_entries"] = fmt["avg_entries"].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "n/a"
        )

    ordered = [
        "dca_rule", "trade_count", "avg_return", "median_return", "win_rate",
        "avg_entries", "max_entries", "avg_mae", "median_mae",
        "worst_trade_return", "worst_mae",
    ]
    labels = {
        "dca_rule": "DCA Rule",
        "trade_count": "Trades",
        "avg_return": "Avg Return",
        "median_return": "Med Return",
        "win_rate": "Win Rate",
        "avg_entries": "Avg Entries",
        "max_entries": "Max Entries",
        "avg_mae": "Avg MAE",
        "median_mae": "Med MAE",
        "worst_trade_return": "Worst Return",
        "worst_mae": "Worst MAE",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_personality_high_low_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Personality high-vs-low table is unavailable.")

    fmt = df.copy()
    if "stock_personality" in fmt.columns:
        fmt["stock_personality"] = pd.Categorical(
            fmt["stock_personality"], categories=_PERSONALITY_LABELS, ordered=True
        )
        fmt = fmt.sort_values("stock_personality")
    for col in [
        "rows", "high_signal_rows", "high_signal_valid_20d_rows",
        "low_signal_rows", "low_signal_valid_20d_rows",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)
    for col in [
        "high_signal_avg_excess_vs_sector_etf_20d",
        "high_signal_median_excess_vs_sector_etf_20d",
        "low_signal_avg_excess_vs_sector_etf_20d",
        "low_signal_median_excess_vs_sector_etf_20d",
        "high_minus_low_excess_vs_sector_etf_20d",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))
    for col in [
        "high_signal_excess_vs_sector_etf_20d_win_rate",
        "low_signal_excess_vs_sector_etf_20d_win_rate",
    ]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))

    ordered = [
        "stock_personality", "rows",
        "high_signal_rows", "high_signal_valid_20d_rows",
        "high_signal_avg_excess_vs_sector_etf_20d",
        "high_signal_median_excess_vs_sector_etf_20d",
        "high_signal_excess_vs_sector_etf_20d_win_rate",
        "low_signal_rows", "low_signal_valid_20d_rows",
        "low_signal_avg_excess_vs_sector_etf_20d",
        "low_signal_median_excess_vs_sector_etf_20d",
        "low_signal_excess_vs_sector_etf_20d_win_rate",
        "high_minus_low_excess_vs_sector_etf_20d",
        "behavior_label",
    ]
    labels = {
        "stock_personality": "Personality",
        "rows": "Rows",
        "high_signal_rows": "High Rows",
        "high_signal_valid_20d_rows": "High Valid 20D",
        "high_signal_avg_excess_vs_sector_etf_20d": "High Ex SecETF 20D",
        "high_signal_median_excess_vs_sector_etf_20d": "High Med Ex SecETF 20D",
        "high_signal_excess_vs_sector_etf_20d_win_rate": "High Ex Win 20D",
        "low_signal_rows": "Low Rows",
        "low_signal_valid_20d_rows": "Low Valid 20D",
        "low_signal_avg_excess_vs_sector_etf_20d": "Low Ex SecETF 20D",
        "low_signal_median_excess_vs_sector_etf_20d": "Low Med Ex SecETF 20D",
        "low_signal_excess_vs_sector_etf_20d_win_rate": "Low Ex Win 20D",
        "high_minus_low_excess_vs_sector_etf_20d": "High Minus Low",
        "behavior_label": "Behavior",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_ticker_excess_rank_table(
    df: pd.DataFrame,
    metric_col: str,
    valid_col: str,
    raw_col: str,
    secondary_col: str,
    win_col: str,
    metric_label: str,
    secondary_label: str,
) -> str:
    if df.empty or metric_col not in df.columns:
        return _warning_html(f"Ticker ranking is unavailable for {metric_label}.")

    fmt = df.dropna(subset=[metric_col]).copy()
    fmt = fmt[fmt[valid_col].fillna(0) > 0] if valid_col in fmt.columns else fmt
    if fmt.empty:
        return _warning_html(f"No valid ticker rows for {metric_label}.")
    fmt = fmt.sort_values(metric_col, ascending=False)
    fmt.insert(0, "rank", range(1, len(fmt) + 1))

    for col in ["rank", "total_rows", valid_col]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_count)
    rate_cols = {c for c in [metric_col, win_col] if "win_rate" in c}
    for col in dict.fromkeys([raw_col, metric_col, secondary_col, win_col]):
        if col in fmt.columns:
            if col in rate_cols:
                fmt[col] = fmt[col].apply(lambda x: _pct(x, 1, colored=False))
            else:
                fmt[col] = fmt[col].apply(lambda x: _pct(x, 2, colored=True))

    ordered = ["rank", "ticker", "sector", "instrument_type", "sector_etf", valid_col]
    for col in [raw_col, metric_col, secondary_col, win_col]:
        if col not in ordered:
            ordered.append(col)
    labels = {
        "rank": "Rank",
        "ticker": "Ticker",
        "sector": "Sector",
        "instrument_type": "Type",
        "sector_etf": "Sector ETF",
        valid_col: "Valid 20D",
        raw_col: "Raw Avg 20D",
        metric_col: metric_label,
        secondary_col: secondary_label,
        win_col: "Ex Win 20D",
    }
    fmt = fmt[[c for c in ordered if c in fmt.columns]].rename(columns=labels)
    return fmt.to_html(escape=False, classes="data-table", border=0, index=False)


def _format_ticker_excess_rankings(df: pd.DataFrame) -> str:
    if df.empty:
        return _warning_html("Ticker-level excess ranking is unavailable.")

    return f"""
<h3>High Signal Ranked by Avg Excess vs Sector ETF</h3>
<div class="table-wrap">{_format_ticker_excess_rank_table(
    df,
    "high_signal_avg_excess_vs_sector_etf_20d",
    "high_signal_valid_20d_rows",
    "high_signal_raw_avg_20d_return",
    "high_signal_median_excess_vs_sector_etf_20d",
    "high_signal_excess_vs_sector_etf_20d_win_rate",
    "Avg Ex SecETF 20D",
    "Med Ex SecETF 20D",
)}</div>
<h3>Low Signal Ranked by Avg Excess vs Sector ETF</h3>
<div class="table-wrap">{_format_ticker_excess_rank_table(
    df,
    "low_signal_avg_excess_vs_sector_etf_20d",
    "low_signal_valid_20d_rows",
    "low_signal_raw_avg_20d_return",
    "low_signal_median_excess_vs_sector_etf_20d",
    "low_signal_excess_vs_sector_etf_20d_win_rate",
    "Avg Ex SecETF 20D",
    "Med Ex SecETF 20D",
)}</div>
<h3>High Signal Ranked by Excess Win Rate</h3>
<div class="table-wrap">{_format_ticker_excess_rank_table(
    df,
    "high_signal_excess_vs_sector_etf_20d_win_rate",
    "high_signal_valid_20d_rows",
    "high_signal_raw_avg_20d_return",
    "high_signal_avg_excess_vs_sector_etf_20d",
    "high_signal_excess_vs_sector_etf_20d_win_rate",
    "Ex Win 20D",
    "Avg Ex SecETF 20D",
)}</div>
<h3>Low Signal Ranked by Excess Win Rate</h3>
<div class="table-wrap">{_format_ticker_excess_rank_table(
    df,
    "low_signal_excess_vs_sector_etf_20d_win_rate",
    "low_signal_valid_20d_rows",
    "low_signal_raw_avg_20d_return",
    "low_signal_avg_excess_vs_sector_etf_20d",
    "low_signal_excess_vs_sector_etf_20d_win_rate",
    "Ex Win 20D",
    "Avg Ex SecETF 20D",
)}</div>"""


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px; background: #f0f2f5; color: #1a1a1a; line-height: 1.5;
}
.container { max-width: 1300px; margin: 0 auto; padding: 28px 20px 60px; }
header { margin-bottom: 28px; }
header h1 { font-size: 1.65em; font-weight: 700; color: #111; }
header .subtitle { color: #666; margin-top: 5px; font-size: 0.92em; }

.card {
    background: #fff; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    padding: 24px 28px; margin-bottom: 22px;
}
.card h2 {
    font-size: 1.1em; font-weight: 700; color: #2c3e50;
    border-bottom: 2px solid #e8eaf0; padding-bottom: 9px; margin-bottom: 16px;
}
.card h3 { font-size: 0.95em; font-weight: 600; color: #555; margin: 20px 0 8px; }
.card h3:first-of-type { margin-top: 4px; }

.overview-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px;
}
.kv-item { background: #f7f8fb; border-radius: 6px; padding: 11px 14px; }
.kv-item .label { font-size: 0.73em; color: #999; text-transform: uppercase; letter-spacing: 0.05em; }
.kv-item .value { font-size: 1.05em; font-weight: 600; margin-top: 3px; }
.kv-item .value.mono { font-family: monospace; font-size: 0.82em; word-break: break-all; font-weight: 400; }

.config-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 8px;
    font-size: 0.88em;
}
.config-item { background: #f7f8fb; border-radius: 5px; padding: 8px 12px; }
.config-item .ck { color: #999; font-size: 0.82em; }
.config-item .cv { font-family: monospace; font-weight: 600; margin-top: 2px; }

.table-wrap { overflow-x: auto; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 0.84em; }
table.data-table th {
    background: #f0f2f5; padding: 8px 12px; text-align: left; font-weight: 600;
    color: #444; border-bottom: 2px solid #dde1e7; white-space: nowrap;
}
table.data-table td { padding: 6px 12px; white-space: nowrap; }
table.data-table tbody tr:hover td { background: #f7f9fc; }

.filter-note { color: #666; font-size: 0.87em; margin-bottom: 14px; }

.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.chart-full { grid-column: 1 / -1; }
@media (max-width: 860px) { .charts-grid { grid-template-columns: 1fr; } }

footer {
    text-align: center; color: #bbb; font-size: 0.78em;
    margin-top: 36px; padding-top: 14px; border-top: 1px solid #e0e0e0;
}
"""


# ── Main public function ──────────────────────────────────────────────────────

def generate_signal_report_html(
    df: pd.DataFrame,
    out_path: str | Path,
    cfg_summary: dict[str, Any] | None = None,
    ticker_groups: dict[str, list[str]] | None = None,
) -> Path:
    """
    Generate a self-contained interactive HTML validation report.

    Sections included:
        1. Dataset Overview
        2. Per-Ticker Summary
        3. Signal Bucket Analysis  (all 7 buckets guaranteed)
        4. Market Regime Filter    (if market_healthy_either column present)
        5. Stock Trend Filter      (if stock_trend_healthy column present)
        6. Combined Conditions     (if either filter present)
        7. Per-Ticker Signal Buckets
        8. Per-Ticker Signal Quality
        9. Ticker Charts
        10. Market-Filtered Per-Ticker Comparison
        11. Manual Ticker Groups (if configured)
        12. Charts
        13. Interpretation

    Args:
        df:          Full signal dataset (all tickers, market regime joined).
        out_path:    Destination .html file path.
        cfg_summary: Optional config key/values shown in a header card.
        ticker_groups: Optional mapping of group name to ticker symbols.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = Path(out_path)

    has_market = "market_healthy_either" in df.columns
    has_stock = "stock_trend_healthy" in df.columns

    # ── Pre-compute tables ───────────────────────────────────────────────────
    ticker_df = build_ticker_summary(df)
    bucket_df = build_signal_bucket_summary(df)
    ticker_bucket_df = build_per_ticker_signal_bucket_table(df)
    ticker_quality_df = build_ticker_signal_quality_summary(df)
    market_ticker_df = build_market_filtered_ticker_comparison(df) if has_market else pd.DataFrame()
    group_bucket_df = build_group_signal_bucket_table(df, ticker_groups)
    group_high_df = build_group_high_signal_summary(df, ticker_groups)
    edge_global_df = build_global_signal_excess_summary(df)
    edge_high_low_df = build_high_vs_low_excess_summary(df)
    edge_sector_df = build_sector_signal_excess_summary(df)
    edge_ticker_df = build_ticker_signal_excess_summary(df)
    personality_signal_df = build_personality_signal_excess_summary(df)
    personality_high_low_df = build_personality_high_vs_low_summary(df)
    personality_bucket_df = build_personality_bucket_excess_summary(df)
    path_personality_df = build_path_by_personality_signal(df)
    path_focus_df = build_path_focus_summary(path_personality_df)
    path_sector_df = build_path_by_sector_signal(df)
    path_ticker_personality_df = build_path_by_ticker_personality(df)
    signal_lab_df = build_signal_lab_summary(df)
    rsi_bb_sector_df = build_rsi_bb_by_sector(df)
    rsi_bb_personality_df = build_rsi_bb_by_personality(df)
    rsi_bb_market_df = build_rsi_bb_market_breakdown(df)
    rsi_bb_exit_df = build_rsi_bb_exit_rule_summary(df)
    rsi_bb_dca_df = build_rsi_bb_dca_summary(df)
    sector_bucket_df = build_sector_signal_bucket_table(df)
    sector_summary_df = build_sector_summary(df)
    sector_market_df = build_sector_market_regime_comparison(df) if has_market else pd.DataFrame()
    sector_etf_df = build_sector_etf_vs_stocks(df) if has_market else pd.DataFrame()
    broad_group_df = build_broad_group_regime_analysis(df) if has_market else pd.DataFrame()

    # ── Section 1: Dataset overview ──────────────────────────────────────────
    tickers_list = sorted(df["ticker"].unique().tolist())
    missing = df.isnull().sum()
    missing_parts = []
    for col, cnt in missing.items():
        color = "#d62728" if cnt > 0 else "#2ca02c"
        missing_parts.append(
            f'<span style="color:{color}">{_html.escape(col)}: {cnt:,}</span>'
        )
    missing_html = " &nbsp;·&nbsp; ".join(missing_parts)

    overview_html = f"""
<div class="overview-grid">
  <div class="kv-item"><div class="label">Total Rows</div>
    <div class="value">{len(df):,}</div></div>
  <div class="kv-item"><div class="label">Tickers</div>
    <div class="value">{df['ticker'].nunique()}</div></div>
  <div class="kv-item"><div class="label">First Date</div>
    <div class="value">{df['date'].min().strftime('%Y-%m-%d')}</div></div>
  <div class="kv-item"><div class="label">Last Date</div>
    <div class="value">{df['date'].max().strftime('%Y-%m-%d')}</div></div>
  <div class="kv-item"><div class="label">Signal Range</div>
    <div class="value">{df['signal'].min()*100:+.1f}% to {df['signal'].max()*100:+.1f}%</div></div>
  <div class="kv-item"><div class="label">Signal Mean</div>
    <div class="value">{df['signal'].mean()*100:+.2f}%</div></div>
  <div class="kv-item" style="grid-column:1/-1">
    <div class="label">Ticker List</div>
    <div class="value mono">{_html.escape(', '.join(tickers_list))}</div></div>
  <div class="kv-item" style="grid-column:1/-1">
    <div class="label">Missing Values per Column</div>
    <div class="value" style="font-weight:400;font-size:0.88em">{missing_html}</div></div>
</div>"""

    # ── Config section ───────────────────────────────────────────────────────
    config_html = ""
    if cfg_summary:
        items = "".join(
            f'<div class="config-item">'
            f'<div class="ck">{_html.escape(str(k))}</div>'
            f'<div class="cv">{_html.escape(str(v))}</div></div>'
            for k, v in cfg_summary.items()
        )
        config_html = f"""
<div class="card">
  <h2>Model Configuration</h2>
  <div class="config-grid">{items}</div>
</div>"""

    # ── Section 2 & 3: Ticker + bucket tables ────────────────────────────────
    ticker_table = _format_ticker_table(ticker_df)
    bucket_table = _format_bucket_table(bucket_df)
    edge_section = f"""
<div class="card">
  <h2>2. Does the Markov Signal Add Edge?</h2>
  <p class="filter-note">
    Raw forward returns are compared against date-aligned universe, SPY, QQQ,
    and sector ETF baselines. Sector ETF instrument rows use themselves as the
    sector ETF baseline, so their sector-ETF excess is a self-reference by design.
  </p>
  <h3>A. Global Signal Buckets</h3>
  <div class="table-wrap">{_format_edge_summary_table(edge_global_df, ["signal_bucket"])}</div>
  <h3>B. High Signal vs Low Signal</h3>
  <div class="table-wrap">{_format_edge_summary_table(edge_high_low_df, ["condition"])}</div>
  <h3>C. Sector-Level Excess Return</h3>
  <div class="table-wrap">{_format_edge_summary_table(edge_sector_df, ["sector", "condition"], group_col="sector")}</div>
  <h3>D. Ticker-Level Excess Return Rankings</h3>
  <p class="filter-note">
    Rankings use 20D excess returns versus each row's sector ETF. Positive excess
    means the stock outperformed its sector ETF over the same forward window.
  </p>
  {_format_ticker_excess_rankings(edge_ticker_df)}
</div>"""

    # ── Section 4: Market regime filter ─────────────────────────────────────
    personality_section = f"""
<div class="card">
  <h2>3. Stock Personality Regime Analysis</h2>
  <p class="filter-note">
    Personality labels use only trailing price information available at each
    signal date. This section excludes sector ETF rows because their sector-ETF
    excess baseline is self-referential.
  </p>
  <h3>Signal Conditions by Personality</h3>
  <div class="table-wrap">{_format_edge_summary_table(personality_signal_df, ["stock_personality", "condition"], group_col="stock_personality")}</div>
  <h3>High-vs-Low Comparison</h3>
  <div class="table-wrap">{_format_personality_high_low_table(personality_high_low_df)}</div>
  <h3>Charts</h3>
  <div class="charts-grid" style="margin-top:20px">
    <div>{_chart_personality_high_low_metric(personality_high_low_df, "high_signal_avg_excess_vs_sector_etf_20d", "High-Signal Excess vs Sector ETF by Personality", "Excess 20D (%)")}</div>
    <div>{_chart_personality_high_low_metric(personality_high_low_df, "low_signal_avg_excess_vs_sector_etf_20d", "Low-Signal Excess vs Sector ETF by Personality", "Excess 20D (%)")}</div>
    <div class="chart-full">{_chart_personality_bucket_heatmap(personality_bucket_df)}</div>
    <div class="chart-full">{_chart_personality_raw_vs_excess(personality_signal_df)}</div>
  </div>
</div>"""
    path_section = f"""
<div class="card">
  <h2>4. Forward Path / Tradability Analysis</h2>
  <p class="filter-note">
    These are validation labels built from future high/low/close data after the
    signal date. They measure path risk and target/stop outcomes only; they are
    not a backtest and are not used as signal features. Same-day target/stop
    touches are counted conservatively as stop first.
  </p>
  <h3>Focused Cases</h3>
  <div class="table-wrap">{_format_path_summary_table(path_focus_df, ["focus_case", "stock_personality", "condition"])}</div>
  <h3>By Stock Personality and Signal Condition</h3>
  <div class="table-wrap">{_format_path_summary_table(path_personality_df, ["stock_personality", "condition"], group_col="stock_personality")}</div>
  <h3>By Sector and Signal Condition</h3>
  <div class="table-wrap">{_format_path_summary_table(path_sector_df, ["sector", "condition"], group_col="sector")}</div>
  <h3>By Ticker / Personality / Signal Condition</h3>
  <div class="table-wrap">{_format_path_summary_table(path_ticker_personality_df, ["ticker", "stock_personality", "condition"], group_col="ticker")}</div>
</div>"""
    rsi_bb_section = f"""
<div class="card">
  <h2>5. RSI/Bollinger vs Markov Signal Lab</h2>
  <p class="filter-note">
    RSI/Bollinger oversold = close below the 20-period lower Bollinger Band
    and RSI(14) below 24. These are research labels only. Exit-rule and DCA
    simulations are simple per-signal validations, not a portfolio backtest.
  </p>
  <h3>Signal Condition Comparison</h3>
  <div class="table-wrap">{_format_signal_lab_summary_table(signal_lab_df, ["condition"])}</div>
  <h3>RSI/BB Oversold by Sector</h3>
  <div class="table-wrap">{_format_signal_lab_summary_table(rsi_bb_sector_df, ["sector"])}</div>
  <h3>RSI/BB Oversold by Stock Personality</h3>
  <div class="table-wrap">{_format_signal_lab_summary_table(rsi_bb_personality_df, ["group_type", "group_value"], group_col="group_type")}</div>
  <h3>RSI/BB Oversold by Market and Sector ETF Trend</h3>
  <div class="table-wrap">{_format_signal_lab_summary_table(rsi_bb_market_df, ["breakdown", "bucket"], group_col="breakdown")}</div>
  <h3>Simple Exit-Rule Simulation</h3>
  <div class="table-wrap">{_format_exit_rule_table(rsi_bb_exit_df)}</div>
  <h3>Simple DCA/Add Simulation</h3>
  <div class="table-wrap">{_format_dca_table(rsi_bb_dca_df)}</div>
</div>"""
    market_section = ""
    if has_market:
        mkt_df = build_filter_comparison_table(
            df, "market_healthy_either", "Market Healthy ✓", "Market Weak ✗"
        )
        c_mkt_ret = _chart_filter_metric(
            mkt_df, "avg_20d", "Market Filter: Avg 20D Return by Threshold",
            yaxis_title="Avg 20D Return (%)",
        )
        c_mkt_win = _chart_filter_metric(
            mkt_df, "win_20d", "Market Filter: 20D Win Rate by Threshold",
            yaxis_title="Win Rate (%)",
        )
        market_section = f"""
<div class="card">
  <h2>8. Market Regime Filter Analysis</h2>
  <p class="filter-note">
    Market healthy = either SPY or QQQ satisfies:
    <em>close &gt; EMA50</em> <strong>and</strong> <em>EMA10 &gt; EMA20</em>
    (PineScript default rule).
  </p>
  <div class="table-wrap">{_format_comparison_table(mkt_df)}</div>
  <div class="charts-grid" style="margin-top:20px">
    <div>{c_mkt_ret}</div>
    <div>{c_mkt_win}</div>
  </div>
</div>"""

    # ── Section 5: Stock trend filter ────────────────────────────────────────
    stock_section = ""
    if has_stock:
        stk_df = build_filter_comparison_table(
            df, "stock_trend_healthy", "Trend Healthy ✓", "Trend Weak ✗"
        )
        c_stk_ret = _chart_filter_metric(
            stk_df, "avg_20d", "Stock Trend Filter: Avg 20D Return by Threshold",
            yaxis_title="Avg 20D Return (%)",
        )
        c_stk_win = _chart_filter_metric(
            stk_df, "win_20d", "Stock Trend Filter: 20D Win Rate by Threshold",
            yaxis_title="Win Rate (%)",
        )
        stock_section = f"""
<div class="card">
  <h2>9. Stock Trend Filter Analysis</h2>
  <p class="filter-note">
    Stock trend healthy = stock <em>close &gt; EMA50</em>
    <strong>and</strong> <em>EMA10 &gt; EMA20</em>.
  </p>
  <div class="table-wrap">{_format_comparison_table(stk_df)}</div>
  <div class="charts-grid" style="margin-top:20px">
    <div>{c_stk_ret}</div>
    <div>{c_stk_win}</div>
  </div>
</div>"""

    # ── Section 6: Combined conditions ──────────────────────────────────────
    combined_section = ""
    if has_market or has_stock:
        comb_df = build_combined_conditions_table(df)
        c_comb_ret = _chart_filter_metric(
            comb_df, "avg_20d", "Combined Conditions: Avg 20D Return by Threshold",
            yaxis_title="Avg 20D Return (%)",
        )
        c_comb_win = _chart_filter_metric(
            comb_df, "win_20d", "Combined Conditions: 20D Win Rate by Threshold",
            yaxis_title="Win Rate (%)",
        )
        c_comb_count = _chart_sample_counts(comb_df)
        combined_section = f"""
<div class="card">
  <h2>10. Combined Filter Conditions</h2>
  <p class="filter-note">
    All filter combinations at each signal threshold.
    Rows shown only when signal ≥ threshold.
  </p>
  <div class="table-wrap">{_format_comparison_table(comb_df)}</div>
  <div class="charts-grid" style="margin-top:20px">
    <div>{c_comb_ret}</div>
    <div>{c_comb_win}</div>
    <div class="chart-full">{c_comb_count}</div>
  </div>
</div>"""

    # ── Section 7: Charts ────────────────────────────────────────────────────
    per_ticker_bucket_section = f"""
<div class="card">
  <h2>11. Per-Ticker Signal Bucket Table</h2>
  <p class="filter-note">
    Same signal buckets as the global analysis, expanded by ticker. Empty ticker/bucket
    combinations are shown with 0 rows.
  </p>
  <div class="table-wrap">{_format_per_ticker_bucket_table(ticker_bucket_df)}</div>
</div>"""

    per_ticker_quality_section = f"""
<div class="card">
  <h2>12. Per-Ticker Signal Quality Summary</h2>
  <p class="filter-note">
    High signal = signal &gt;= 40%; low signal = signal &lt;= -20%.
    Verdicts are heuristic research labels, not final trading rules.
  </p>
  <div class="table-wrap">{_format_ticker_quality_table(ticker_quality_df)}</div>
</div>"""

    ticker_charts_section = f"""
<div class="card">
  <h2>13. Ticker Charts</h2>
  <div class="charts-grid">
    <div class="chart-full">{_chart_ticker_bucket_heatmap(ticker_bucket_df)}</div>
    <div>{_chart_quality_metric(ticker_quality_df, "high_signal_avg_20d_return", "High-Signal Avg 20D Return by Ticker (Signal &gt;= 40%)", "Avg 20D Return (%)")}</div>
    <div>{_chart_quality_metric(ticker_quality_df, "high_signal_20d_win_rate", "High-Signal 20D Win Rate by Ticker (Signal &gt;= 40%)", "Win Rate (%)", percent_decimals=1, signed=False)}</div>
    <div>{_chart_quality_metric(ticker_quality_df, "directional_spread", "Directional Spread by Ticker", "High Avg 20D minus Low Avg 20D (%)")}</div>
    <div class="chart-full">{_chart_ticker_bucket_sample_counts(ticker_bucket_df)}</div>
  </div>
</div>"""

    if has_market:
        market_ticker_section = f"""
<div class="card">
  <h2>14. Market-Filtered Per-Ticker Comparison</h2>
  <p class="filter-note">
    Compares signal &gt;= 40% rows with and without the SPY/QQQ market regime filter.
    This is intended to show which tickers improve when market_healthy_either is used.
  </p>
  <div class="table-wrap">{_format_market_ticker_comparison_table(market_ticker_df)}</div>
</div>"""
    else:
        market_ticker_section = f"""
<div class="card">
  <h2>14. Market-Filtered Per-Ticker Comparison</h2>
  {_warning_html("market_healthy_either is missing; skipping market-filtered ticker comparison.")}
</div>"""

    group_section = ""
    if ticker_groups:
        if group_bucket_df.empty:
            group_section = f"""
<div class="card">
  <h2>15. Manual Ticker Groups</h2>
  {_warning_html("ticker_groups is configured, but no configured tickers matched the dataset.")}
</div>"""
        else:
            group_section = f"""
<div class="card">
  <h2>15. Manual Ticker Groups</h2>
  <p class="filter-note">
    Group-level buckets use the optional ticker_groups mapping from config.yaml.
  </p>
  <div class="table-wrap">{_format_group_bucket_table(group_bucket_df)}</div>
  <div class="charts-grid" style="margin-top:20px">
    <div class="chart-full">{_chart_group_bucket_heatmap(group_bucket_df)}</div>
    <div>{_chart_group_high_signal_metric(group_high_df, "high_signal_avg_20d_return", "High-Signal Avg 20D Return by Group", "Avg 20D Return (%)")}</div>
    <div>{_chart_group_high_signal_metric(group_high_df, "high_signal_20d_win_rate", "High-Signal 20D Win Rate by Group", "Win Rate (%)", signed=False)}</div>
  </div>
</div>"""

    sector_summary_section = f"""
<div class="card">
  <h2>16. Sector Summary</h2>
  <p class="filter-note">
    Sector-level high signal rows use signal &gt;= 40%. Valid sample counts exclude
    missing future returns.
  </p>
  <div class="table-wrap">{_format_sector_summary_table(sector_summary_df)}</div>
</div>"""

    if has_market:
        sector_market_section = f"""
<div class="card">
  <h2>17. Sector Market-Regime Comparison</h2>
  <p class="filter-note">
    Compares high Markov signal rows by sector under all market conditions,
    SPY/QQQ healthy, and SPY/QQQ weak.
  </p>
  <div class="table-wrap">{_format_sector_market_table(sector_market_df)}</div>
</div>"""
        sector_etf_section = f"""
<div class="card">
  <h2>18. Sector ETF vs Stocks</h2>
  <p class="filter-note">
    Compares each sector ETF against the stock basket in the same sector under
    high Markov signal conditions.
  </p>
  <div class="table-wrap">{_format_sector_etf_vs_stocks_table(sector_etf_df)}</div>
</div>"""
        broad_group_section = f"""
<div class="card">
  <h2>19. Defensive vs Growth/Cyclical Regime Analysis</h2>
  <p class="filter-note">
    Broad groups are heuristic sector groupings and can overlap by design.
  </p>
  <div class="table-wrap">{_format_broad_group_regime_table(broad_group_df)}</div>
</div>"""
    else:
        sector_market_section = f"""
<div class="card">
  <h2>17. Sector Market-Regime Comparison</h2>
  {_warning_html("market_healthy_either is missing; skipping sector market-regime comparison.")}
</div>"""
        sector_etf_section = f"""
<div class="card">
  <h2>18. Sector ETF vs Stocks</h2>
  {_warning_html("market_healthy_either is missing; skipping sector ETF vs stocks market splits.")}
</div>"""
        broad_group_section = f"""
<div class="card">
  <h2>19. Defensive vs Growth/Cyclical Regime Analysis</h2>
  {_warning_html("market_healthy_either is missing; skipping broad group regime analysis.")}
</div>"""

    sector_charts_section = f"""
<div class="card">
  <h2>20. Sector Charts</h2>
  <div class="charts-grid">
    <div class="chart-full">{_chart_sector_bucket_heatmap(sector_bucket_df)}</div>
    <div>{_chart_sector_high_signal_metric(sector_summary_df, "high_signal_avg_20d_return", "High-Signal Avg 20D Return by Sector", "Avg 20D Return (%)")}</div>
    <div>{_chart_sector_high_signal_metric(sector_summary_df, "high_signal_20d_win_rate", "High-Signal 20D Win Rate by Sector", "Win Rate (%)", signed=False)}</div>
    <div>{_chart_sector_market_comparison_metric(sector_market_df, "avg_20d_return", "Market Healthy vs Weak Avg 20D by Sector", "Avg 20D Return (%)")}</div>
    <div>{_chart_sector_market_comparison_metric(sector_market_df, "20d_win_rate", "Market Healthy vs Weak 20D Win Rate by Sector", "Win Rate (%)", signed=False)}</div>
    <div class="chart-full">{_chart_broad_group_regime_heatmap(broad_group_df)}</div>
  </div>
</div>"""

    timeline_html = "".join(
        f"<div>{_chart_ticker_timeline(df, tkr)}</div>"
        for tkr in sorted(df["ticker"].unique())
    )

    # ── Section 8: Interpretation ─────────────────────────────────────────────
    interp_html = _generate_interpretation(df, bucket_df, ticker_df)

    # ── Assemble ─────────────────────────────────────────────────────────────
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Markov Signal Validation Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">

  <header>
    <h1>Markov Signal Validation Report</h1>
    <p class="subtitle">Generated {timestamp} &nbsp;·&nbsp; Research only — not financial advice</p>
  </header>

  {config_html}

  <div class="card">
    <h2>1. Dataset Overview</h2>
    {overview_html}
  </div>

  {edge_section}

  {personality_section}

  {path_section}

  {rsi_bb_section}

  <div class="card">
    <h2>6. Per-Ticker Summary</h2>
    <div class="table-wrap">{ticker_table}</div>
  </div>

  <div class="card">
    <h2>7. Signal Bucket Analysis</h2>
    <p class="filter-note">
      All 7 buckets are shown. Buckets with 0 rows had no observations in this dataset.
      Forward returns are raw price returns — no entry/exit logic applied.
    </p>
    <div class="table-wrap">{bucket_table}</div>
  </div>

  {market_section}

  {stock_section}

  {combined_section}

  {per_ticker_bucket_section}

  {per_ticker_quality_section}

  {ticker_charts_section}

  {market_ticker_section}

  {group_section}

  {sector_summary_section}

  {sector_market_section}

  {sector_etf_section}

  {broad_group_section}

  {sector_charts_section}

  <div class="card">
    <h2>21. Distribution &amp; Signal Charts</h2>

    <h3>Forward Returns by Bucket</h3>
    <div class="charts-grid">
      <div class="chart-full">{_chart_returns_by_bucket(bucket_df)}</div>
      <div class="chart-full">{_chart_hit_rate_by_bucket(bucket_df)}</div>
    </div>

    <h3>Ticker Performance at High Signal (≥ 10%)</h3>
    <div class="charts-grid">
      <div class="chart-full">{_chart_ticker_at_high_signal(df)}</div>
    </div>

    <h3>Signal vs 20D Return — Scatter &amp; Distribution</h3>
    <div class="charts-grid">
      <div>{_chart_scatter(df)}</div>
      <div>{_chart_box_by_bucket(df)}</div>
    </div>

    <h3>Close Price &amp; Signal Timeline — Per Ticker</h3>
    <div class="charts-grid">{timeline_html}</div>
  </div>

  <div class="card">
    <h2>22. Interpretation</h2>
    {interp_html}
  </div>

  <footer>
    Markov Regime Matrix Signal Validation &nbsp;·&nbsp;
    Generated {timestamp} &nbsp;·&nbsp;
    Research only — not a trading system — not financial advice.
  </footer>

</div>
</body>
</html>"""

    out_path.write_text(html_out, encoding="utf-8")
    return out_path
