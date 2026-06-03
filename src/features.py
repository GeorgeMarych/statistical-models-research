"""
Current wide research feature builder.

This module still assembles signal-date features, legacy Markov transition
outputs, RSI/Bollinger labels, personality labels, forward returns, path
outcomes, and excess-return baselines into one wide validation dataset.

The long-term architecture should split this into explicit state, transition,
outcome, and research panels. Current functions remain here for compatibility
with the existing CSV/report generation pipeline.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .config import Config
from .markov_model import classify_states, compute_rolling_markov


# ── EMA helper ───────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average using the recursive formula (adjust=False).
    Matches TradingView / PineScript default EMA behavior:
        alpha = 2 / (span + 1)
        EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
    """
    return series.ewm(span=span, adjust=False).mean()


# ── Stock trend filters ───────────────────────────────────────────────────────

def compute_stock_trend_filters(close: pd.Series) -> pd.DataFrame:
    """
    Compute per-bar EMA-based trend filter columns for a single stock.

    Columns returned (all binary int 0/1):
        stock_ema10_above_ema20   — EMA10 > EMA20
        stock_close_above_ema50   — close > EMA50
        stock_close_above_ema100  — close > EMA100
        stock_close_above_ema200  — close > EMA200
        stock_trend_healthy       — close > EMA50 AND EMA10 > EMA20
                                    (mirrors PineScript EMA trend filter)
    """
    ema10 = compute_ema(close, 10)
    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    ema100 = compute_ema(close, 100)
    ema200 = compute_ema(close, 200)

    return pd.DataFrame(
        {
            "stock_ema10_above_ema20": (ema10 > ema20).astype(int),
            "stock_close_above_ema50": (close > ema50).astype(int),
            "stock_close_above_ema100": (close > ema100).astype(int),
            "stock_close_above_ema200": (close > ema200).astype(int),
            "stock_trend_healthy": (
                (close > ema50) & (ema10 > ema20)
            ).astype(int),
        },
        index=close.index,
    )


# ── Market regime features ────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style RSI using only current and prior closes."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def compute_rsi_bollinger_features(close: pd.Series) -> pd.DataFrame:
    """
    Compute RSI/Bollinger mean-reversion setup features.

    All features use current-or-prior bars only. The oversold signal is:
    close < lower Bollinger Band and RSI(14) < 24.
    """
    rsi = compute_rsi(close, length=14)
    basis = close.rolling(window=20, min_periods=20).mean()
    stdev = close.rolling(window=20, min_periods=20).std(ddof=0)
    upper = basis + 2.0 * stdev
    lower = basis - 2.0 * stdev
    width = (upper - lower) / basis

    close_below_lower = close < lower
    rsi_below = rsi < 24
    oversold = close_below_lower & rsi_below

    return pd.DataFrame(
        {
            "rsi_14": rsi,
            "bb_basis_20": basis,
            "bb_upper_20_2": upper,
            "bb_lower_20_2": lower,
            "bb_width_20_2": width,
            "close_below_lower_bb": close_below_lower.astype(bool),
            "rsi_below_24": rsi_below.astype(bool),
            "rsi_bb_oversold_signal": oversold.astype(bool),
        },
        index=close.index,
    )


def compute_long_term_personality_features(close: pd.Series) -> pd.DataFrame:
    """
    Compute past-only long-term trend/personality features for each bar.

    All columns use current-or-prior prices only. They are validation/regime
    descriptors and must not be computed from future returns.
    """
    trading_days = {
        "1y": 252,
        "2y": 504,
        "3y": 756,
        "5y": 1260,
    }

    features: dict[str, pd.Series] = {}
    for label, window in trading_days.items():
        features[f"trailing_{label}_return"] = close / close.shift(window) - 1.0

    for label, window in [("1y", 252), ("3y", 756), ("5y", 1260)]:
        rolling_high = close.rolling(window=window, min_periods=window).max()
        features[f"distance_from_{label}_high"] = close / rolling_high - 1.0

    log_close = np.log(close)
    features["long_term_slope_200d"] = (log_close - log_close.shift(200)) / 200.0
    features["long_term_slope_500d"] = (log_close - log_close.shift(500)) / 500.0

    out = pd.DataFrame(features, index=close.index)
    trailing_3y = out["trailing_3y_return"]
    drawdown_3y = out["distance_from_3y_high"]

    conditions = [
        trailing_3y.isna(),
        drawdown_3y <= -0.40,
        trailing_3y >= 1.00,
        (trailing_3y >= 0.30) & (trailing_3y < 1.00),
        (trailing_3y > -0.20) & (trailing_3y < 0.30),
        trailing_3y <= -0.20,
    ]
    choices = [
        "unknown",
        "deep_drawdown",
        "secular_winner",
        "uptrend",
        "flat",
        "laggard",
    ]
    out["stock_personality"] = np.select(conditions, choices, default="unknown")

    group_map = {
        "secular_winner": "winner_or_uptrend",
        "uptrend": "winner_or_uptrend",
        "flat": "flat_or_laggard",
        "laggard": "flat_or_laggard",
        "deep_drawdown": "deep_drawdown",
        "unknown": "unknown",
    }
    out["stock_personality_group"] = out["stock_personality"].map(group_map)
    return out


def compute_market_regime_features(
    spy_close: pd.Series,
    qqq_close: pd.Series,
) -> pd.DataFrame:
    """
    Compute SPY/QQQ EMA features and regime flags for each trading date.

    Regime rule matches PineScript default:
        healthy = price > EMA50  AND  EMA10 > EMA20

    Columns returned:
        spy_close, spy_ema10, spy_ema20, spy_ema50
        qqq_close, qqq_ema10, qqq_ema20, qqq_ema50
        spy_healthy_price_above_ema50  (binary)
        spy_healthy_ema10_above_ema20  (binary)
        qqq_healthy_price_above_ema50  (binary)
        qqq_healthy_ema10_above_ema20  (binary)
        market_healthy_either          — SPY healthy OR QQQ healthy
        market_healthy_both            — SPY healthy AND QQQ healthy
    """
    spy_ema10 = compute_ema(spy_close, 10)
    spy_ema20 = compute_ema(spy_close, 20)
    spy_ema50 = compute_ema(spy_close, 50)

    qqq_ema10 = compute_ema(qqq_close, 10)
    qqq_ema20 = compute_ema(qqq_close, 20)
    qqq_ema50 = compute_ema(qqq_close, 50)

    spy_pa50 = spy_close > spy_ema50
    spy_e10_e20 = spy_ema10 > spy_ema20
    qqq_pa50 = qqq_close > qqq_ema50
    qqq_e10_e20 = qqq_ema10 > qqq_ema20

    spy_healthy = spy_pa50 & spy_e10_e20
    qqq_healthy = qqq_pa50 & qqq_e10_e20

    return pd.DataFrame(
        {
            "spy_close": spy_close,
            "spy_ema10": spy_ema10,
            "spy_ema20": spy_ema20,
            "spy_ema50": spy_ema50,
            "qqq_close": qqq_close,
            "qqq_ema10": qqq_ema10,
            "qqq_ema20": qqq_ema20,
            "qqq_ema50": qqq_ema50,
            "spy_healthy_price_above_ema50": spy_pa50.astype(int),
            "spy_healthy_ema10_above_ema20": spy_e10_e20.astype(int),
            "qqq_healthy_price_above_ema50": qqq_pa50.astype(int),
            "qqq_healthy_ema10_above_ema20": qqq_e10_e20.astype(int),
            "market_healthy_either": (spy_healthy | qqq_healthy).astype(int),
            "market_healthy_both": (spy_healthy & qqq_healthy).astype(int),
        },
        index=spy_close.index,
    )


# ── Forward returns ──────────────────────────────────────────────────────────

def compute_forward_returns(close: pd.Series, days: list[int]) -> pd.DataFrame:
    """
    Compute forward returns for multiple horizons.

    future_Nd_return = close.shift(-N) / close - 1

    The last N rows will be NaN for each horizon — this is expected.
    """
    frames: dict[str, pd.Series] = {}
    for d in days:
        frames[f"future_{d}d_return"] = close.shift(-d) / close - 1.0
    return pd.DataFrame(frames, index=close.index)


def _future_rolling_extreme(series: pd.Series, horizon: int, extreme: str) -> pd.Series:
    """Rolling max/min over the next horizon bars, excluding the current bar."""
    future = series.shift(-1)
    reversed_future = future.iloc[::-1]
    rolling = reversed_future.rolling(window=horizon, min_periods=horizon)
    if extreme == "max":
        values = rolling.max()
    elif extreme == "min":
        values = rolling.min()
    else:
        raise ValueError(f"Unsupported rolling extreme: {extreme}")
    return values.iloc[::-1]


def _target_before_stop(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int,
    target_return: float,
    stop_return: float,
) -> pd.Series:
    """
    Label whether a long-side target is hit before a stop over a future horizon.

    This is an evaluation label, not trading logic. The current bar is excluded.
    If both target and stop are touched on the same future daily bar, the stop is
    treated as occurring first as a conservative assumption. Rows without a full
    future horizon are NaN; rows where neither level is touched are 0.
    """
    close_values = close.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    out = np.full(len(close_values), np.nan, dtype=float)

    for i in range(len(close_values) - horizon):
        base = close_values[i]
        if np.isnan(base) or base <= 0:
            continue

        target_price = base * (1.0 + target_return)
        stop_price = base * (1.0 + stop_return)
        result = 0.0

        for j in range(i + 1, i + horizon + 1):
            future_high = high_values[j]
            future_low = low_values[j]
            if np.isnan(future_high) or np.isnan(future_low):
                result = np.nan
                break
            if future_low <= stop_price:
                result = 0.0
                break
            if future_high >= target_price:
                result = 1.0
                break

        out[i] = result

    return pd.Series(out, index=close.index)


def compute_forward_path_metrics(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    days: list[int],
) -> pd.DataFrame:
    """
    Compute future path/risk labels for validation only.

    MFE/MAE use future high/low data over each horizon, excluding the current
    bar. Target/stop labels are 20D long-side path outcomes and deliberately
    conservative on same-bar target/stop touches.
    """
    frames: dict[str, pd.Series] = {}
    for d in days:
        max_high = _future_rolling_extreme(high, d, "max")
        min_low = _future_rolling_extreme(low, d, "min")
        frames[f"max_favorable_excursion_{d}d"] = max_high / close - 1.0
        frames[f"max_adverse_excursion_{d}d"] = min_low / close - 1.0

    if 20 in days:
        rules = [
            ("hit_plus_5_before_minus_5", 0.05, -0.05),
            ("hit_plus_10_before_minus_5", 0.10, -0.05),
            ("hit_plus_10_before_minus_10", 0.10, -0.10),
            ("hit_plus_15_before_minus_10", 0.15, -0.10),
        ]
        for name, target, stop in rules:
            frames[name] = _target_before_stop(
                close=close,
                high=high,
                low=low,
                horizon=20,
                target_return=target,
                stop_return=stop,
            )

    return pd.DataFrame(frames, index=close.index)


def ensure_market_forward_returns(
    regime: pd.DataFrame,
    days: list[int],
) -> pd.DataFrame:
    """
    Add SPY/QQQ forward-return outcome columns to the market regime frame.

    These columns are validation baselines only. They are computed from future
    closes and must not be used as signal/filter features.
    """
    out = regime.copy()
    for prefix in ["spy", "qqq"]:
        close_col = f"{prefix}_close"
        if close_col not in out.columns:
            continue
        close = out[close_col]
        for d in days:
            col = f"{prefix}_future_{d}d_return"
            if col not in out.columns:
                out[col] = close.shift(-d) / close - 1.0
    return out


def add_baseline_return_features(
    dataset: pd.DataFrame,
    raw_data: dict[str, pd.DataFrame],
    days: list[int],
) -> pd.DataFrame:
    """
    Add benchmark forward returns and excess returns to the validation dataset.

    Baselines are aligned by the row's signal date:
        - SPY/QQQ columns are expected to already be joined from market regime.
        - Sector ETF columns are joined from raw_data using sector_etf + date.
        - Universe average is the same-date average forward return across rows.

    For sector ETF rows, the sector ETF baseline is the ETF itself; the
    sector_etf_baseline_is_self flag makes that explicit.
    """
    out = dataset.copy()
    if "date" not in out.columns:
        return out

    out["date"] = pd.to_datetime(out["date"])

    for d in days:
        for prefix in ["spy", "qqq"]:
            baseline_col = f"{prefix}_future_{d}d_return"
            excess_col = f"excess_vs_{prefix}_{d}d"
            if f"future_{d}d_return" in out.columns and baseline_col in out.columns:
                out[excess_col] = out[f"future_{d}d_return"] - out[baseline_col]
            else:
                out[excess_col] = np.nan

    for d in days:
        sector_col = f"sector_etf_future_{d}d_return"
        if sector_col not in out.columns:
            out[sector_col] = np.nan

    if "sector_etf" in out.columns:
        etfs = sorted(
            {
                str(x).strip().upper()
                for x in out["sector_etf"].dropna().unique()
                if str(x).strip()
            }
        )
        sector_frames: list[pd.DataFrame] = []
        sector_trend_frames: list[pd.DataFrame] = []
        for etf in etfs:
            ohlcv = raw_data.get(etf)
            if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
                continue
            forward = compute_forward_returns(ohlcv["close"], days=days).reset_index()
            forward["date"] = pd.to_datetime(forward["date"])
            forward["sector_etf"] = etf
            forward = forward.rename(
                columns={
                    f"future_{d}d_return": f"sector_etf_future_{d}d_return"
                    for d in days
                }
            )
            sector_frames.append(forward)

            trend = compute_stock_trend_filters(ohlcv["close"]).reset_index()
            trend["date"] = pd.to_datetime(trend["date"])
            trend["sector_etf"] = etf
            trend = trend.rename(
                columns={"stock_trend_healthy": "sector_etf_trend_healthy"}
            )
            sector_trend_frames.append(
                trend[["sector_etf", "date", "sector_etf_trend_healthy"]]
            )

        if sector_frames:
            sector_baselines = pd.concat(sector_frames, ignore_index=True)
            merge_cols = ["sector_etf", "date"] + [
                f"sector_etf_future_{d}d_return" for d in days
            ]
            out = out.drop(columns=[c for c in merge_cols[2:] if c in out.columns])
            out = out.merge(
                sector_baselines[merge_cols],
                on=["sector_etf", "date"],
                how="left",
            )
        if sector_trend_frames:
            sector_trends = pd.concat(sector_trend_frames, ignore_index=True)
            out = out.drop(columns=["sector_etf_trend_healthy"], errors="ignore")
            out = out.merge(
                sector_trends,
                on=["sector_etf", "date"],
                how="left",
            )
    if "sector_etf_trend_healthy" not in out.columns:
        out["sector_etf_trend_healthy"] = np.nan

    if {"ticker", "sector_etf"}.issubset(out.columns):
        out["sector_etf_baseline_is_self"] = (
            out["ticker"].astype(str).str.upper()
            == out["sector_etf"].astype(str).str.upper()
        )
    else:
        out["sector_etf_baseline_is_self"] = False

    for d in days:
        raw_col = f"future_{d}d_return"
        sector_col = f"sector_etf_future_{d}d_return"
        excess_col = f"excess_vs_sector_etf_{d}d"
        if raw_col in out.columns and sector_col in out.columns:
            out[excess_col] = out[raw_col] - out[sector_col]
        else:
            out[excess_col] = np.nan

    for d in days:
        raw_col = f"future_{d}d_return"
        avg_col = f"universe_avg_future_{d}d_return"
        excess_col = f"excess_vs_universe_avg_{d}d"
        if raw_col not in out.columns:
            out[avg_col] = np.nan
            out[excess_col] = np.nan
            continue
        universe_avg = out.groupby("date", dropna=False)[raw_col].mean().rename(avg_col)
        out = out.drop(columns=[avg_col], errors="ignore")
        out = out.merge(universe_avg, on="date", how="left")
        out[excess_col] = out[raw_col] - out[avg_col]

    return out


# ── Per-ticker dataset builder ────────────────────────────────────────────────

def add_signal_comparison_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean research labels comparing Markov and RSI/Bollinger signals."""
    out = df.copy()
    oversold = (
        out["rsi_bb_oversold_signal"].astype(bool)
        if "rsi_bb_oversold_signal" in out.columns
        else pd.Series(False, index=out.index)
    )
    markov_high = out["signal"] >= 0.40
    markov_low = out["signal"] <= -0.20
    not_bearish = out["signal"] > -0.20
    deep_drawdown = (
        out["stock_personality"].astype(str).eq("deep_drawdown")
        if "stock_personality" in out.columns
        else pd.Series(False, index=out.index)
    )

    out["signal_markov_high"] = markov_high.astype(bool)
    out["signal_markov_low"] = markov_low.astype(bool)
    out["signal_rsi_bb"] = oversold.astype(bool)
    out["signal_rsi_bb_markov_high"] = (oversold & markov_high).astype(bool)
    out["signal_rsi_bb_markov_low"] = (oversold & markov_low).astype(bool)
    out["signal_rsi_bb_markov_not_bearish"] = (oversold & not_bearish).astype(bool)
    out["signal_rsi_bb_deep_drawdown"] = (oversold & deep_drawdown).astype(bool)
    out["signal_rsi_bb_deep_drawdown_markov_high"] = (
        oversold & deep_drawdown & markov_high
    ).astype(bool)
    out["signal_rsi_bb_deep_drawdown_markov_low"] = (
        oversold & deep_drawdown & markov_low
    ).astype(bool)
    return out


def build_signal_dataset(
    ticker: str,
    ohlcv: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Build the full signal dataset for one ticker.

    Steps:
        1. Classify each bar as Bull / Sideways / Bear.
        2. Compute rolling Markov transition probabilities.
        3. Compute forward returns and forward path labels for each horizon.
        4. Compute stock EMA trend filter and RSI/Bollinger columns.
        5. Add comparison signal labels.
        6. Join everything and drop the warm-up rows (signal is NaN there).

    Returns a flat DataFrame with one row per bar (after warm-up), columns:
        ticker, date, close, state, p_bull, p_bear,
        signal, sample_count,
        stock_ema10_above_ema20, stock_close_above_ema50, ..., stock_trend_healthy,
        future_5d_return, ...
    """
    close = ohlcv["close"]
    high = ohlcv["high"] if "high" in ohlcv.columns else close
    low = ohlcv["low"] if "low" in ohlcv.columns else close

    states = classify_states(
        close,
        lookback=cfg.state_lookback,
        bull_threshold=cfg.bull_threshold,
        bear_threshold=cfg.bear_threshold,
    )

    markov = compute_rolling_markov(states, window=cfg.training_window)
    forward = compute_forward_returns(close, days=cfg.forward_return_days)
    path = compute_forward_path_metrics(
        close=close,
        high=high,
        low=low,
        days=cfg.forward_return_days,
    )
    stock_trend = compute_stock_trend_filters(close)
    rsi_bollinger = compute_rsi_bollinger_features(close)
    personality = compute_long_term_personality_features(close)

    df = pd.concat(
        [
            close.rename("close"),
            high.rename("high"),
            low.rename("low"),
            states.rename("state"),
            markov,
            stock_trend,
            rsi_bollinger,
            personality,
            forward,
            path,
        ],
        axis=1,
    )

    df.insert(0, "ticker", ticker)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # Keep only rows where a full-window signal has been computed
    df = df[df["signal"].notna()].copy()

    # Clean up dtypes now that NaN rows are gone
    df["state"] = df["state"].astype(int)
    df["sample_count"] = df["sample_count"].astype(int)
    df = add_signal_comparison_labels(df)

    return df
