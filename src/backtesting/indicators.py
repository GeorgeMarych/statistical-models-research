"""Small trailing indicator helpers for strategy research.

All helpers in this module use current-or-prior bars only. The backtest engine
handles the execution delay by filling entry and close-based exit signals on the
next bar's open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int, kind: str = "sma") -> pd.Series:
    """Return a simple or exponential moving average."""
    if window <= 0:
        raise ValueError("window must be positive")
    kind_clean = str(kind).lower()
    if kind_clean == "ema":
        return series.ewm(span=window, adjust=False, min_periods=window).mean()
    if kind_clean == "sma":
        return series.rolling(window=window, min_periods=window).mean()
    raise ValueError(f"Unsupported moving average kind: {kind}")


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Compute Wilder-style RSI from trailing closes."""
    if length <= 0:
        raise ValueError("length must be positive")
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


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    stdev_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Return trailing Bollinger Band columns: middle, upper, lower."""
    if window <= 0:
        raise ValueError("window must be positive")
    middle = close.rolling(window=window, min_periods=window).mean()
    stdev = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + stdev_multiplier * stdev
    lower = middle - stdev_multiplier * stdev
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
        },
        index=close.index,
    )


def true_range(data: pd.DataFrame) -> pd.Series:
    """Compute true range from high, low, and close."""
    high = data["high"]
    low = data["low"]
    close = data["close"]
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_atr(
    data: pd.DataFrame,
    length: int = 14,
    method: str = "wilder",
) -> pd.Series:
    """Compute ATR using Wilder smoothing or a simple moving average."""
    if length <= 0:
        raise ValueError("length must be positive")
    tr = true_range(data)
    method_clean = str(method).lower()
    if method_clean == "sma":
        return tr.rolling(window=length, min_periods=length).mean()
    if method_clean == "wilder":
        return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    raise ValueError(f"Unsupported ATR method: {method}")
