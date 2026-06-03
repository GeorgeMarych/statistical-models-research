"""Sector regime state panel builder.

This module builds one row per sector ETF per date using only current and
historical sector and benchmark data. Forward sector returns are computed only
for validation summaries.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_sector_regime_daily(
    sector_symbols: list[str],
    sector_data: dict[str, pd.DataFrame],
    market_panel: pd.DataFrame,
    sector_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build one daily sector-regime row per sector ETF."""
    sector_names = sector_names or {}
    frames: list[pd.DataFrame] = []
    market = market_panel.copy()
    if "date" in market.columns:
        market["date"] = pd.to_datetime(market["date"])
        market = market.set_index("date")

    benchmark_index = market.index if not market.empty else None
    spy = _market_close(market, "spy")
    rsp = _market_close(market, "rsp")

    for symbol in _dedupe(sector_symbols):
        df = sector_data.get(symbol)
        if df is None or df.empty:
            continue
        normalized = _normalize_ohlcv(df)
        index = normalized.index
        if benchmark_index is not None and not benchmark_index.empty:
            index = index.union(benchmark_index).sort_values()

        close = pd.to_numeric(normalized["close"], errors="coerce").reindex(index)
        frame = pd.DataFrame(
            {
                "date": index,
                "sector_etf": symbol,
                "sector": sector_names.get(symbol, symbol),
                "close": close,
            }
        )
        frame = frame.set_index("date")
        _add_sector_features(frame, close)

        if spy is not None:
            _add_relative_strength(frame, close, spy.reindex(index), "spy")
        if rsp is not None:
            _add_relative_strength(frame, close, rsp.reindex(index), "rsp")

        frame["sector_regime"] = classify_sector_regime(frame)
        frames.append(frame.reset_index())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "sector_etf"])


def classify_sector_regime(frame: pd.DataFrame) -> pd.Series:
    """Assign simple transparent sector regime labels."""
    above20 = _flag(frame, "close_above_sma20")
    above50 = _flag(frame, "close_above_sma50")
    above200 = _flag(frame, "close_above_sma200")
    slope20_up = _gt(frame, "sma20_slope_5d", 0.0)
    slope50_up = _gt(frame, "sma50_slope_5d", 0.0)
    slope20_down = _lt(frame, "sma20_slope_5d", 0.0)
    slope50_down = _lt(frame, "sma50_slope_5d", 0.0)

    rs_spy20_pos = _gt(frame, "rs_vs_spy_20d", 0.0)
    rs_spy20_neg = _lt(frame, "rs_vs_spy_20d", 0.0)
    rs_rsp20_pos = _gt_or_unavailable(frame, "rs_vs_rsp_20d", 0.0)
    rs_rsp20_neg = _lt(frame, "rs_vs_rsp_20d", 0.0)

    strong = above20 & above50 & above200 & slope20_up & slope50_up & rs_spy20_pos & rs_rsp20_pos
    improving = above20 & slope20_up & (rs_spy20_pos | _gt(frame, "rs_vs_spy_50d", 0.0))
    weak = (~above50) & (~above200) & slope20_down & slope50_down & rs_spy20_neg
    weakening = ((~above20) | slope20_down) & (rs_spy20_neg | rs_rsp20_neg)

    labels = np.select(
        [strong, weak, improving, weakening],
        ["strong", "weak", "improving", "weakening"],
        default="neutral",
    )
    return pd.Series(labels, index=frame.index, name="sector_regime")


def build_sector_regime_summary(
    sector_panel: pd.DataFrame,
    days: list[int],
) -> pd.DataFrame:
    """Average forward sector ETF returns by sector regime."""
    if sector_panel.empty or "sector_regime" not in sector_panel.columns:
        return pd.DataFrame()

    panel = sector_panel.sort_values(["sector_etf", "date"]).copy()
    for d in days:
        panel[f"forward_{d}d_return"] = (
            panel.groupby("sector_etf")["close"].shift(-d) / panel["close"] - 1.0
        )

    rows: list[dict[str, Any]] = []
    for regime in sorted(panel["sector_regime"].dropna().unique()):
        subset = panel[panel["sector_regime"] == regime]
        row: dict[str, Any] = {
            "sector_regime": regime,
            "rows": len(subset),
            "sector_count": subset["sector_etf"].nunique(),
        }
        for d in days:
            values = pd.to_numeric(subset[f"forward_{d}d_return"], errors="coerce").dropna()
            row[f"valid_{d}d_rows"] = len(values)
            row[f"avg_forward_{d}d_return"] = float(values.mean()) if len(values) else np.nan
            row[f"median_forward_{d}d_return"] = float(values.median()) if len(values) else np.nan
            row[f"forward_{d}d_win_rate"] = float((values > 0).mean()) if len(values) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _add_sector_features(frame: pd.DataFrame, close: pd.Series) -> None:
    for d in [5, 10, 20]:
        frame[f"return_{d}d"] = close / close.shift(d) - 1.0

    for window in [20, 50, 200]:
        sma = close.rolling(window=window, min_periods=window).mean()
        frame[f"sma{window}"] = sma
        frame[f"close_above_sma{window}"] = _bool_as_float(close > sma, sma.notna())
        frame[f"distance_sma{window}"] = close / sma - 1.0
        frame[f"sma{window}_slope_5d"] = sma / sma.shift(5) - 1.0

    for window in [20, 50]:
        prior_high = close.shift(1).rolling(window=window, min_periods=window).max()
        prior_low = close.shift(1).rolling(window=window, min_periods=window).min()
        frame[f"high_breakout_{window}d"] = _bool_as_float(close > prior_high, prior_high.notna())
        frame[f"low_breakdown_{window}d"] = _bool_as_float(close < prior_low, prior_low.notna())


def _add_relative_strength(
    frame: pd.DataFrame,
    close: pd.Series,
    benchmark_close: pd.Series,
    benchmark_name: str,
) -> None:
    ratio = close / benchmark_close
    for d in [20, 50]:
        frame[f"rs_vs_{benchmark_name}_{d}d"] = ratio / ratio.shift(d) - 1.0


def _market_close(market: pd.DataFrame, prefix: str) -> pd.Series | None:
    col = f"{prefix}_close"
    if market.empty or col not in market.columns:
        return None
    return pd.to_numeric(market[col], errors="coerce")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.set_index("date")
        else:
            out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "date"
    out.columns = [str(c).lower() for c in out.columns]
    return out.sort_index()


def _dedupe(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def _bool_as_float(condition: pd.Series, valid: pd.Series) -> pd.Series:
    return pd.Series(np.where(valid, condition.astype(float), np.nan), index=condition.index)


def _flag(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce").fillna(0) > 0


def _gt(frame: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce") > threshold


def _lt(frame: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce") < threshold


def _gt_or_unavailable(frame: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    if col not in frame.columns or frame[col].isna().all():
        return pd.Series(True, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce") > threshold
