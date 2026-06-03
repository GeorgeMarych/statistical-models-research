"""Market regime state panel builder.

This module builds daily market/global regime features using only data known at
or before each date. Forward returns are computed only in summary functions and
are not used by the regime label logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.data_loader import download_ticker

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning, module=__name__)


DEFAULT_MARKET_REGIME_UNIVERSE: dict[str, list[str]] = {
    "core": ["SPY", "QQQ", "RSP", "IWM"],
    "breadth": ["MMFI", "MMTH", "HIGN", "LOWN"],
    "fear": ["VIX", "PCC"],
    "rates": ["US02Y", "US10Y", "T10Y2"],
    "macro": ["DXY", "GC1", "SI1", "HG1", "CL1"],
    "sectors": ["XLK", "XLC", "XLY", "XLF", "XLU", "XLE", "XLP", "XLV", "XLI", "XLB", "XLRE"],
}

# Logical research symbol -> yfinance candidates. Some requested breadth symbols
# are TradingView-style symbols with no reliable yfinance mapping, so they are
# left unmapped and reported as unavailable.
YFINANCE_SYMBOL_CANDIDATES: dict[str, list[str]] = {
    "SPY": ["SPY"],
    "QQQ": ["QQQ"],
    "RSP": ["RSP"],
    "IWM": ["IWM"],
    "VIX": ["^VIX"],
    "PCC": ["^CPC"],
    "US02Y": ["^UST2Y"],
    "US10Y": ["^TNX"],
    "DXY": ["DX-Y.NYB"],
    "GC1": ["GC=F"],
    "SI1": ["SI=F"],
    "HG1": ["HG=F"],
    "CL1": ["CL=F"],
}

UNMAPPED_SYMBOLS: dict[str, str] = {
    "MMFI": "TradingView/market-breadth symbol; no reliable yfinance mapping configured",
    "MMTH": "TradingView/market-breadth symbol; no reliable yfinance mapping configured",
    "HIGN": "TradingView/market-breadth symbol; no reliable yfinance mapping configured",
    "LOWN": "TradingView/market-breadth symbol; no reliable yfinance mapping configured",
    "T10Y2": "Synthetic series computed from US10Y minus US02Y when both are available",
}

RATE_SYMBOLS = {"US02Y", "US10Y", "T10Y2"}
MACRO_RETURN_SYMBOLS = {"DXY", "GC1", "SI1", "HG1", "CL1"}


@dataclass
class SymbolLoadResult:
    """Loaded market regime data and unavailable-symbol diagnostics."""

    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    resolved_symbols: dict[str, str] = field(default_factory=dict)


def normalize_market_regime_universe(raw: dict[str, Any] | None) -> dict[str, list[str]]:
    """Return a normalized market regime universe with default groups filled in."""
    source = raw or {}
    normalized: dict[str, list[str]] = {}
    for group, defaults in DEFAULT_MARKET_REGIME_UNIVERSE.items():
        values = source.get(group, defaults)
        normalized[group] = _normalize_symbols(values)
    return normalized


def flatten_market_regime_symbols(universe: dict[str, list[str]]) -> list[str]:
    """Return all logical symbols in universe order without duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for group in ["core", "breadth", "fear", "rates", "macro", "sectors"]:
        for symbol in universe.get(group, []):
            if symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
    return out


def load_market_regime_symbol_data(
    universe: dict[str, list[str]],
    start: str,
    interval: str,
    cache_dir: str | Path,
    existing_data: dict[str, pd.DataFrame] | None = None,
) -> SymbolLoadResult:
    """
    Load available market-regime symbols from cache/yfinance.

    Missing symbols are recorded in the result and do not abort the pipeline.
    Data is cached by logical research symbol, not by yfinance symbol, so
    TradingView-style names such as `GC1` can remain stable in downstream code.
    """
    result = SymbolLoadResult()
    existing_data = existing_data or {}
    cache_root = Path(cache_dir)

    for symbol in flatten_market_regime_symbols(universe):
        if symbol == "T10Y2":
            result.missing[symbol] = UNMAPPED_SYMBOLS[symbol]
            continue

        if symbol in existing_data and not existing_data[symbol].empty:
            result.data[symbol] = _normalize_ohlcv(existing_data[symbol])
            result.resolved_symbols[symbol] = symbol
            continue

        cache_path = cache_root / f"{symbol}.parquet"
        if cache_path.exists():
            try:
                result.data[symbol] = _normalize_ohlcv(pd.read_parquet(cache_path))
                result.resolved_symbols[symbol] = symbol
                continue
            except Exception as exc:
                logger.warning("%s: cached regime data could not be read: %s", symbol, exc)

        candidates = YFINANCE_SYMBOL_CANDIDATES.get(symbol)
        if not candidates:
            result.missing[symbol] = UNMAPPED_SYMBOLS.get(
                symbol,
                "no yfinance mapping configured",
            )
            continue

        loaded = False
        for yf_symbol in candidates:
            df = download_ticker(yf_symbol, start=start, interval=interval)
            if df.empty:
                continue
            df = _normalize_ohlcv(df)
            df = _normalize_yield_scale(symbol, df)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            result.data[symbol] = df
            result.resolved_symbols[symbol] = yf_symbol
            loaded = True
            break

        if not loaded:
            result.missing[symbol] = (
                "no data returned from yfinance candidates: "
                + ", ".join(candidates)
            )

    return result


def build_market_regime_daily(
    universe: dict[str, list[str]],
    symbol_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one daily market/global regime row per date."""
    index = _union_index(symbol_data)
    if index.empty:
        return pd.DataFrame()

    panel = pd.DataFrame(index=index)
    panel.index.name = "date"

    for symbol in universe.get("core", []):
        close = _close(symbol_data.get(symbol), index)
        if close is not None:
            _add_symbol_trend_features(panel, symbol, close)

    _add_rsp_spy_features(panel, symbol_data, index)

    for symbol in universe.get("breadth", []):
        close = _close(symbol_data.get(symbol), index)
        if close is not None:
            _add_level_change_features(panel, symbol, close, change_kind="diff")

    _add_new_high_low_spread(panel)

    for symbol in universe.get("fear", []):
        close = _close(symbol_data.get(symbol), index)
        if close is not None:
            _add_level_change_features(panel, symbol, close, change_kind="diff")

    for symbol in universe.get("rates", []):
        if symbol == "T10Y2":
            continue
        close = _close(symbol_data.get(symbol), index)
        if close is not None:
            _add_level_change_features(panel, symbol, close, change_kind="diff")
    _add_t10y2_synthetic(panel)

    for symbol in universe.get("macro", []):
        close = _close(symbol_data.get(symbol), index)
        if close is not None:
            _add_level_change_features(panel, symbol, close, change_kind="return")

    core_cols = [col for col in ["spy_close", "qqq_close"] if col in panel.columns]
    if core_cols:
        panel = panel[panel[core_cols].notna().any(axis=1)].copy()

    panel["market_regime"] = classify_market_regime(panel)
    return panel.reset_index()


def classify_market_regime(panel: pd.DataFrame) -> pd.Series:
    """
    Assign transparent broad market regime labels.

    The logic is intentionally simple. It uses only current and trailing market,
    breadth, fear, and moving-average features.
    """
    index = panel.index

    core_available = _has_value(panel, "spy_close") & _has_value(panel, "qqq_close")
    spy_strong = _trend_strong(panel, "spy")
    qqq_strong = _trend_strong(panel, "qqq")
    spy_holding = _has_value(panel, "spy_close") & (_above(panel, "spy", 50) | _above(panel, "spy", 200))
    qqq_holding = _has_value(panel, "qqq_close") & (_above(panel, "qqq", 50) | _above(panel, "qqq", 200))
    spy_weak = _has_value(panel, "spy_close") & (~_above(panel, "spy", 50)) & (~_above(panel, "spy", 200))
    qqq_weak = _has_value(panel, "qqq_close") & (~_above(panel, "qqq", 50)) & (~_above(panel, "qqq", 200))

    indexes_strong = core_available & spy_strong & qqq_strong
    indexes_holding = core_available & spy_holding & qqq_holding
    indexes_weak = core_available & spy_weak & qqq_weak

    rsp_improving = _gt(panel, "rsp_spy_ratio_change_20d", 0.0)
    rsp_weak = _lt(panel, "rsp_spy_ratio_change_20d", 0.0)
    has_breadth = _has_any(panel, ["rsp_spy_ratio_change_20d", "mmfi_change_20d", "mmth_change_20d"])

    breadth_improving = (
        rsp_improving
        | _gt(panel, "mmfi_change_20d", 0.0)
        | _gt(panel, "mmth_change_20d", 0.0)
        | _gt(panel, "hign_change_20d", 0.0)
        | _lt(panel, "lown_change_20d", 0.0)
    )
    breadth_weak = (
        rsp_weak
        | _lt(panel, "mmfi_change_20d", 0.0)
        | _lt(panel, "mmth_change_20d", 0.0)
        | _lt(panel, "hign_change_20d", 0.0)
        | _gt(panel, "lown_change_20d", 0.0)
        | _lt(panel, "new_highs_minus_lows", 0.0)
    )

    fear_rising = (
        _gt(panel, "vix_change_20d", 5.0)
        | _gt(panel, "vix_change_5d", 3.0)
        | _gt(panel, "pcc_change_20d", 0.20)
    )
    vix_spike = _gt(panel, "vix_level", 35.0) | _gt(panel, "vix_change_5d", 7.0) | _gt(panel, "vix_change_20d", 10.0)
    indexes_breaking = (
        _flag(panel, "spy_low_breakdown_20d")
        | _flag(panel, "spy_low_breakdown_50d")
        | _flag(panel, "qqq_low_breakdown_20d")
        | _flag(panel, "qqq_low_breakdown_50d")
        | indexes_weak
    )

    panic = core_available & (
        (vix_spike & indexes_breaking)
        | (indexes_breaking & _gt(panel, "lown_change_20d", 0.0))
    )
    risk_off = indexes_weak & (breadth_weak | fear_rising | indexes_breaking)
    risk_on_broad = indexes_strong & (breadth_improving & has_breadth) & ~fear_rising
    risk_on_narrow = indexes_strong & ~risk_on_broad & ~fear_rising
    fragile = (indexes_holding & (breadth_weak | fear_rising)) | (indexes_strong & fear_rising)

    labels = np.select(
        [panic, risk_off, fragile, risk_on_broad, risk_on_narrow],
        ["panic", "risk_off", "fragile", "risk_on_broad", "risk_on_narrow"],
        default="neutral",
    )
    return pd.Series(labels, index=index, name="market_regime")


def build_market_regime_summary(
    market_panel: pd.DataFrame,
    days: list[int],
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Average forward SPY/QQQ/RSP returns by market regime."""
    if market_panel.empty or "market_regime" not in market_panel.columns:
        return pd.DataFrame()

    symbols = symbols or ["SPY", "QQQ", "RSP"]
    panel = market_panel.sort_values("date").copy()
    rows: list[dict[str, Any]] = []

    for regime in sorted(panel["market_regime"].dropna().unique()):
        subset = panel[panel["market_regime"] == regime]
        row: dict[str, Any] = {"market_regime": regime, "rows": len(subset)}
        for symbol in symbols:
            prefix = symbol.lower()
            close_col = f"{prefix}_close"
            if close_col not in panel.columns:
                continue
            close = pd.to_numeric(panel[close_col], errors="coerce")
            for d in days:
                fwd = close.shift(-d) / close - 1.0
                values = fwd.loc[subset.index].dropna()
                row[f"{prefix}_valid_{d}d_rows"] = len(values)
                row[f"{prefix}_avg_forward_{d}d_return"] = float(values.mean()) if len(values) else np.nan
                row[f"{prefix}_median_forward_{d}d_return"] = float(values.median()) if len(values) else np.nan
                row[f"{prefix}_forward_{d}d_win_rate"] = float((values > 0).mean()) if len(values) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_market_regime_console_summary(
    market_panel: pd.DataFrame,
    missing: dict[str, str],
) -> list[str]:
    """Compact console summary for the market regime layer."""
    if market_panel.empty:
        return ["Market regime panel unavailable."]

    latest = market_panel.sort_values("date").iloc[-1]
    counts = market_panel["market_regime"].value_counts(dropna=False)
    count_text = ", ".join(f"{k}: {v}" for k, v in counts.items())
    lines = [
        f"Rows: {len(market_panel):,}",
        f"Date range: {market_panel['date'].min().date()} -> {market_panel['date'].max().date()}",
        f"Latest regime: {latest['market_regime']} ({latest['date'].date()})",
        f"Regime counts: {count_text}",
    ]
    if missing:
        missing_text = ", ".join(f"{symbol} ({reason})" for symbol, reason in missing.items())
        lines.append(f"Unavailable symbols: {missing_text}")
    else:
        lines.append("Unavailable symbols: none")
    return lines


def _normalize_symbols(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


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


def _normalize_yield_scale(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if symbol not in {"US02Y", "US10Y"} or "close" not in df.columns:
        return df
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty or close.median() <= 20:
        return df
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            out[col] = out[col] / 10.0
    return out


def _union_index(symbol_data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    indexes = [
        _normalize_ohlcv(df).index
        for df in symbol_data.values()
        if df is not None and not df.empty
    ]
    if not indexes:
        return pd.DatetimeIndex([], name="date")
    index = indexes[0]
    for other in indexes[1:]:
        index = index.union(other)
    return pd.DatetimeIndex(index).sort_values()


def _close(df: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series | None:
    if df is None or df.empty:
        return None
    normalized = _normalize_ohlcv(df)
    if "close" not in normalized.columns:
        return None
    return pd.to_numeric(normalized["close"], errors="coerce").reindex(index)


def _add_symbol_trend_features(panel: pd.DataFrame, symbol: str, close: pd.Series) -> None:
    prefix = symbol.lower()
    panel[f"{prefix}_close"] = close
    for d in [5, 10, 20]:
        panel[f"{prefix}_return_{d}d"] = close / close.shift(d) - 1.0

    for window in [20, 50, 200]:
        sma = close.rolling(window=window, min_periods=window).mean()
        panel[f"{prefix}_sma{window}"] = sma
        panel[f"{prefix}_close_above_sma{window}"] = _bool_as_float(close > sma, sma.notna())
        panel[f"{prefix}_distance_sma{window}"] = close / sma - 1.0
        panel[f"{prefix}_sma{window}_slope_5d"] = sma / sma.shift(5) - 1.0

    for window in [20, 50]:
        prior_high = close.shift(1).rolling(window=window, min_periods=window).max()
        prior_low = close.shift(1).rolling(window=window, min_periods=window).min()
        panel[f"{prefix}_high_breakout_{window}d"] = _bool_as_float(close > prior_high, prior_high.notna())
        panel[f"{prefix}_low_breakdown_{window}d"] = _bool_as_float(close < prior_low, prior_low.notna())


def _add_rsp_spy_features(
    panel: pd.DataFrame,
    symbol_data: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
) -> None:
    rsp = _close(symbol_data.get("RSP"), index)
    spy = _close(symbol_data.get("SPY"), index)
    if rsp is None or spy is None:
        return
    ratio = rsp / spy
    panel["rsp_spy_ratio"] = ratio
    panel["rsp_spy_ratio_change_5d"] = ratio / ratio.shift(5) - 1.0
    panel["rsp_spy_ratio_change_20d"] = ratio / ratio.shift(20) - 1.0


def _add_level_change_features(
    panel: pd.DataFrame,
    symbol: str,
    close: pd.Series,
    change_kind: str,
) -> None:
    prefix = symbol.lower()
    panel[f"{prefix}_level"] = close
    for d in [5, 20]:
        if change_kind == "return":
            panel[f"{prefix}_return_{d}d"] = close / close.shift(d) - 1.0
        else:
            panel[f"{prefix}_change_{d}d"] = close - close.shift(d)


def _add_new_high_low_spread(panel: pd.DataFrame) -> None:
    if "hign_level" in panel.columns and "lown_level" in panel.columns:
        panel["new_highs_minus_lows"] = panel["hign_level"] - panel["lown_level"]


def _add_t10y2_synthetic(panel: pd.DataFrame) -> None:
    if "us10y_level" not in panel.columns or "us02y_level" not in panel.columns:
        return
    spread = panel["us10y_level"] - panel["us02y_level"]
    panel["t10y2_level"] = spread
    panel["t10y2_change_5d"] = spread - spread.shift(5)
    panel["t10y2_change_20d"] = spread - spread.shift(20)


def _bool_as_float(condition: pd.Series, valid: pd.Series) -> pd.Series:
    return pd.Series(np.where(valid, condition.astype(float), np.nan), index=condition.index)


def _above(panel: pd.DataFrame, prefix: str, window: int) -> pd.Series:
    return _flag(panel, f"{prefix}_close_above_sma{window}")


def _trend_strong(panel: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        _above(panel, prefix, 20)
        & _above(panel, prefix, 50)
        & _gt(panel, f"{prefix}_sma20_slope_5d", 0.0)
        & _gt(panel, f"{prefix}_sma50_slope_5d", 0.0)
    )


def _flag(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(False, index=panel.index)
    return pd.to_numeric(panel[col], errors="coerce").fillna(0) > 0


def _gt(panel: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(False, index=panel.index)
    return pd.to_numeric(panel[col], errors="coerce") > threshold


def _lt(panel: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(False, index=panel.index)
    return pd.to_numeric(panel[col], errors="coerce") < threshold


def _has_any(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [col for col in cols if col in panel.columns]
    if not available:
        return pd.Series(False, index=panel.index)
    return panel[available].notna().any(axis=1)


def _has_value(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(False, index=panel.index)
    return panel[col].notna()
