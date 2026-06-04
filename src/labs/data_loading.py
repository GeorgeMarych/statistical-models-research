"""Shared data loading helpers for strategy labs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.engine import prepare_ohlcv
from src.data_loader import download_universe


def normalize_symbols(symbols: list[str]) -> list[str]:
    """Return uppercase, deduped symbols."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def load_price_data(
    symbols: list[str],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data from parquet cache or yfinance-backed loader."""
    data_config = config.get("data", {}) or {}
    source = str(data_config.get("source", "cache_only")).lower()
    cache_dir = Path(data_config.get("cache_dir", "data/raw"))
    interval = str(config.get("interval", data_config.get("interval", "1d")))
    start = config.get("start_date", data_config.get("start_date", "2010-01-01"))
    end = config.get("end_date", data_config.get("end_date"))
    symbols = normalize_symbols(symbols)

    if source == "cache_only":
        raw_data = _read_cached_parquets(symbols, cache_dir)
    elif source in {"cache_or_yfinance", "yfinance"}:
        raw_data = download_universe(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError("data.source must be cache_only, cache_or_yfinance, or yfinance")

    out: dict[str, pd.DataFrame] = {}
    for symbol, raw in raw_data.items():
        if raw is None or raw.empty:
            continue
        prices = prepare_ohlcv(raw)
        if start:
            prices = prices[prices.index >= pd.Timestamp(start)]
        if end:
            prices = prices[prices.index <= pd.Timestamp(end)]
        if not prices.empty:
            out[symbol.upper()] = prices
    return out


def _read_cached_parquets(symbols: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path)
    return out
