"""Download and cache OHLCV data from Yahoo Finance via yfinance."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .features import compute_market_regime_features

logger = logging.getLogger(__name__)


def download_ticker(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV history for a single ticker using yfinance.

    Returns a DataFrame with lowercase columns (open, high, low, close, volume)
    and a tz-naive DatetimeIndex named 'date'. Returns empty DataFrame on failure.
    """
    try:
        obj = yf.Ticker(ticker)
        df = obj.history(start=start, end=end, interval=interval, auto_adjust=True)
    except Exception as exc:
        logger.warning(f"{ticker}: download failed — {exc}")
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"{ticker}: no data returned")
        return pd.DataFrame()

    # Remove timezone info so all dates are tz-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df.dropna(subset=["close"], inplace=True)
    return df


def download_universe(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for all tickers with optional per-ticker parquet cache.

    On the first run, data is fetched from Yahoo Finance and saved to cache_dir.
    On subsequent runs, cached files are loaded directly (much faster).
    To force a re-download, delete the relevant .parquet files from cache_dir.
    """
    data: dict[str, pd.DataFrame] = {}

    for ticker in tqdm(tickers, desc="Downloading tickers"):
        cache_path: Path | None = None

        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{ticker}.parquet"
            if cache_path.exists():
                logger.info(f"{ticker}: loading from cache ({cache_path})")
                data[ticker] = pd.read_parquet(cache_path)
                continue

        df = download_ticker(ticker, start=start, end=end, interval=interval)

        if df.empty:
            logger.warning(f"{ticker}: skipping — no data")
            continue

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.debug(f"{ticker}: cached to {cache_path}")

        data[ticker] = df

    logger.info(f"Downloaded {len(data)}/{len(tickers)} tickers successfully")
    return data


def download_market_regime(
    start: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download SPY and QQQ, compute EMA-based regime features, and return a
    date-indexed DataFrame with all regime columns.

    Caches result to MARKET_REGIME.parquet inside cache_dir if provided.
    Delete that file to force a re-download.

    Returns empty DataFrame if SPY or QQQ download fails.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "MARKET_REGIME.parquet"
        if cache_path.exists():
            logger.info("Market regime: loading from cache")
            df = pd.read_parquet(cache_path)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df

    logger.info("Downloading SPY and QQQ for market regime features ...")
    spy_df = download_ticker("SPY", start=start)
    qqq_df = download_ticker("QQQ", start=start)

    if spy_df.empty or qqq_df.empty:
        logger.error("SPY or QQQ download failed — market regime features unavailable")
        return pd.DataFrame()

    regime = compute_market_regime_features(spy_df["close"], qqq_df["close"])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        regime.to_parquet(cache_path)
        logger.info(f"Market regime cached to {cache_path}")

    return regime
