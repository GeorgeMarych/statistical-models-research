"""Universe management — resolve and filter the ticker list."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_tickers(tickers: list[str]) -> list[str]:
    """Return deduplicated, uppercased tickers in original order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tickers:
        t = raw.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def filter_valid(
    data: dict[str, pd.DataFrame],
    min_bars: int = 100,
) -> list[str]:
    """
    Return only tickers that have at least min_bars rows of OHLCV data.
    Tickers below the threshold are logged and excluded.
    """
    valid: list[str] = []
    for ticker, df in data.items():
        if len(df) >= min_bars:
            valid.append(ticker)
        else:
            logger.warning(
                f"{ticker}: only {len(df)} bars — need {min_bars}. Dropping."
            )
    return valid
