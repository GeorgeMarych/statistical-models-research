"""Broad market regime filters."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import moving_average
from src.filters.base import BaseFilter, make_filter_frame


@dataclass
class MarketRegimeFilter(BaseFilter):
    """Gate trades using a benchmark symbol such as SPY."""

    benchmark_symbol: str = "SPY"
    length: int = 200
    ma_type: str = "sma"
    mode: str = "long_above_sma"
    enabled: bool = True
    label: str = "market_regime_filter"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "benchmark_symbol": self.benchmark_symbol,
            "length": self.length,
            "ma_type": self.ma_type,
            "mode": self.mode,
            "enabled": self.enabled,
        }

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        if not self.enabled:
            return make_filter_frame(data, True, True)

        context = context or {}
        market_data = context.get("market_data", {})
        benchmark = market_data.get(self.benchmark_symbol.upper())
        if benchmark is None or benchmark.empty:
            raise ValueError(f"{self.name}: missing benchmark data for {self.benchmark_symbol}")

        close = benchmark["close"].reindex(data.index).ffill()
        ma = moving_average(close, self.length, self.ma_type)
        above = close > ma
        below = close < ma
        mode = str(self.mode).lower()

        if mode in {"long_above_sma", "close_above_sma"}:
            allow_long = above
            allow_short = pd.Series(True, index=data.index)
        elif mode in {"short_below_sma", "close_below_sma"}:
            allow_long = pd.Series(True, index=data.index)
            allow_short = below
        elif mode == "long_above_short_below":
            allow_long = above
            allow_short = below
        else:
            raise ValueError(f"Unsupported market regime filter mode: {self.mode}")

        return make_filter_frame(
            data,
            allow_long,
            allow_short,
            {
                "benchmark_close": close,
                "benchmark_ma": ma,
            },
        )
