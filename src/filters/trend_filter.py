"""Trend filters."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import moving_average
from src.filters.base import BaseFilter, make_filter_frame


@dataclass
class TrendFilter(BaseFilter):
    """Allow trades only when price is on the configured side of a moving average."""

    length: int = 200
    ma_type: str = "sma"
    mode: str = "long_above_sma"
    enabled: bool = True
    label: str = "trend_filter"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
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

        close = data["close"]
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
        elif mode == "long_below_sma":
            allow_long = below
            allow_short = pd.Series(True, index=data.index)
        else:
            raise ValueError(f"Unsupported trend filter mode: {self.mode}")

        return make_filter_frame(
            data,
            allow_long,
            allow_short,
            {"trend_ma": ma},
        )
