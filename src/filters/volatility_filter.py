"""Volatility filters."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import compute_atr
from src.filters.base import BaseFilter, make_filter_frame


@dataclass
class VolatilityFilter(BaseFilter):
    """Allow trades when ATR percentage is within configured bounds."""

    atr_length: int = 14
    min_atr_pct: float | None = None
    max_atr_pct: float | None = None
    label: str = "volatility_filter"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "atr_length": self.atr_length,
            "min_atr_pct": self.min_atr_pct,
            "max_atr_pct": self.max_atr_pct,
        }

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        atr = compute_atr(data, self.atr_length)
        atr_pct = atr / data["close"]
        mask = pd.Series(True, index=data.index)
        if self.min_atr_pct is not None:
            mask &= atr_pct >= self.min_atr_pct
        if self.max_atr_pct is not None:
            mask &= atr_pct <= self.max_atr_pct
        return make_filter_frame(data, mask, mask, {"atr_pct": atr_pct})
