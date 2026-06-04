"""Small price-pattern entry examples."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class PricePatternEntry(BaseEntry):
    """Enter on simple bullish/bearish outside bars."""

    pattern: str = "outside_bar"
    include_short_signals: bool = True
    label: str = "price_pattern"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "pattern": self.pattern,
            "include_short_signals": self.include_short_signals,
        }

    @property
    def min_lookback(self) -> int:
        return 2

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pattern = str(self.pattern).lower()
        if pattern != "outside_bar":
            raise ValueError("Only outside_bar is currently supported")

        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)
        outside = (data["high"] > prev_high) & (data["low"] < prev_low)
        bullish = outside & (data["close"] > data["open"])
        bearish = outside & (data["close"] < data["open"])
        signal = bullish.astype(int)
        if self.include_short_signals:
            signal = signal.mask(bearish, -1)
        return make_signal_frame(
            data,
            signal,
            {
                "outside_bar": outside.astype(bool),
                "long_entry": bullish.astype(bool),
                "short_entry": bearish.astype(bool),
            },
        )
