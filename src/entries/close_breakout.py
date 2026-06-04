"""Close-based breakout entry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class CloseBreakoutEntry(BaseEntry):
    """Enter when close breaks the prior trailing close channel."""

    length: int = 20
    include_short_signals: bool = True
    label: str = "close_breakout"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "length": self.length,
            "include_short_signals": self.include_short_signals,
        }

    @property
    def min_lookback(self) -> int:
        return self.length + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        prior_high = close.rolling(self.length, min_periods=self.length).max().shift(1)
        prior_low = close.rolling(self.length, min_periods=self.length).min().shift(1)
        long_signal = close > prior_high
        short_signal = close < prior_low
        signal = long_signal.astype(int)
        if self.include_short_signals:
            signal = signal.mask(short_signal, -1)
        return make_signal_frame(
            data,
            signal,
            {
                "breakout_high": prior_high,
                "breakout_low": prior_low,
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
