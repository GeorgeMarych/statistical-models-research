"""Donchian close breakout entry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class DonchianBreakoutEntry(BaseEntry):
    """Enter on close breakouts beyond the prior trailing close channel."""

    lookback: int = 55
    include_short_signals: bool = True
    label: str = "donchian_breakout"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "lookback": self.lookback,
            "include_short_signals": self.include_short_signals,
        }

    @property
    def min_lookback(self) -> int:
        return self.lookback + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

        close = data["close"]
        prior_highest_close = (
            close.rolling(window=self.lookback, min_periods=self.lookback)
            .max()
            .shift(1)
        )
        prior_lowest_close = (
            close.rolling(window=self.lookback, min_periods=self.lookback)
            .min()
            .shift(1)
        )

        long_signal = close > prior_highest_close
        short_signal = close < prior_lowest_close
        signal = long_signal.astype(int)
        if self.include_short_signals:
            signal = signal.mask(short_signal, -1)

        return make_signal_frame(
            data,
            signal,
            {
                "donchian_high": prior_highest_close,
                "donchian_low": prior_lowest_close,
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
