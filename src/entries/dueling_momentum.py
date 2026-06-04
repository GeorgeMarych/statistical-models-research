"""Dueling momentum entry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class DuelingMomentumEntry(BaseEntry):
    """Compare fast and slow trailing returns to choose a side."""

    fast_length: int = 20
    slow_length: int = 60
    threshold: float = 0.0
    include_short_signals: bool = True
    label: str = "dueling_momentum"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "fast_length": self.fast_length,
            "slow_length": self.slow_length,
            "threshold": self.threshold,
            "include_short_signals": self.include_short_signals,
        }

    @property
    def min_lookback(self) -> int:
        return max(self.fast_length, self.slow_length)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        fast_return = close / close.shift(self.fast_length) - 1.0
        slow_return = close / close.shift(self.slow_length) - 1.0
        spread = fast_return - slow_return
        long_signal = (spread > self.threshold) & (fast_return > 0)
        short_signal = (spread < -self.threshold) & (fast_return < 0)
        signal = long_signal.astype(int)
        if self.include_short_signals:
            signal = signal.mask(short_signal, -1)
        return make_signal_frame(
            data,
            signal,
            {
                "fast_return": fast_return,
                "slow_return": slow_return,
                "momentum_spread": spread,
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
