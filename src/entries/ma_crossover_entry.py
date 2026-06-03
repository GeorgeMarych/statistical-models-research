"""Moving average crossover entry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import moving_average
from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class MovingAverageCrossoverEntry(BaseEntry):
    """Enter long on fast-over-slow crosses and optionally emit short signals."""

    fast_window: int = 20
    slow_window: int = 50
    ma_type: str = "sma"
    include_short_signals: bool = True
    label: str = "ma_crossover"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "ma_type": self.ma_type,
            "include_short_signals": self.include_short_signals,
        }

    @property
    def min_lookback(self) -> int:
        return max(self.fast_window, self.slow_window)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("moving-average windows must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window should be less than slow_window")

        close = data["close"]
        fast_ma = moving_average(close, self.fast_window, self.ma_type)
        slow_ma = moving_average(close, self.slow_window, self.ma_type)

        long_signal = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        short_signal = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signal = long_signal.astype(int)
        if self.include_short_signals:
            signal = signal.mask(short_signal, -1)

        return make_signal_frame(
            data,
            signal,
            {
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
