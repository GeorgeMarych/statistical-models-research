"""Simple mean-reversion entries."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import compute_rsi
from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class MeanReversionEntry(BaseEntry):
    """Enter after a close at a trailing extreme with optional RSI confirmation."""

    length: int = 10
    rsi_length: int = 14
    long_rsi_max: float | None = 35.0
    short_rsi_min: float | None = 65.0
    include_short_signals: bool = False
    label: str = "mean_reversion"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "length": self.length,
            "rsi_length": self.rsi_length,
            "long_rsi_max": self.long_rsi_max,
            "short_rsi_min": self.short_rsi_min,
            "include_short_signals": self.include_short_signals,
        }

    @property
    def min_lookback(self) -> int:
        return max(self.length, self.rsi_length)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        trailing_low = close.rolling(self.length, min_periods=self.length).min()
        trailing_high = close.rolling(self.length, min_periods=self.length).max()
        rsi = compute_rsi(close, self.rsi_length)

        long_signal = close <= trailing_low
        if self.long_rsi_max is not None:
            long_signal &= rsi <= self.long_rsi_max

        short_signal = close >= trailing_high
        if self.short_rsi_min is not None:
            short_signal &= rsi >= self.short_rsi_min

        signal = long_signal.astype(int)
        if self.include_short_signals:
            signal = signal.mask(short_signal, -1)

        return make_signal_frame(
            data,
            signal,
            {
                "trailing_low": trailing_low,
                "trailing_high": trailing_high,
                "rsi": rsi,
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
