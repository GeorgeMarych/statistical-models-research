"""Stop-style breakout entry signals."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class StopBreakoutEntry(BaseEntry):
    """
    Emit a signal when intraday range breaks a prior high/low channel.

    The current engine still fills the next open. This entry records that a
    stop-style level was touched and leaves true stop-entry fills for a later,
    explicit engine extension.
    """

    length: int = 20
    include_short_signals: bool = True
    label: str = "stop_breakout"

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
        prior_high = data["high"].rolling(self.length, min_periods=self.length).max().shift(1)
        prior_low = data["low"].rolling(self.length, min_periods=self.length).min().shift(1)
        long_signal = data["high"] > prior_high
        short_signal = data["low"] < prior_low
        signal = long_signal.astype(int)
        if self.include_short_signals:
            signal = signal.mask(short_signal, -1)
        return make_signal_frame(
            data,
            signal,
            {
                "stop_breakout_high": prior_high,
                "stop_breakout_low": prior_low,
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
