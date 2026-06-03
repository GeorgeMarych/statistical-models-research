"""RSI plus Bollinger Band mean-reversion entry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import bollinger_bands, compute_rsi
from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class RsiBollingerEntry(BaseEntry):
    """Long mean-reversion entry when price is below the lower band and RSI is low."""

    rsi_length: int = 14
    rsi_threshold: float = 24.0
    bb_window: int = 20
    bb_stdev: float = 2.0
    enable_short: bool = False
    short_rsi_threshold: float = 76.0
    label: str = "rsi_bb_mean_reversion"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "rsi_length": self.rsi_length,
            "rsi_threshold": self.rsi_threshold,
            "bb_window": self.bb_window,
            "bb_stdev": self.bb_stdev,
            "enable_short": self.enable_short,
            "short_rsi_threshold": self.short_rsi_threshold,
        }

    @property
    def min_lookback(self) -> int:
        return max(self.rsi_length, self.bb_window)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        rsi = compute_rsi(close, self.rsi_length)
        bands = bollinger_bands(close, self.bb_window, self.bb_stdev)

        long_signal = (close < bands["bb_lower"]) & (rsi < self.rsi_threshold)
        signal = long_signal.astype(int)

        short_signal = pd.Series(False, index=data.index)
        if self.enable_short:
            short_signal = (close > bands["bb_upper"]) & (
                rsi > self.short_rsi_threshold
            )
            signal = signal.mask(short_signal, -1)

        return make_signal_frame(
            data,
            signal,
            {
                "rsi": rsi,
                "bb_middle": bands["bb_middle"],
                "bb_upper": bands["bb_upper"],
                "bb_lower": bands["bb_lower"],
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
