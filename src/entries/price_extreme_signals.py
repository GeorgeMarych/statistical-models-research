"""Reusable close-at-extreme signal components."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LowestCloseSignal:
    """Signal when the current close is the lowest close over a trailing window."""

    length: int = 2
    label: str = "lowest_close"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"length": self.length}

    @property
    def min_lookback(self) -> int:
        return self.length

    def generate(self, data: pd.DataFrame) -> pd.Series:
        if self.length <= 0:
            raise ValueError("length must be positive")
        trailing_low = data["close"].rolling(self.length, min_periods=self.length).min()
        return (data["close"] <= trailing_low).fillna(False)


@dataclass
class HighestCloseSignal:
    """Signal when the current close is the highest close over a trailing window."""

    length: int = 11
    label: str = "highest_close"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"length": self.length}

    @property
    def min_lookback(self) -> int:
        return self.length

    def generate(self, data: pd.DataFrame) -> pd.Series:
        if self.length <= 0:
            raise ValueError("length must be positive")
        trailing_high = data["close"].rolling(self.length, min_periods=self.length).max()
        return (data["close"] >= trailing_high).fillna(False)
