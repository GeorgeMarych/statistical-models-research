"""Volume and dollar-volume filters."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.filters.base import BaseFilter, make_filter_frame


@dataclass
class VolumeFilter(BaseFilter):
    """Allow trades when trailing average dollar volume is large enough."""

    length: int = 20
    min_avg_dollar_volume: float = 25_000_000.0
    label: str = "volume_filter"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "length": self.length,
            "min_avg_dollar_volume": self.min_avg_dollar_volume,
        }

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        dollar_volume = data["close"] * data["volume"]
        average = dollar_volume.rolling(self.length, min_periods=self.length).mean()
        mask = average >= self.min_avg_dollar_volume
        return make_filter_frame(data, mask, mask, {"avg_dollar_volume": average})


@dataclass
class VolumeAboveAverageFilter(BaseFilter):
    """Allow bars where volume is above its trailing average times a multiplier."""

    length: int = 10
    multiplier: float = 1.0
    label: str = "volume_above_average"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"length": self.length, "multiplier": self.multiplier}

    def condition(self, data: pd.DataFrame) -> pd.Series:
        if self.length <= 0:
            raise ValueError("length must be positive")
        average = data["volume"].rolling(self.length, min_periods=self.length).mean()
        return (data["volume"] > average * self.multiplier).fillna(False)

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        average = data["volume"].rolling(self.length, min_periods=self.length).mean()
        mask = (data["volume"] > average * self.multiplier).fillna(False)
        return make_filter_frame(data, mask, mask, {"avg_volume": average})
