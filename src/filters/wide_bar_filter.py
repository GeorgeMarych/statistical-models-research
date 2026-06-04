"""Wide-bar filters used by entries and exits."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.filters.base import BaseFilter, make_filter_frame


@dataclass
class WideBullishBarFilter(BaseFilter):
    """
    Detect wide bullish bars.

    When block_when_true is true, the generated filter allows entries/exits only
    when the wide bullish condition is false.
    """

    range_length: int = 10
    range_mult: float = 1.0
    block_when_true: bool = True
    enabled: bool = True
    apply_to_long: bool = True
    apply_to_short: bool = True
    label: str = "wide_bullish_bar"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "range_length": self.range_length,
            "range_mult": self.range_mult,
            "block_when_true": self.block_when_true,
            "enabled": self.enabled,
            "apply_to_long": self.apply_to_long,
            "apply_to_short": self.apply_to_short,
        }

    def condition(self, data: pd.DataFrame) -> pd.Series:
        if self.range_length <= 0:
            raise ValueError("range_length must be positive")
        bar_range = data["high"] - data["low"]
        average_range = bar_range.rolling(
            self.range_length,
            min_periods=self.range_length,
        ).mean()
        wide = bar_range > average_range * self.range_mult
        bullish = data["close"] > data["open"]
        return (wide & bullish).fillna(False)

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        if not self.enabled:
            return make_filter_frame(data, True, True)
        condition = self.condition(data)
        mask = ~condition if self.block_when_true else condition
        allow_long = mask if self.apply_to_long else pd.Series(True, index=data.index)
        allow_short = mask if self.apply_to_short else pd.Series(True, index=data.index)
        return make_filter_frame(data, allow_long, allow_short, {"wide_bullish_bar": condition})
