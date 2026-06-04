"""Volume-fade reversal entries."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.entries.base import BaseEntry, make_signal_frame
from src.entries.price_extreme_signals import HighestCloseSignal, LowestCloseSignal
from src.filters.base import ALLOW_LONG, ALLOW_SHORT, FilterProtocol, validate_filter_mask
from src.filters.volume_filter import VolumeAboveAverageFilter


@dataclass
class VolumeFadeLongEntry(BaseEntry):
    """Long entry for high-volume short-term weakness."""

    extreme_signal: LowestCloseSignal = field(default_factory=LowestCloseSignal)
    volume_filter: VolumeAboveAverageFilter = field(default_factory=VolumeAboveAverageFilter)
    filters: list[FilterProtocol] = field(default_factory=list)
    direction: str = "long"
    label: str = "volume_fade_long"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "direction": self.direction,
            "extreme_signal": self.extreme_signal.parameters,
            "volume_filter": self.volume_filter.parameters,
            "filters": [
                {"name": filter_module.name, "parameters": filter_module.parameters}
                for filter_module in self.filters
            ],
        }

    @property
    def min_lookback(self) -> int:
        return max(
            getattr(self.extreme_signal, "min_lookback", 0),
            getattr(self.volume_filter, "length", 0),
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        direction = str(self.direction).lower()
        if direction != "long":
            raise ValueError("VolumeFadeLongEntry currently supports direction='long'")

        extreme = self.extreme_signal.generate(data)
        volume_mask = self.volume_filter.generate_mask(data)
        allowed = volume_mask[ALLOW_LONG].astype(bool)

        for filter_module in self.filters:
            mask = filter_module.generate_mask(data)
            validate_filter_mask(data, mask, filter_module.name)
            allowed &= mask[ALLOW_LONG].astype(bool)

        long_signal = (extreme & allowed).fillna(False)
        signal = long_signal.astype(int)
        return make_signal_frame(
            data,
            signal,
            {
                "lowest_close_signal": extreme.astype(bool),
                "volume_above_average": volume_mask[ALLOW_LONG].astype(bool),
                "long_entry": long_signal.astype(bool),
            },
        )


@dataclass
class VolumeFadeReversalEntry(BaseEntry):
    """Two-sided volume-fade entry for long weakness and short strength."""

    long_extreme_signal: LowestCloseSignal = field(default_factory=LowestCloseSignal)
    short_extreme_signal: HighestCloseSignal = field(default_factory=HighestCloseSignal)
    volume_filter: VolumeAboveAverageFilter = field(default_factory=VolumeAboveAverageFilter)
    long_filters: list[FilterProtocol] = field(default_factory=list)
    short_filters: list[FilterProtocol] = field(default_factory=list)
    enable_long_entries: bool = True
    enable_short_entries: bool = True
    label: str = "volume_fade_reversal"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "enable_long_entries": self.enable_long_entries,
            "enable_short_entries": self.enable_short_entries,
            "long_extreme_signal": self.long_extreme_signal.parameters,
            "short_extreme_signal": self.short_extreme_signal.parameters,
            "volume_filter": self.volume_filter.parameters,
            "long_filters": [
                {"name": filter_module.name, "parameters": filter_module.parameters}
                for filter_module in self.long_filters
            ],
            "short_filters": [
                {"name": filter_module.name, "parameters": filter_module.parameters}
                for filter_module in self.short_filters
            ],
        }

    @property
    def min_lookback(self) -> int:
        return max(
            getattr(self.long_extreme_signal, "min_lookback", 0),
            getattr(self.short_extreme_signal, "min_lookback", 0),
            getattr(self.volume_filter, "length", 0),
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        long_extreme = self.long_extreme_signal.generate(data)
        short_extreme = self.short_extreme_signal.generate(data)
        volume_mask = self.volume_filter.generate_mask(data)

        allow_long = volume_mask[ALLOW_LONG].astype(bool)
        allow_short = volume_mask[ALLOW_SHORT].astype(bool)

        for filter_module in self.long_filters:
            mask = filter_module.generate_mask(data)
            validate_filter_mask(data, mask, filter_module.name)
            allow_long &= mask[ALLOW_LONG].astype(bool)

        for filter_module in self.short_filters:
            mask = filter_module.generate_mask(data)
            validate_filter_mask(data, mask, filter_module.name)
            allow_short &= mask[ALLOW_SHORT].astype(bool)

        long_signal = (long_extreme & allow_long & bool(self.enable_long_entries)).fillna(False)
        short_signal = (short_extreme & allow_short & bool(self.enable_short_entries)).fillna(False)
        signal = long_signal.astype(int).mask(short_signal, -1)

        return make_signal_frame(
            data,
            signal,
            {
                "lowest_close_signal": long_extreme.astype(bool),
                "highest_close_signal": short_extreme.astype(bool),
                "volume_above_average": volume_mask[ALLOW_LONG].astype(bool),
                "long_entry": long_signal.astype(bool),
                "short_entry": short_signal.astype(bool),
            },
        )
