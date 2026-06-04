"""Opposite-strength exits for volume-fade strategies."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.entries.price_extreme_signals import HighestCloseSignal, LowestCloseSignal
from src.exits.base import BaseExit, ExitDecision, PositionContext
from src.filters.base import ALLOW_LONG, ALLOW_SHORT, FilterProtocol, validate_filter_mask
from src.filters.volume_filter import VolumeAboveAverageFilter


@dataclass
class VolumeStrengthSignal:
    """High-volume close-at-extreme signal used by exits."""

    direction: str = "short"
    extreme_signal: HighestCloseSignal | LowestCloseSignal = field(default_factory=HighestCloseSignal)
    volume_filter: VolumeAboveAverageFilter = field(default_factory=VolumeAboveAverageFilter)
    filters: list[FilterProtocol] = field(default_factory=list)
    label: str = "volume_strength_signal"

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

    def generate(self, data: pd.DataFrame) -> pd.Series:
        extreme = self.extreme_signal.generate(data)
        volume_mask = self.volume_filter.generate_mask(data)
        direction = str(self.direction).lower()
        allow_column = ALLOW_SHORT if direction in {"short", "sell"} else ALLOW_LONG
        allowed = volume_mask[allow_column].astype(bool)
        for filter_module in self.filters:
            mask = filter_module.generate_mask(data)
            validate_filter_mask(data, mask, filter_module.name)
            allowed &= mask[allow_column].astype(bool)
        return (extreme & allowed).fillna(False)


@dataclass
class OppositeStrengthExit(BaseExit):
    """Exit when an opposite high-volume strength signal appears."""

    signal: VolumeStrengthSignal = field(default_factory=VolumeStrengthSignal)
    reverse_on_opposite: bool = False
    label: str = "opposite_strength"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "signal": self.signal.parameters,
            "reverse_on_opposite": self.reverse_on_opposite,
        }

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["opposite_strength"] = self.signal.generate(data)
        return out

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if context.get("phase") != "close":
            return ExitDecision.no_action()

        prepared = (context.get("prepared_exits") or {}).get(self.name)
        if prepared is None or "opposite_strength" not in prepared.columns:
            return ExitDecision.no_action()

        if not bool(prepared["opposite_strength"].iloc[index]):
            return ExitDecision.no_action()

        signal_direction = str(self.signal.direction).lower()
        exits_long = position.side == 1 and signal_direction in {"short", "sell", "opposite"}
        exits_short = position.side == -1 and signal_direction in {"long", "buy", "opposite"}
        if not exits_long and not exits_short:
            return ExitDecision.no_action()

        reverse_side = None
        if self.reverse_on_opposite:
            reverse_side = -position.side
            if reverse_side == -1 and not context.get("allow_short", False):
                reverse_side = None

        return ExitDecision(
            should_exit=True,
            reason="opposite_strength",
            timing="next_open",
            reverse_to_side=reverse_side,
        )
