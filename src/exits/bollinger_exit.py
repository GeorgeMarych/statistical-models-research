"""Bollinger Band based exits."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import bollinger_bands
from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class MiddleBollingerExit(BaseExit):
    """Exit at the next open after close reaches the middle band."""

    bb_window: int = 20
    bb_stdev: float = 2.0
    label: str = "middle_bollinger"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"bb_window": self.bb_window, "bb_stdev": self.bb_stdev}

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return bollinger_bands(data["close"], self.bb_window, self.bb_stdev)

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if context.get("phase") != "close":
            return ExitDecision.no_action()
        prepared = context["prepared_exits"].get(self.name)
        if prepared is None:
            return ExitDecision.no_action()
        middle = prepared["bb_middle"].iloc[index]
        close = data["close"].iloc[index]
        if pd.isna(middle) or pd.isna(close):
            return ExitDecision.no_action()
        if position.side == 1 and close >= middle:
            return ExitDecision(True, "middle_bollinger", "next_open")
        if position.side == -1 and close <= middle:
            return ExitDecision(True, "middle_bollinger", "next_open")
        return ExitDecision.no_action()


@dataclass
class UpperBollingerExit(BaseExit):
    """Exit longs at the upper band and shorts at the lower band."""

    bb_window: int = 20
    bb_stdev: float = 2.0
    trigger: str = "touch"
    touch_uses_previous_band: bool = True
    label: str = "upper_bollinger"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "bb_window": self.bb_window,
            "bb_stdev": self.bb_stdev,
            "trigger": self.trigger,
            "touch_uses_previous_band": self.touch_uses_previous_band,
        }

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        bands = bollinger_bands(data["close"], self.bb_window, self.bb_stdev)
        bands["bb_upper_prev"] = bands["bb_upper"].shift(1)
        bands["bb_lower_prev"] = bands["bb_lower"].shift(1)
        return bands

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        prepared = context["prepared_exits"].get(self.name)
        if prepared is None:
            return ExitDecision.no_action()

        trigger = str(self.trigger).lower()
        if trigger not in {"touch", "close"}:
            raise ValueError("trigger must be 'touch' or 'close'")

        if trigger == "touch":
            if context.get("phase") != "intrabar":
                return ExitDecision.no_action()
            upper_col = "bb_upper_prev" if self.touch_uses_previous_band else "bb_upper"
            lower_col = "bb_lower_prev" if self.touch_uses_previous_band else "bb_lower"
            if position.side == 1:
                level = prepared[upper_col].iloc[index]
                if pd.notna(level) and data["high"].iloc[index] >= level:
                    return ExitDecision(True, "upper_bollinger_touch", "intrabar", float(level))
            else:
                level = prepared[lower_col].iloc[index]
                if pd.notna(level) and data["low"].iloc[index] <= level:
                    return ExitDecision(True, "lower_bollinger_touch", "intrabar", float(level))
            return ExitDecision.no_action()

        if context.get("phase") != "close":
            return ExitDecision.no_action()
        if position.side == 1:
            level = prepared["bb_upper"].iloc[index]
            if pd.notna(level) and data["close"].iloc[index] >= level:
                return ExitDecision(True, "upper_bollinger_close", "next_open")
        else:
            level = prepared["bb_lower"].iloc[index]
            if pd.notna(level) and data["close"].iloc[index] <= level:
                return ExitDecision(True, "lower_bollinger_close", "next_open")
        return ExitDecision.no_action()
