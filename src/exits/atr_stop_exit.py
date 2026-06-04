"""ATR multiple emergency stop exit."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import compute_atr
from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class ATRStopExit(BaseExit):
    """Exit intrabar when price breaches an ATR multiple stop."""

    atr_length: int = 14
    atr_multiple: float = 2.5
    atr_mult: float | None = None
    atr_method: str = "wilder"
    trailing: bool = False
    label: str = "atr_stop"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "atr_length": self.atr_length,
            "atr_multiple": self._active_multiple(),
            "atr_method": self.atr_method,
            "trailing": self.trailing,
        }

    def _active_multiple(self) -> float:
        return self.atr_mult if self.atr_mult is not None else self.atr_multiple

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"atr": compute_atr(data, self.atr_length, self.atr_method)},
            index=data.index,
        )

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if context.get("phase") != "intrabar":
            return ExitDecision.no_action()

        prepared = context["prepared_exits"].get(self.name)
        if prepared is None:
            return ExitDecision.no_action()

        atr_index = max(position.signal_index, 0)
        atr = prepared["atr"].iloc[atr_index]
        if pd.isna(atr) or atr <= 0:
            return ExitDecision.no_action()

        if position.side == 1:
            multiple = self._active_multiple()
            stop_price = position.entry_price - multiple * atr
            if self.trailing:
                high_since_entry = data["high"].iloc[position.entry_index : index + 1].max()
                prev_atr = prepared["atr"].iloc[max(index - 1, atr_index)]
                if pd.notna(prev_atr) and prev_atr > 0:
                    stop_price = max(stop_price, high_since_entry - multiple * prev_atr)
            if data["low"].iloc[index] <= stop_price:
                return ExitDecision(True, "atr_stop", "intrabar", float(stop_price))
        else:
            multiple = self._active_multiple()
            stop_price = position.entry_price + multiple * atr
            if self.trailing:
                low_since_entry = data["low"].iloc[position.entry_index : index + 1].min()
                prev_atr = prepared["atr"].iloc[max(index - 1, atr_index)]
                if pd.notna(prev_atr) and prev_atr > 0:
                    stop_price = min(stop_price, low_since_entry + multiple * prev_atr)
            if data["high"].iloc[index] >= stop_price:
                return ExitDecision(True, "atr_stop", "intrabar", float(stop_price))

        return ExitDecision.no_action()
