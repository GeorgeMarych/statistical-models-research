"""Trailing-profit exit."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import compute_atr
from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class TrailingProfitExit(BaseExit):
    """Trail a stop from the best price since entry after a profit threshold."""

    trail_pct: float = 0.06
    activate_profit_pct: float = 0.03
    atr_length: int = 14
    atr_multiple: float | None = None
    label: str = "trailing_profit"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "trail_pct": self.trail_pct,
            "activate_profit_pct": self.activate_profit_pct,
            "atr_length": self.atr_length,
            "atr_multiple": self.atr_multiple,
        }

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.atr_multiple is None:
            return pd.DataFrame(index=data.index)
        return pd.DataFrame(
            {"atr": compute_atr(data, self.atr_length)},
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
            prepared = pd.DataFrame(index=data.index)

        if position.side == 1:
            best = data["high"].iloc[position.entry_index : index + 1].max()
            activated = best >= position.entry_price * (1.0 + self.activate_profit_pct)
            if not activated:
                return ExitDecision.no_action()
            stop = best * (1.0 - self.trail_pct)
            if self.atr_multiple is not None and "atr" in prepared.columns:
                atr = prepared["atr"].iloc[max(index - 1, position.signal_index)]
                if pd.notna(atr) and atr > 0:
                    stop = max(stop, best - self.atr_multiple * atr)
            if data["low"].iloc[index] <= stop:
                return ExitDecision(True, "trailing_profit", "intrabar", float(stop))
        else:
            best = data["low"].iloc[position.entry_index : index + 1].min()
            activated = best <= position.entry_price * (1.0 - self.activate_profit_pct)
            if not activated:
                return ExitDecision.no_action()
            stop = best * (1.0 + self.trail_pct)
            if self.atr_multiple is not None and "atr" in prepared.columns:
                atr = prepared["atr"].iloc[max(index - 1, position.signal_index)]
                if pd.notna(atr) and atr > 0:
                    stop = min(stop, best + self.atr_multiple * atr)
            if data["high"].iloc[index] >= stop:
                return ExitDecision(True, "trailing_profit", "intrabar", float(stop))
        return ExitDecision.no_action()
