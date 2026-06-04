"""Standalone profit-target exit."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class ProfitTargetExit(BaseExit):
    """Exit intrabar at a fixed percentage profit target."""

    take_profit_pct: float = 0.08
    label: str = "profit_target"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"take_profit_pct": self.take_profit_pct}

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if context.get("phase") != "intrabar":
            return ExitDecision.no_action()
        if position.side == 1:
            target = position.entry_price * (1.0 + self.take_profit_pct)
            if data["high"].iloc[index] >= target:
                return ExitDecision(True, "profit_target", "intrabar", float(target))
        else:
            target = position.entry_price * (1.0 - self.take_profit_pct)
            if data["low"].iloc[index] <= target:
                return ExitDecision(True, "profit_target", "intrabar", float(target))
        return ExitDecision.no_action()
