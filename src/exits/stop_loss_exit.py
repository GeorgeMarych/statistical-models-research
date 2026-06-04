"""Standalone stop-loss exit."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class StopLossExit(BaseExit):
    """Exit intrabar at a fixed percentage stop."""

    stop_loss_pct: float = 0.04
    label: str = "stop_loss"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"stop_loss_pct": self.stop_loss_pct}

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
            stop = position.entry_price * (1.0 - self.stop_loss_pct)
            if data["low"].iloc[index] <= stop:
                return ExitDecision(True, "stop_loss", "intrabar", float(stop))
        else:
            stop = position.entry_price * (1.0 + self.stop_loss_pct)
            if data["high"].iloc[index] >= stop:
                return ExitDecision(True, "stop_loss", "intrabar", float(stop))
        return ExitDecision.no_action()
