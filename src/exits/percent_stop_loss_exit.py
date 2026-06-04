"""Percent stop-loss exit with enable switch."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class PercentStopLossExit(BaseExit):
    """Exit intrabar at a fixed percent stop when enabled."""

    stop_loss_pct: float = 20.0
    enabled: bool = False
    label: str = "percent_stop_loss"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "enabled": self.enabled,
        }

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if not self.enabled or context.get("phase") != "intrabar":
            return ExitDecision.no_action()

        stop_fraction = self.stop_loss_pct / 100.0 if self.stop_loss_pct > 1 else self.stop_loss_pct
        if stop_fraction <= 0:
            return ExitDecision.no_action()

        if position.side == 1:
            stop = position.entry_price * (1.0 - stop_fraction)
            if data["low"].iloc[index] <= stop:
                return ExitDecision(True, "percent_stop_loss", "intrabar", float(stop))
        else:
            stop = position.entry_price * (1.0 + stop_fraction)
            if data["high"].iloc[index] >= stop:
                return ExitDecision(True, "percent_stop_loss", "intrabar", float(stop))
        return ExitDecision.no_action()
