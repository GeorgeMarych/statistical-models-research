"""Exit after a fixed number of bars in a trade."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class FixedBarsExit(BaseExit):
    """Close a position after max_bars bars have elapsed."""

    max_bars: int = 10
    label: str = "fixed_bars"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"max_bars": self.max_bars}

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if self.max_bars <= 0:
            raise ValueError("max_bars must be positive")
        if context.get("phase") != "close":
            return ExitDecision.no_action()
        if position.bars_held >= self.max_bars:
            return ExitDecision(
                should_exit=True,
                reason=f"fixed_bars_{self.max_bars}",
                timing="next_open",
            )
        return ExitDecision.no_action()
