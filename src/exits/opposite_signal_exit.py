"""Exit or reverse when the entry module emits an opposite signal."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entries.base import SIGNAL_COLUMN
from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class OppositeSignalExit(BaseExit):
    """React to an opposite entry signal on the signal bar close."""

    reverse_on_opposite: bool = True
    label: str = "opposite_signal"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"reverse_on_opposite": self.reverse_on_opposite}

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        if context.get("phase") != "close":
            return ExitDecision.no_action()

        entry_signals = context.get("entry_signals")
        if entry_signals is None or SIGNAL_COLUMN not in entry_signals.columns:
            return ExitDecision.no_action()

        signal = int(entry_signals[SIGNAL_COLUMN].iloc[index])
        if signal == 0 or signal == position.side:
            return ExitDecision.no_action()

        reverse_side = None
        if self.reverse_on_opposite:
            reverse_side = signal
            if reverse_side == -1 and not context.get("allow_short", False):
                reverse_side = None

        return ExitDecision(
            should_exit=True,
            reason="opposite_signal",
            timing="next_open",
            reverse_to_side=reverse_side,
        )
