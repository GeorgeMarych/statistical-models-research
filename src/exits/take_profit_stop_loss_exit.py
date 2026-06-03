"""Fixed percentage or ATR-based target/stop exit."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.indicators import compute_atr
from src.exits.base import BaseExit, ExitDecision, PositionContext


@dataclass
class TakeProfitStopLossExit(BaseExit):
    """Exit intrabar at a target or stop, with stop-first same-bar handling."""

    take_profit_pct: float | None = 0.08
    stop_loss_pct: float | None = 0.04
    atr_target_multiple: float | None = None
    atr_stop_multiple: float | None = None
    atr_length: int = 14
    atr_method: str = "wilder"
    label: str = "take_profit_stop_loss"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "atr_target_multiple": self.atr_target_multiple,
            "atr_stop_multiple": self.atr_stop_multiple,
            "atr_length": self.atr_length,
            "atr_method": self.atr_method,
        }

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        needs_atr = self.atr_target_multiple is not None or self.atr_stop_multiple is not None
        if not needs_atr:
            return pd.DataFrame(index=data.index)
        return pd.DataFrame(
            {"atr": compute_atr(data, self.atr_length, self.atr_method)},
            index=data.index,
        )

    def _levels(
        self,
        prepared: pd.DataFrame,
        position: PositionContext,
    ) -> tuple[float | None, float | None]:
        atr = None
        if "atr" in prepared.columns:
            atr_value = prepared["atr"].iloc[max(position.signal_index, 0)]
            if pd.notna(atr_value) and atr_value > 0:
                atr = float(atr_value)

        target = None
        stop = None
        if position.side == 1:
            if self.take_profit_pct is not None:
                target = position.entry_price * (1.0 + self.take_profit_pct)
            if self.stop_loss_pct is not None:
                stop = position.entry_price * (1.0 - self.stop_loss_pct)
            if self.atr_target_multiple is not None and atr is not None:
                target = position.entry_price + self.atr_target_multiple * atr
            if self.atr_stop_multiple is not None and atr is not None:
                stop = position.entry_price - self.atr_stop_multiple * atr
        else:
            if self.take_profit_pct is not None:
                target = position.entry_price * (1.0 - self.take_profit_pct)
            if self.stop_loss_pct is not None:
                stop = position.entry_price * (1.0 + self.stop_loss_pct)
            if self.atr_target_multiple is not None and atr is not None:
                target = position.entry_price - self.atr_target_multiple * atr
            if self.atr_stop_multiple is not None and atr is not None:
                stop = position.entry_price + self.atr_stop_multiple * atr
        return target, stop

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

        target, stop = self._levels(prepared, position)
        high = data["high"].iloc[index]
        low = data["low"].iloc[index]

        if position.side == 1:
            stop_hit = stop is not None and low <= stop
            target_hit = target is not None and high >= target
        else:
            stop_hit = stop is not None and high >= stop
            target_hit = target is not None and low <= target

        if stop_hit:
            return ExitDecision(True, "stop_loss", "intrabar", float(stop))
        if target_hit:
            return ExitDecision(True, "take_profit", "intrabar", float(target))
        return ExitDecision.no_action()
