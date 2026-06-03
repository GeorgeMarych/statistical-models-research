"""Portfolio primitives for the research backtest engine."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PositionSizing:
    """Position sizing settings for one-position-at-a-time research runs."""

    mode: str = "percent_equity"
    value: float = 1.0
    fractional: bool = True

    def quantity(self, price: float, equity: float) -> float:
        """Resolve order quantity from the configured sizing mode."""
        if price <= 0:
            return 0.0
        mode = str(self.mode).lower()
        if mode == "fixed_quantity":
            qty = self.value
        elif mode == "fixed_notional":
            qty = self.value / price
        elif mode == "percent_equity":
            qty = equity * self.value / price
        else:
            raise ValueError(f"Unsupported sizing mode: {self.mode}")

        qty = max(float(qty), 0.0)
        if not self.fractional:
            qty = float(int(qty))
        return qty


@dataclass
class Position:
    """Open position state tracked by the engine."""

    symbol: str
    side: int
    quantity: float
    entry_price: float
    entry_index: int
    signal_index: int
    entry_date: pd.Timestamp
    signal_date: pd.Timestamp
    entry_commission: float


def mark_to_market(cash: float, position: Position | None, close_price: float) -> float:
    """Return equity from cash plus the marked value of the open position."""
    if position is None:
        return float(cash)
    return float(cash + position.side * position.quantity * close_price)
