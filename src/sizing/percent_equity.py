"""Percent-of-equity sizing adapter."""
from __future__ import annotations

from src.backtesting.portfolio import PositionSizing


class PercentEquitySizing(PositionSizing):
    """Position sizing expressed as a user-facing percent of equity."""

    def __init__(self, percent: float = 100.0, fractional: bool = True) -> None:
        self.percent = float(percent)
        super().__init__(
            mode="percent_equity",
            value=self.percent / 100.0,
            fractional=fractional,
        )
