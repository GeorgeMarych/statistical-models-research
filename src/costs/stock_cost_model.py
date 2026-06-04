"""Stock/ETF cost-model adapter."""
from __future__ import annotations

from src.backtesting.costs import TradingCosts


class StockCostModel(TradingCosts):
    """Commission and slippage model using percent/bps inputs."""

    def __init__(
        self,
        commission_percent: float = 0.03,
        slippage_bps_per_side: float = 0.0,
        spread_bps: float = 0.0,
        min_commission_per_order: float = 0.0,
    ) -> None:
        self.commission_percent = float(commission_percent)
        super().__init__(
            commission_pct=self.commission_percent / 100.0,
            slippage_bps_per_side=float(slippage_bps_per_side),
            spread_bps=float(spread_bps),
            min_commission_per_order=float(min_commission_per_order),
        )
