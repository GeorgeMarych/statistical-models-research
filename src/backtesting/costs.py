"""Trading cost and slippage helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingCosts:
    """Simple per-order cost model for research backtests."""

    commission_per_trade: float = 0.0
    commission_pct: float = 0.0
    commission_per_share: float = 0.0
    min_commission_per_order: float = 0.0
    slippage_bps: float = 0.0
    slippage_bps_per_side: float = 0.0
    spread_bps: float = 0.0

    def price_with_costs(self, price: float, order_side: int) -> float:
        """
        Adjust a fill price for slippage/spread.

        order_side is +1 for buy orders and -1 for sell orders.
        """
        if price <= 0:
            raise ValueError("price must be positive")
        if order_side not in {-1, 1}:
            raise ValueError("order_side must be -1 or 1")
        total_bps = self.slippage_bps + self.slippage_bps_per_side + self.spread_bps / 2.0
        adjustment = total_bps / 10000.0
        if order_side == 1:
            return float(price * (1.0 + adjustment))
        return float(price * (1.0 - adjustment))

    def commission(self, notional: float, quantity: float = 0.0) -> float:
        """Return total commission for an order notional."""
        commission = self.commission_per_trade + abs(notional) * self.commission_pct
        if self.commission_per_share:
            commission += abs(quantity) * self.commission_per_share
        if self.min_commission_per_order and commission > 0:
            commission = max(commission, self.min_commission_per_order)
        return float(commission)
