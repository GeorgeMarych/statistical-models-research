"""Example breakout ETF strategy definition."""
from src.backtesting.exit_stack import ExitStack
from src.backtesting.costs import TradingCosts
from src.backtesting.portfolio import PositionSizing
from src.entries.close_breakout import CloseBreakoutEntry
from src.exits.atr_stop_exit import ATRStopExit
from src.exits.fixed_bars_exit import FixedBarsExit
from src.filters.trend_filter import TrendFilter
from src.strategies.strategy_definition import StrategyDefinition


def build_strategy(symbols: list[str] | None = None) -> StrategyDefinition:
    """Build a simple close-breakout ETF strategy."""
    return StrategyDefinition(
        name="breakout_etf_strategy",
        symbols=symbols or ["SPY", "QQQ", "IWM", "XLK", "XLF"],
        direction_mode="long_only",
        entry=CloseBreakoutEntry(length=20),
        filters=[TrendFilter(length=200, mode="long_above_sma")],
        exit_stack=ExitStack(
            [
                ATRStopExit(atr_length=14, atr_multiple=3.0),
                FixedBarsExit(bars=10),
            ],
            label="atr_stop_plus_fixed_bars",
        ),
        sizing=PositionSizing(mode="percent_equity", value=0.95),
        costs=TradingCosts(
            commission_per_share=0.005,
            min_commission_per_order=1.00,
            slippage_bps_per_side=3.0,
        ),
    )
