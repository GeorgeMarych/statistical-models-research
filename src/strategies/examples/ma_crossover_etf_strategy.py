"""Example moving-average crossover ETF strategy definition."""
from src.backtesting.exit_stack import ExitStack
from src.backtesting.costs import TradingCosts
from src.backtesting.portfolio import PositionSizing
from src.entries.ma_crossover_entry import MovingAverageCrossoverEntry
from src.exits.opposite_signal_exit import OppositeSignalExit
from src.exits.trailing_profit_exit import TrailingProfitExit
from src.strategies.strategy_definition import StrategyDefinition


def build_strategy(symbols: list[str] | None = None) -> StrategyDefinition:
    """Build a simple moving-average crossover ETF strategy."""
    return StrategyDefinition(
        name="ma_crossover_etf_strategy",
        symbols=symbols or ["SPY", "QQQ", "IWM", "XLK", "XLF"],
        direction_mode="long_short",
        entry=MovingAverageCrossoverEntry(
            fast_window=20,
            slow_window=50,
            include_short_signals=True,
        ),
        filters=[],
        exit_stack=ExitStack(
            [
                TrailingProfitExit(trail_pct=0.06, activate_profit_pct=0.03),
                OppositeSignalExit(reverse_on_opposite=True),
            ],
            label="trailing_plus_opposite",
        ),
        sizing=PositionSizing(mode="percent_equity", value=0.95),
        costs=TradingCosts(
            commission_per_share=0.005,
            min_commission_per_order=1.00,
            slippage_bps_per_side=3.0,
        ),
    )
