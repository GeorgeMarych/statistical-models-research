"""Example mean-reversion ETF strategy definition."""
from src.backtesting.exit_stack import ExitStack
from src.backtesting.costs import TradingCosts
from src.backtesting.portfolio import PositionSizing
from src.entries.mean_reversion import MeanReversionEntry
from src.exits.fixed_bars_exit import FixedBarsExit
from src.exits.profit_target_exit import ProfitTargetExit
from src.exits.stop_loss_exit import StopLossExit
from src.filters.market_regime_filter import MarketRegimeFilter
from src.strategies.strategy_definition import StrategyDefinition


def build_strategy(symbols: list[str] | None = None) -> StrategyDefinition:
    """Build a simple ETF mean-reversion strategy."""
    return StrategyDefinition(
        name="mean_reversion_etf_strategy",
        symbols=symbols or ["SPY", "QQQ", "IWM", "XLK", "XLF"],
        direction_mode="long_only",
        entry=MeanReversionEntry(length=10, long_rsi_max=35.0),
        filters=[MarketRegimeFilter(benchmark_symbol="SPY", length=200)],
        exit_stack=ExitStack(
            [
                StopLossExit(stop_loss_pct=0.05),
                ProfitTargetExit(take_profit_pct=0.05),
                FixedBarsExit(bars=8),
            ],
            label="target_stop_time",
        ),
        sizing=PositionSizing(mode="percent_equity", value=0.95),
        costs=TradingCosts(
            commission_per_share=0.005,
            min_commission_per_order=1.00,
            slippage_bps_per_side=3.0,
        ),
    )
