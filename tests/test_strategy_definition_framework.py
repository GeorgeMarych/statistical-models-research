from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.exit_stack import ExitStack
from src.backtesting.portfolio import PositionSizing
from src.backtesting.strategy_runner import run_strategy_definition
from src.entries.close_breakout import CloseBreakoutEntry
from src.exits.fixed_bars_exit import FixedBarsExit
from src.exits.stop_loss_exit import StopLossExit
from src.filters.trend_filter import TrendFilter
from src.optimization.optimizer import run_parameter_optimization
from src.reports.optimization_report import write_optimization_outputs
from src.reports.single_strategy_report import write_single_strategy_outputs
from src.strategies.strategy_definition import StrategyDefinition


def synthetic_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=180, freq="B")
    close = np.concatenate(
        [
            np.linspace(90, 100, 50),
            np.linspace(100, 125, 60),
            np.linspace(125, 115, 70),
        ]
    )
    open_ = pd.Series(close, index=dates).shift(1).fillna(close[0]).to_numpy()
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )


class StrategyDefinitionFrameworkTests(unittest.TestCase):
    def test_filter_mask_aligned_with_data(self) -> None:
        data = synthetic_ohlcv()
        filter_module = TrendFilter(length=20, mode="long_above_sma")
        mask = filter_module.generate_mask(data)
        self.assertTrue(mask.index.equals(data.index))
        self.assertIn("allow_long", mask.columns)
        self.assertIn("allow_short", mask.columns)

    def test_exit_stack_records_exit_reason(self) -> None:
        data = synthetic_ohlcv()
        strategy = StrategyDefinition(
            name="test_breakout",
            symbols=["TEST"],
            direction_mode="long_only",
            entry=CloseBreakoutEntry(length=10, include_short_signals=False),
            filters=[],
            exit_stack=ExitStack(
                [StopLossExit(stop_loss_pct=0.50), FixedBarsExit(bars=3)],
                label="test_stack",
            ),
            sizing=PositionSizing(mode="percent_equity", value=0.95),
            min_bars=20,
        )
        result = run_strategy_definition(strategy, {"TEST": data})
        self.assertFalse(result.trades.empty)
        self.assertTrue(result.trades["exit_reason"].str.contains("fixed_bars").any())

    def test_strategy_definition_runs_end_to_end(self) -> None:
        data = synthetic_ohlcv()
        strategy = StrategyDefinition(
            name="test_strategy",
            symbols=["TEST"],
            direction_mode="long_only",
            entry=CloseBreakoutEntry(length=10, include_short_signals=False),
            filters=[TrendFilter(length=20, mode="long_above_sma")],
            exit_stack=ExitStack([FixedBarsExit(bars=5)]),
            sizing=PositionSizing(mode="percent_equity", value=0.95),
            min_bars=20,
        )
        result = run_strategy_definition(strategy, {"TEST": data})
        self.assertFalse(result.summary.empty)
        self.assertFalse(result.equity.empty)

    def test_optimizer_tiny_grid_and_reports(self) -> None:
        data = synthetic_ohlcv()
        strategy = StrategyDefinition(
            name="test_optimizer",
            symbols=["TEST"],
            direction_mode="long_only",
            entry=CloseBreakoutEntry(length=10, include_short_signals=False),
            filters=[],
            exit_stack=ExitStack([FixedBarsExit(bars=4, label="fixed_bars")]),
            sizing=PositionSizing(mode="percent_equity", value=0.95),
            min_bars=20,
        )
        optimization = run_parameter_optimization(
            strategy,
            {"TEST": data},
            {
                "entry.length": [8, 10],
                "exits.fixed_bars.bars": [3, 5],
            },
            top_n=2,
        )
        self.assertEqual(len(optimization.results), 4)
        self.assertFalse(optimization.stability.empty)
        with tempfile.TemporaryDirectory() as tmp:
            single_paths = write_single_strategy_outputs(
                strategy.name,
                run_strategy_definition(strategy, {"TEST": data}).summary,
                run_strategy_definition(strategy, {"TEST": data}).trades,
                run_strategy_definition(strategy, {"TEST": data}).equity,
                Path(tmp) / "single",
            )
            opt_paths = write_optimization_outputs(
                optimization.results,
                optimization.top_results,
                optimization.stability,
                Path(tmp) / "optimization",
            )
            self.assertTrue(single_paths["html_report"].exists())
            self.assertTrue(opt_paths["html_report"].exists())


if __name__ == "__main__":
    unittest.main()
