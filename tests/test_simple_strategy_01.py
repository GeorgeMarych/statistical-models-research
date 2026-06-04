from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.strategy_runner import run_strategy_definition
from src.entries.volume_fade_entry import VolumeFadeLongEntry
from src.filters.base import ALLOW_LONG
from src.filters.volume_filter import VolumeAboveAverageFilter
from src.filters.weekday_filter import WeekdayFilter
from src.filters.wide_bar_filter import WideBullishBarFilter
from src.optimization.simple_grid_optimizer import run_simple_grid_optimizer
from src.reports.simple_strategy_01_report import write_simple_strategy_01_outputs
from src.robustness.monte_carlo_skip import run_monte_carlo_skip
from src.robustness.trade_sequence_randomization import run_trade_sequence_randomization
from src.strategies.examples.simple_strategy_01 import (
    build_software_volume_fade_reversal_strategy,
)


def synthetic_volume_fade_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    close = np.array(
        [
            100,
            99,
            98,
            97,
            96,
            95,
            96,
            97,
            99,
            101,
            103,
            104,
            102,
            101,
            100,
            99,
        ]
        * 5,
        dtype=float,
    )[:80]
    open_ = pd.Series(close, index=dates).shift(1).fillna(close[0]).to_numpy()
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.where(
        pd.Series(close).rolling(2).min().to_numpy() == close,
        2_000_000,
        1_000_000,
    )
    volume = np.where(
        pd.Series(close).rolling(2).max().to_numpy() == close,
        2_200_000,
        volume,
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class SimpleStrategy01Tests(unittest.TestCase):
    def test_strategy_definition_is_composition_based(self) -> None:
        strategy = build_software_volume_fade_reversal_strategy(["TEST"])
        self.assertIsInstance(strategy.entry, VolumeFadeLongEntry)
        self.assertTrue(hasattr(strategy.entry, "extreme_signal"))
        self.assertTrue(hasattr(strategy.entry, "volume_filter"))
        self.assertTrue(any(filter_module.name == "weekday_skip_wednesday" for filter_module in strategy.filters))
        self.assertTrue(strategy.exit_stack.find("opposite_strength"))

    def test_weekday_filter_skips_configured_days(self) -> None:
        data = synthetic_volume_fade_data()
        mask = WeekdayFilter(skip_days=["Wednesday"]).generate_mask(data)
        wednesdays = data.index.weekday == 2
        self.assertFalse(mask.loc[wednesdays, ALLOW_LONG].any())
        self.assertTrue(mask.loc[~wednesdays, ALLOW_LONG].all())

    def test_volume_filter_works(self) -> None:
        data = synthetic_volume_fade_data()
        mask = VolumeAboveAverageFilter(length=2, multiplier=1.0).generate_mask(data)
        self.assertTrue(mask[ALLOW_LONG].any())
        self.assertFalse(mask[ALLOW_LONG].iloc[0])

    def test_wide_bullish_bar_filter_blocks_wide_bullish_bars(self) -> None:
        data = synthetic_volume_fade_data()
        data.iloc[10, data.columns.get_loc("high")] = data["low"].iloc[10] + 20
        data.iloc[10, data.columns.get_loc("close")] = data["open"].iloc[10] + 10
        mask = WideBullishBarFilter(range_length=3, range_mult=1.0).generate_mask(data)
        self.assertFalse(bool(mask[ALLOW_LONG].iloc[10]))

    def test_long_only_exits_on_opposite_strength_signal(self) -> None:
        data = synthetic_volume_fade_data()
        strategy = build_software_volume_fade_reversal_strategy(
            ["TEST"],
            long_length=2,
            short_length=2,
            volume_avg_length=2,
            avoid_wednesday_longs=False,
            wide_bullish_filter=False,
            fixed_bars_exit=True,
            exit_bars=40,
            direction_mode="long_only",
            min_bars=5,
        )
        result = run_strategy_definition(strategy, {"TEST": data})
        self.assertFalse(result.trades.empty)
        self.assertTrue(result.trades["exit_reason"].str.contains("opposite_strength").any())
        self.assertTrue((result.trades["side"] == "long").all())

    def test_fixed_bars_exit_works(self) -> None:
        data = synthetic_volume_fade_data()
        strategy = build_software_volume_fade_reversal_strategy(
            ["TEST"],
            long_length=2,
            short_length=50,
            volume_avg_length=2,
            avoid_wednesday_longs=False,
            wide_bullish_filter=False,
            fixed_bars_exit=True,
            exit_bars=3,
            min_bars=5,
        )
        result = run_strategy_definition(strategy, {"TEST": data})
        self.assertTrue(result.trades["exit_reason"].str.contains("fixed_bars").any())

    def test_monte_carlo_skip_changes_trade_count(self) -> None:
        data = synthetic_volume_fade_data()
        strategy = build_software_volume_fade_reversal_strategy(
            ["TEST"],
            long_length=2,
            short_length=2,
            volume_avg_length=2,
            avoid_wednesday_longs=False,
            wide_bullish_filter=False,
            min_bars=5,
        )
        zero_results, _ = run_monte_carlo_skip(strategy, {"TEST": data}, [0], num_runs=1, random_seed=1)
        full_results, _ = run_monte_carlo_skip(strategy, {"TEST": data}, [100], num_runs=1, random_seed=1)
        self.assertGreater(int(zero_results["number_of_trades"].iloc[0]), int(full_results["number_of_trades"].iloc[0]))

    def test_trade_sequence_randomization_produces_multiple_paths(self) -> None:
        trades = pd.DataFrame({"net_return": [0.05, -0.03, 0.02, -0.01, 0.04]})
        randomized = run_trade_sequence_randomization(trades, num_runs=10, random_seed=1)
        self.assertEqual(len(randomized), 10)

    def test_optimizer_runs_tiny_grid(self) -> None:
        data = synthetic_volume_fade_data()
        strategy = build_software_volume_fade_reversal_strategy(["TEST"], min_bars=5)
        result = run_simple_grid_optimizer(
            strategy,
            {"TEST": data},
            {
                "entry.extreme_signal.length": [2, 3],
                "exits.fixed_bars.bars": [5, 8],
            },
            top_n=2,
        )
        self.assertEqual(len(result.results), 4)
        self.assertFalse(result.top_results.empty)

    def test_report_files_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_simple_strategy_01_outputs(
                output_dir=Path(tmp),
                summary_by_symbol=pd.DataFrame([{"symbol": "CRWD", "total_return": 0.1}]),
                trade_log=pd.DataFrame(),
                equity_curves=pd.DataFrame(),
                strategy_vs_buy_hold=pd.DataFrame(),
                monte_carlo_skip_results=pd.DataFrame(),
                monte_carlo_skip_summary=pd.DataFrame(),
                trade_sequence_randomization=pd.DataFrame(),
                permutation_results=pd.DataFrame(),
                permutation_summary=pd.DataFrame(),
                optimization_results=pd.DataFrame(),
                top_20_results=pd.DataFrame(),
                parameter_stability=pd.DataFrame(),
            )
            self.assertTrue(paths["html_report"].exists())
            self.assertTrue(paths["summary_by_symbol"].exists())


if __name__ == "__main__":
    unittest.main()
