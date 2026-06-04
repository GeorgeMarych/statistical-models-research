"""Monte Carlo trade-skip robustness tests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtesting.strategy_runner import aggregate_strategy_summary, run_strategy_definition
from src.entries.base import SIGNAL_COLUMN, EntryProtocol
from src.strategies.strategy_definition import StrategyDefinition


@dataclass
class MonteCarloSkipEntry:
    """Entry wrapper that randomly removes otherwise valid entry signals."""

    entry: EntryProtocol
    skip_pct: float
    seed: int
    label: str = "monte_carlo_skip_entry"

    @property
    def name(self) -> str:
        return f"{self.entry.name}_{self.label}"

    @property
    def parameters(self) -> dict:
        return {
            "entry": self.entry.parameters,
            "skip_pct": self.skip_pct,
            "seed": self.seed,
        }

    @property
    def min_lookback(self) -> int:
        return getattr(self.entry, "min_lookback", 0)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = self.entry.generate_signals(data).copy()
        rng = np.random.default_rng(self.seed)
        active = signals[SIGNAL_COLUMN] != 0
        random_values = pd.Series(rng.random(len(signals)), index=signals.index)
        skipped = active & (random_values < self.skip_pct / 100.0)
        signals.loc[skipped, SIGNAL_COLUMN] = 0
        signals["mc_skipped_signal"] = skipped
        return signals


def run_monte_carlo_skip(
    strategy: StrategyDefinition,
    price_data: dict[str, pd.DataFrame],
    skip_pct_values: list[float],
    num_runs: int = 100,
    random_seed: int = 0,
    buy_hold_return: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated backtests with random entry-signal removal."""
    rows: list[dict] = []
    for skip_pct in skip_pct_values:
        for run_number in range(1, num_runs + 1):
            candidate = deepcopy(strategy)
            seed = int(random_seed + skip_pct * 1000 + run_number)
            candidate.entry = MonteCarloSkipEntry(
                entry=candidate.entry,
                skip_pct=skip_pct,
                seed=seed,
            )
            run = run_strategy_definition(candidate, price_data)
            aggregate = aggregate_strategy_summary(run.summary)
            row = {
                "run_number": run_number,
                "skip_pct": skip_pct,
                "seed": seed,
            }
            row.update(aggregate)
            rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame()

    rows_summary: list[dict] = []
    for skip_pct, group in results.groupby("skip_pct", sort=True):
        total_return = group["total_return"].astype(float)
        max_drawdown = group["max_drawdown"].astype(float)
        profit_factor = group["profit_factor"].astype(float)
        row = {
            "skip_pct": skip_pct,
            "runs": int(len(group)),
            "median_total_return": float(total_return.median()),
            "p05_total_return": float(total_return.quantile(0.05)),
            "worst_total_return": float(total_return.min()),
            "median_max_drawdown": float(max_drawdown.median()),
            "worst_max_drawdown": float(max_drawdown.min()),
            "median_profit_factor": float(profit_factor.median()),
            "pct_profitable": float((total_return > 0).mean()),
            "median_number_of_trades": float(group["number_of_trades"].median()),
            "median_final_equity": float(group["final_equity"].median()),
        }
        if buy_hold_return is not None:
            row["pct_beating_buy_hold"] = float((total_return > buy_hold_return).mean())
        else:
            row["pct_beating_buy_hold"] = float("nan")
        rows_summary.append(row)
    summary = pd.DataFrame(rows_summary)
    return results, summary
