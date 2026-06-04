"""Simple strategy parameter optimizer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.backtesting.strategy_runner import aggregate_strategy_summary, run_strategy_definition
from src.optimization.objective import score_summary
from src.optimization.parameter_grid import iter_parameter_grid
from src.optimization.stability import build_parameter_stability
from src.strategies.strategy_definition import StrategyDefinition


@dataclass
class OptimizationResult:
    """Optimization outputs."""

    results: pd.DataFrame
    top_results: pd.DataFrame
    stability: pd.DataFrame


def run_parameter_optimization(
    strategy: StrategyDefinition,
    price_data: dict[str, pd.DataFrame],
    parameter_grid: dict[str, list[Any]],
    objective: str = "balanced",
    top_n: int = 20,
) -> OptimizationResult:
    """Run a small Cartesian parameter grid for one strategy."""
    rows: list[dict] = []
    parameter_columns = list(parameter_grid.keys())

    for run_number, parameters in enumerate(iter_parameter_grid(parameter_grid), start=1):
        candidate = strategy.clone_with_parameters(parameters)
        run = run_strategy_definition(candidate, price_data)
        aggregate = aggregate_strategy_summary(run.summary)
        score = score_summary(aggregate, objective)
        row = {
            "optimization_run": run_number,
            "strategy": strategy.name,
            "objective": objective,
            "score": score,
            "parameters": json.dumps(parameters, sort_keys=True),
        }
        row.update(parameters)
        row.update(aggregate)
        rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values("score", ascending=False, na_position="last")
    top_results = results.head(top_n).copy()
    stability = build_parameter_stability(results, parameter_columns)
    return OptimizationResult(results, top_results, stability)
