"""Simple grid optimizer adapter for StrategyDefinition objects."""
from __future__ import annotations

import pandas as pd

from src.optimization.optimizer import OptimizationResult, run_parameter_optimization
from src.strategies.strategy_definition import StrategyDefinition


def run_simple_grid_optimizer(
    strategy: StrategyDefinition,
    price_data: dict,
    parameter_grid: dict,
    objective: str = "balanced",
    top_n: int = 20,
) -> OptimizationResult:
    """Run the existing parameter optimizer under a simple lab-specific name."""
    result = run_parameter_optimization(
        strategy=strategy,
        price_data=price_data,
        parameter_grid=parameter_grid,
        objective=objective,
        top_n=top_n,
    )
    result.results = add_optimization_flags(result.results)
    result.top_results = result.results.head(top_n).copy()
    return result


def add_optimization_flags(results: pd.DataFrame) -> pd.DataFrame:
    """Flag weak or potentially overfit configurations."""
    if results.empty:
        return results
    out = results.copy()
    reasons: list[str] = []
    for _, row in out.iterrows():
        row_reasons: list[str] = []
        trades = int(row.get("number_of_trades", 0) or 0)
        max_drawdown = float(row.get("max_drawdown", 0.0) or 0.0)
        profit_factor = float(row.get("profit_factor", 0.0) or 0.0)
        expectancy = float(row.get("expectancy_pct", 0.0) or 0.0)
        if trades < 20:
            row_reasons.append("fewer_than_20_trades")
        if max_drawdown < -0.70:
            row_reasons.append("max_drawdown_gt_70pct")
        if profit_factor < 1.2:
            row_reasons.append("profit_factor_lt_1_2")
        if expectancy < 0.001:
            row_reasons.append("average_trade_too_small_after_costs")
        reasons.append(";".join(row_reasons))
    out["rejection_reason"] = reasons
    out["passes_research_filters"] = out["rejection_reason"].eq("")
    return out
