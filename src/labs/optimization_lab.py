"""Optimization lab runner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtesting.strategy_runner import collect_required_symbols
from src.labs.data_loading import load_price_data
from src.optimization.optimizer import run_parameter_optimization
from src.reports.optimization_report import write_optimization_outputs
from src.strategies.strategy_definition import StrategyDefinition, strategy_from_config


@dataclass
class OptimizationLabResult:
    """Outputs from an optimization lab run."""

    strategy: StrategyDefinition
    output_dir: Path
    paths: dict[str, Path]
    results: pd.DataFrame
    top_results: pd.DataFrame
    stability: pd.DataFrame


def load_optimization_config(path: str | Path) -> dict[str, Any]:
    """Load optimization lab config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_optimization_lab(config_path: str | Path) -> OptimizationLabResult:
    """Run a small parameter-grid optimization for one strategy."""
    config = load_optimization_config(config_path)
    strategy = strategy_from_config(config)
    symbols = collect_required_symbols(strategy)
    price_data = load_price_data(symbols, config)
    optimization_config = config.get("optimization", {}) or {}
    parameter_grid = optimization_config.get("parameter_grid", {}) or {}
    objective = str(optimization_config.get("objective", "balanced"))
    top_n = int(optimization_config.get("top_n", 20))

    optimization = run_parameter_optimization(
        strategy=strategy,
        price_data=price_data,
        parameter_grid=parameter_grid,
        objective=objective,
        top_n=top_n,
    )
    if optimization.results.empty:
        raise ValueError("optimization produced no result rows")

    output_dir = Path(
        config.get(
            "output_path",
            f"data/results/current/optimization_lab/{strategy.name}",
        )
    )
    paths = write_optimization_outputs(
        optimization.results,
        optimization.top_results,
        optimization.stability,
        output_dir,
    )
    return OptimizationLabResult(
        strategy=strategy,
        output_dir=output_dir,
        paths=paths,
        results=optimization.results,
        top_results=optimization.top_results,
        stability=optimization.stability,
    )
