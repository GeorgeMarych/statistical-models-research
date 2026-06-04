"""Single strategy lab runner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtesting.strategy_runner import collect_required_symbols, run_strategy_definition
from src.labs.data_loading import load_price_data
from src.reports.single_strategy_report import write_single_strategy_outputs
from src.strategies.strategy_definition import StrategyDefinition, strategy_from_config


@dataclass
class SingleStrategyLabResult:
    """Outputs from a single strategy lab run."""

    strategy: StrategyDefinition
    output_dir: Path
    paths: dict[str, Path]
    summary: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame


def load_single_strategy_config(path: str | Path) -> dict[str, Any]:
    """Load the single strategy YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_single_strategy_lab(config_path: str | Path) -> SingleStrategyLabResult:
    """Run one configured strategy and write standard outputs."""
    config = load_single_strategy_config(config_path)
    strategy = strategy_from_config(config)
    symbols = collect_required_symbols(strategy)
    price_data = load_price_data(symbols, config)
    result = run_strategy_definition(strategy, price_data)
    if result.summary.empty:
        raise ValueError("single strategy lab produced no summary rows")

    output_dir = Path(
        config.get(
            "output_path",
            f"data/results/current/single_strategy_lab/{strategy.name}",
        )
    )
    paths = write_single_strategy_outputs(
        strategy.name,
        result.summary,
        result.trades,
        result.equity,
        output_dir,
    )
    return SingleStrategyLabResult(
        strategy=strategy,
        output_dir=output_dir,
        paths=paths,
        summary=result.summary,
        trades=result.trades,
        equity=result.equity,
    )
