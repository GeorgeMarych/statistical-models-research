"""Research lab for Simple Strategy #1."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml

from src.backtesting.exit_stack import ExitStack
from src.backtesting.strategy_runner import collect_required_symbols, run_strategy_definition
from src.costs.stock_cost_model import StockCostModel
from src.entries.price_extreme_signals import HighestCloseSignal, LowestCloseSignal
from src.entries.volume_fade_entry import VolumeFadeReversalEntry
from src.exits.fixed_bars_exit import FixedBarsExit
from src.exits.opposite_strength_exit import OppositeStrengthExit, VolumeStrengthSignal
from src.exits.percent_stop_loss_exit import PercentStopLossExit
from src.filters.market_regime_filter import MarketRegimeFilter
from src.filters.trend_filter import TrendFilter
from src.filters.volume_filter import VolumeAboveAverageFilter
from src.filters.weekday_filter import WeekdayFilter
from src.filters.wide_bar_filter import WideBullishBarFilter
from src.labs.data_loading import load_price_data, normalize_symbols
from src.labs.simple_strategy_01_outputs import (
    copy_final_outputs_to_latest,
    prepare_simple_strategy_output,
)
from src.optimization.objective import score_summary
from src.optimization.parameter_grid import iter_parameter_grid
from src.optimization.simple_grid_optimizer import add_optimization_flags
from src.reports.simple_strategy_01_report import write_simple_strategy_01_outputs
from src.robustness.monte_carlo_skip import run_monte_carlo_skip
from src.robustness.permutation_test import run_permutation_test
from src.sizing.percent_equity import PercentEquitySizing
from src.robustness.trade_sequence_randomization import (
    run_trade_sequence_randomization,
    summarize_trade_sequence_randomization,
)
from src.strategies.examples.simple_strategy_01 import (
    PEER_UNIVERSE,
    PRIMARY_SYMBOL,
    build_software_volume_fade_reversal_strategy,
)
from src.strategies.strategy_definition import StrategyDefinition


@dataclass
class SimpleStrategy01LabResult:
    """Outputs from the simple strategy lab."""

    strategy: StrategyDefinition
    output_dir: Path
    paths: dict[str, Path]
    missing_symbols: list[str]
    stage: str
    summary_by_symbol: pd.DataFrame
    strategy_vs_buy_hold: pd.DataFrame
    trade_log: pd.DataFrame


def load_simple_strategy_01_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config for the lab."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_output_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return safe output-organization defaults."""
    config = {
        "organize_runs": True,
        "clean_before_run": False,
        "archive_old_root_outputs": True,
        "copy_final_outputs_to_latest": True,
    }
    config.update(raw or {})
    return config


def run_simple_strategy_01_lab(
    config_path: str | Path,
    stage_override: str | None = None,
) -> SimpleStrategy01LabResult:
    """Run baseline, robustness, optimization, and report generation."""
    config = load_simple_strategy_01_config(config_path)
    if stage_override:
        config["stage"] = stage_override
    config = apply_stage_defaults(config)
    run_flags = config.get("run", {}) or {}
    stage = str(config.get("stage", "custom"))
    base_output_dir = Path(config.get("output_path", "data/results/current/simple_strategy_01"))
    output_config = default_output_config(config.get("output", {}) or {})
    output_layout = prepare_simple_strategy_output(
        base_output_dir,
        stage=stage,
        output_config=output_config,
    )
    output_dir = output_layout.run_dir

    primary_symbol = str(config.get("primary_symbol", PRIMARY_SYMBOL)).upper()
    peer_symbols = normalize_symbols(config.get("peer_universe", PEER_UNIVERSE))
    if primary_symbol not in peer_symbols:
        peer_symbols = [primary_symbol] + peer_symbols

    baseline_symbols = (
        peer_symbols
        if run_flags.get("run_peer_universe", True)
        else [primary_symbol]
    )
    strategy = _build_strategy(config, baseline_symbols)
    symbols_to_load = collect_required_symbols(strategy)
    optimization_config = config.get("optimization", {}) or {}
    optimization_symbols = normalize_symbols(
        optimization_config.get(
            "symbols_to_optimize",
            optimization_config.get("symbols", [primary_symbol]),
        )
    )
    permutation_symbols = normalize_symbols(
        (config.get("permutation_test", {}) or {}).get("symbols", optimization_symbols)
    )
    direction_symbols = normalize_symbols(
        (config.get("direction_mode_comparison", {}) or {}).get(
            "symbols",
            optimization_symbols,
        )
    )
    symbols_to_load = normalize_symbols(
        symbols_to_load + optimization_symbols + permutation_symbols + direction_symbols
    )
    price_data = load_price_data(symbols_to_load, config)
    missing_symbols = [symbol for symbol in baseline_symbols if symbol not in price_data]

    baseline_run = run_strategy_definition(strategy, price_data)
    summary_by_symbol = baseline_run.summary.copy()
    trade_log = baseline_run.trades.copy()
    equity_curves = baseline_run.equity.copy()
    summary_by_symbol = add_trade_contribution_columns(summary_by_symbol, trade_log)
    strategy_vs_buy_hold = build_strategy_vs_buy_hold(summary_by_symbol, price_data)
    cost_stress_summary = (
        run_cost_stress_cases(config, baseline_symbols, price_data)
        if run_flags.get("run_cost_stress", False)
        else pd.DataFrame()
    )

    primary_strategy = _build_strategy(config, [primary_symbol])
    primary_price_data = {
        symbol: frame
        for symbol, frame in price_data.items()
        if symbol in collect_required_symbols(primary_strategy)
    }
    primary_run = run_strategy_definition(primary_strategy, primary_price_data)

    mc_results = pd.DataFrame()
    mc_summary = pd.DataFrame()
    if run_flags.get("run_monte_carlo_skip", True) and primary_price_data:
        mc_config = config.get("monte_carlo_skip", {}) or {}
        primary_buy_hold = _buy_hold_return(price_data.get(primary_symbol))
        mc_results, mc_summary = run_monte_carlo_skip(
            strategy=primary_strategy,
            price_data=primary_price_data,
            skip_pct_values=mc_config.get("skip_pct", [0, 5, 10, 15, 20]),
            num_runs=int(mc_config.get("num_runs", 100)),
            random_seed=int(mc_config.get("random_seed", 0)),
            buy_hold_return=primary_buy_hold,
        )

    sequence_results = pd.DataFrame()
    sequence_summary = pd.DataFrame()
    if run_flags.get("run_trade_sequence_randomization", True) and not primary_run.trades.empty:
        seq_config = config.get("trade_sequence_randomization", {}) or {}
        sequence_results = run_trade_sequence_randomization(
            primary_run.trades,
            num_runs=int(seq_config.get("num_runs", 1000)),
            random_seed=int(seq_config.get("random_seed", 0)),
            initial_capital=primary_strategy.initial_capital,
        )
        sequence_summary = build_trade_sequence_summary(sequence_results)

    permutation_results = pd.DataFrame()
    permutation_summary = pd.DataFrame()
    quasi_p_values = pd.DataFrame()
    if run_flags.get("run_permutation_test", False):
        permutation_results, permutation_summary, quasi_p_values = run_stage7_permutation_tests(
            config,
            permutation_symbols,
            price_data,
            output_dir,
        )

    optimization_results = pd.DataFrame()
    top_20_results = pd.DataFrame()
    parameter_stability = pd.DataFrame()
    baseline_vs_optimized_summary = pd.DataFrame()
    if run_flags.get("run_optimization", True):
        (
            optimization_results,
            top_20_results,
            parameter_stability,
            baseline_vs_optimized_summary,
        ) = run_stage6_parameter_stability(
            config,
            optimization_symbols,
            price_data,
            output_dir,
        )

    direction_mode_comparison = pd.DataFrame()
    if run_flags.get("run_direction_mode_comparison", False):
        direction_mode_comparison = run_direction_mode_comparison(
            config,
            direction_symbols,
            price_data,
        )
    permutation_1000_results = pd.DataFrame()
    permutation_1000_summary = pd.DataFrame()
    quasi_p_values_1000 = pd.DataFrame()
    time_split_validation = pd.DataFrame()
    sizing_stress_summary = pd.DataFrame()
    final_verdict_markdown = ""
    if run_flags.get("run_final_validation", False):
        final_validation = run_final_validation_suite(
            config=config,
            primary_symbol=primary_symbol,
            price_data=price_data,
            output_dir=output_dir,
            baseline_summary=summary_by_symbol,
            strategy_vs_buy_hold=strategy_vs_buy_hold,
        )
        permutation_1000_results = final_validation["permutation_1000_results"]
        permutation_1000_summary = final_validation["permutation_1000_summary"]
        quasi_p_values_1000 = final_validation["quasi_p_values_1000"]
        time_split_validation = final_validation["time_split_validation"]
        direction_mode_comparison = final_validation["direction_mode_comparison"]
        sizing_stress_summary = final_validation["sizing_stress_summary"]
        final_verdict_markdown = final_validation["final_verdict_markdown"]
    final_verdict_summary = build_final_verdict_summary(
        strategy_vs_buy_hold=strategy_vs_buy_hold,
        cost_stress_summary=cost_stress_summary,
        monte_carlo_skip_summary=mc_summary,
        trade_sequence_randomization_summary=sequence_summary,
        baseline_vs_optimized_summary=baseline_vs_optimized_summary,
        quasi_p_values=quasi_p_values,
        direction_mode_comparison=direction_mode_comparison,
    )
    if run_flags.get("run_final_validation", False):
        final_verdict_summary = build_final_validation_verdict_summary(
            baseline_summary=summary_by_symbol,
            strategy_vs_buy_hold=strategy_vs_buy_hold,
            permutation_1000_summary=permutation_1000_summary,
            time_split_validation=time_split_validation,
            direction_mode_comparison=direction_mode_comparison,
            sizing_stress_summary=sizing_stress_summary,
        )

    paths = write_simple_strategy_01_outputs(
        output_dir=output_dir,
        summary_by_symbol=summary_by_symbol,
        trade_log=trade_log,
        equity_curves=equity_curves,
        strategy_vs_buy_hold=strategy_vs_buy_hold,
        monte_carlo_skip_results=mc_results,
        monte_carlo_skip_summary=mc_summary,
        trade_sequence_randomization=sequence_results,
        trade_sequence_randomization_summary=sequence_summary,
        permutation_results=permutation_results,
        permutation_summary=permutation_summary,
        optimization_results=optimization_results,
        top_20_results=top_20_results,
        parameter_stability=parameter_stability,
        baseline_vs_optimized_summary=baseline_vs_optimized_summary,
        quasi_p_values=quasi_p_values,
        permutation_1000_results=permutation_1000_results,
        permutation_1000_summary=permutation_1000_summary,
        quasi_p_values_1000=quasi_p_values_1000,
        time_split_validation=time_split_validation,
        direction_mode_comparison=direction_mode_comparison,
        sizing_stress_summary=sizing_stress_summary,
        final_verdict_summary=final_verdict_summary,
        final_verdict_markdown=(
            final_verdict_markdown
            or build_final_verdict_markdown(final_verdict_summary)
        ),
        cost_stress_summary=cost_stress_summary,
        missing_symbols=missing_symbols,
        config_summary={
            "config_path": str(config_path),
            "stage": stage,
            "primary_symbol": primary_symbol,
            "baseline_symbols": baseline_symbols,
            "optimization_symbols": optimization_symbols,
            "permutation_symbols": permutation_symbols,
            "direction_mode_symbols": direction_symbols,
            "run_flags": run_flags,
            "base_output_dir": str(output_layout.base_dir),
            "run_output_dir": str(output_layout.run_dir),
            "latest_dir": str(output_layout.latest_dir),
            "archived_root_files": [str(dst) for _, dst in output_layout.moved_files],
            "archived_aborted_dirs": [str(dst) for _, dst in output_layout.moved_dirs],
        },
    )
    if output_config.get("copy_final_outputs_to_latest", True):
        latest_paths = copy_final_outputs_to_latest(paths, output_layout.latest_dir)
        paths.update(latest_paths)
    return SimpleStrategy01LabResult(
        strategy=strategy,
        output_dir=output_dir,
        paths=paths,
        missing_symbols=missing_symbols,
        stage=stage,
        summary_by_symbol=summary_by_symbol,
        strategy_vs_buy_hold=strategy_vs_buy_hold,
        trade_log=trade_log,
    )


def apply_stage_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply staged run-mode defaults while preserving explicit config values."""
    out = deepcopy(config)
    stage = str(out.get("stage", "custom")).lower()
    run_flags = dict(out.get("run", {}) or {})

    profiles: dict[str, dict[str, bool]] = {
        "stage_1_pine_parity": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_2_peer_baseline": {
            "run_baseline": True,
            "run_peer_universe": True,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_3_cost_stress": {
            "run_baseline": True,
            "run_peer_universe": True,
            "run_cost_stress": True,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_4_monte_carlo": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": True,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_5_trade_sequence": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": True,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_6_optimization": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": True,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_6a_baseline_neighbors": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": True,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_7_permutation": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": True,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_2_to_5": {
            "run_baseline": True,
            "run_peer_universe": True,
            "run_cost_stress": True,
            "run_monte_carlo_skip": True,
            "run_trade_sequence_randomization": True,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_6_7": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": True,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
        "stage_8_combined": {
            "run_baseline": True,
            "run_peer_universe": True,
            "run_cost_stress": True,
            "run_monte_carlo_skip": True,
            "run_trade_sequence_randomization": True,
            "run_permutation_test": True,
            "run_optimization": True,
            "run_direction_mode_comparison": True,
            "run_final_validation": False,
        },
        "final_validation": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": False,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": True,
        },
        "permutation_1000": {
            "run_baseline": True,
            "run_peer_universe": False,
            "run_cost_stress": False,
            "run_monte_carlo_skip": False,
            "run_trade_sequence_randomization": False,
            "run_permutation_test": True,
            "run_optimization": False,
            "run_direction_mode_comparison": False,
            "run_final_validation": False,
        },
    }
    if stage in profiles:
        merged = dict(run_flags)
        merged.update(profiles[stage])
        out["run"] = merged
    if stage == "permutation_1000":
        perm_config = dict(out.get("permutation_test", {}) or {})
        perm_config["num_runs"] = 1000
        out["permutation_test"] = perm_config
    active_flags = out.get("run", {}) or {}
    if active_flags.get("run_optimization") and active_flags.get("run_permutation_test"):
        active_flags["run_permutation_test"] = False
        out["run"] = active_flags
        print(
            "Optimization and permutation were both requested. "
            "Disabling permutation for this process; run stage_7_permutation separately."
        )
    return out


def add_trade_contribution_columns(
    summary_by_symbol: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> pd.DataFrame:
    """Add long/short counts, PnL, and exit reason counts to summary rows."""
    if summary_by_symbol.empty:
        return summary_by_symbol
    out = summary_by_symbol.copy()
    for column in [
        "long_trade_count",
        "short_trade_count",
        "long_net_pnl",
        "short_net_pnl",
        "exit_reason_counts",
    ]:
        if column not in out.columns:
            if column in {"long_net_pnl", "short_net_pnl"}:
                out[column] = 0.0
            elif column == "exit_reason_counts":
                out[column] = "{}"
            else:
                out[column] = 0
    if trade_log.empty:
        return out

    for idx, row in out.iterrows():
        symbol = str(row["symbol"]).upper()
        trades = trade_log[trade_log["symbol"].astype(str).str.upper() == symbol]
        long_trades = trades[trades["side"].astype(str).str.lower() == "long"]
        short_trades = trades[trades["side"].astype(str).str.lower() == "short"]
        out.loc[idx, "long_trade_count"] = int(len(long_trades))
        out.loc[idx, "short_trade_count"] = int(len(short_trades))
        out.loc[idx, "long_net_pnl"] = float(long_trades["net_pnl"].sum()) if not long_trades.empty else 0.0
        out.loc[idx, "short_net_pnl"] = float(short_trades["net_pnl"].sum()) if not short_trades.empty else 0.0
        counts = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades else {}
        out.loc[idx, "exit_reason_counts"] = json.dumps(counts, sort_keys=True)
    return out


def build_strategy_vs_buy_hold(
    summary_by_symbol: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare per-symbol strategy return to buy-and-hold return."""
    rows: list[dict] = []
    if summary_by_symbol.empty:
        return pd.DataFrame()
    for _, row in summary_by_symbol.iterrows():
        symbol = str(row["symbol"]).upper()
        prices = price_data.get(symbol)
        if prices is None or prices.empty:
            continue
        buy_hold_return = float(prices["close"].iloc[-1] / prices["close"].iloc[0] - 1.0)
        strategy_return = float(row.get("total_return", 0.0))
        rows.append(
            {
                "symbol": symbol,
                "strategy_total_return": strategy_return,
                "buy_hold_return": buy_hold_return,
                "excess_vs_buy_hold": strategy_return - buy_hold_return,
                "strategy_cagr": row.get("cagr"),
                "strategy_max_drawdown": row.get("max_drawdown"),
                "strategy_profit_factor": row.get("profit_factor"),
                "strategy_win_rate": row.get("win_rate"),
                "number_of_trades": row.get("number_of_trades"),
                "long_trade_count": row.get("long_trade_count"),
                "short_trade_count": row.get("short_trade_count"),
                "long_net_pnl": row.get("long_net_pnl"),
                "short_net_pnl": row.get("short_net_pnl"),
                "exit_reason_counts": row.get("exit_reason_counts"),
            }
        )
    return pd.DataFrame(rows)


def build_trade_sequence_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return Stage 5 fields in one compact table."""
    if results.empty:
        return pd.DataFrame()
    final_equity = results["final_equity"].astype(float)
    max_drawdown = results["max_drawdown"].astype(float)
    summary = {
        "median_final_equity": float(final_equity.median()),
        "p05_final_equity": float(final_equity.quantile(0.05)),
        "worst_final_equity": float(final_equity.min()),
        "median_max_drawdown": float(max_drawdown.median()),
        "p05_max_drawdown": float(max_drawdown.quantile(0.05)),
        "worst_max_drawdown": float(max_drawdown.min()),
    }
    if "longest_losing_streak" in results:
        summary["median_longest_losing_streak"] = float(results["longest_losing_streak"].median())
        summary["worst_longest_losing_streak"] = float(results["longest_losing_streak"].max())
    compact = summarize_trade_sequence_randomization(results)
    if not compact.empty:
        for _, row in compact.iterrows():
            metric = str(row["metric"])
            summary[f"{metric}_p95"] = row.get("p95")
    return pd.DataFrame([summary])


def _buy_hold_return(prices: pd.DataFrame | None) -> float | None:
    if prices is None or prices.empty:
        return None
    return float(prices["close"].iloc[-1] / prices["close"].iloc[0] - 1.0)


def run_cost_stress_cases(
    config: dict[str, Any],
    symbols: list[str],
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run configured slippage stress cases on selected symbols."""
    costs = config.get("costs", {}) or {}
    stress_values = costs.get("stress_slippage_bps_per_side", [5, 10, 25])
    baseline_bps = float(costs.get("slippage_bps_per_side", 0.0))
    values = [baseline_bps] + [float(value) for value in stress_values]

    rows: list[dict] = []
    for bps in values:
        stress_config = deepcopy(config)
        stress_config.setdefault("costs", {})
        stress_config["costs"]["slippage_bps_per_side"] = bps
        strategy = _build_strategy(stress_config, symbols)
        stress_data = {
            symbol: frame
            for symbol, frame in price_data.items()
            if symbol in collect_required_symbols(strategy)
        }
        run = run_strategy_definition(strategy, stress_data)
        if run.summary.empty:
            continue
        stress_summary = add_trade_contribution_columns(run.summary, run.trades)
        for _, result_row in stress_summary.iterrows():
            row = result_row.to_dict()
            row["scenario"] = "baseline_costs" if bps == baseline_bps else f"slippage_{bps:g}_bps"
            row["slippage_bps_per_side"] = bps
            rows.append(row)
    return pd.DataFrame(rows)


def run_stage6_parameter_stability(
    config: dict[str, Any],
    symbols: list[str],
    price_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the Stage 6 per-symbol parameter stability grid."""
    opt_config = config.get("optimization", {}) or {}
    parameter_grid = resolve_optimization_parameter_grid(opt_config)
    parameter_columns = list(parameter_grid.keys())
    objective = str(opt_config.get("objective", "balanced"))
    top_n = int(opt_config.get("top_n", 20))
    max_combinations = int(opt_config.get("max_parameter_combinations", 500))
    optimization_mode = str(opt_config.get("optimization_mode", "baseline_neighbors"))
    output_dir = Path(output_dir)
    partial_path = output_dir / "optimization_results_partial.csv"

    all_results: list[pd.DataFrame] = []
    all_top: list[pd.DataFrame] = []
    baseline_rows: list[dict] = []
    available_symbols = [symbol for symbol in normalize_symbols(symbols) if symbol in price_data]
    all_configs = list(_iter_effective_parameter_grid(parameter_grid))
    raw_config_count = len(all_configs)
    if raw_config_count > 2000:
        print(
            f"Parameter grid has {raw_config_count} combinations. "
            "This may be too slow. Use baseline_neighbors first."
        )
    if raw_config_count > max_combinations:
        print(
            f"Limiting Stage 6 optimization from {raw_config_count} to "
            f"{max_combinations} configs per symbol."
        )
        all_configs = all_configs[:max_combinations]
    print(
        f"Stage 6 optimization mode: {optimization_mode}; "
        f"symbols: {', '.join(available_symbols)}; "
        f"configs per symbol: {len(all_configs)}; "
        f"total backtests planned: {len(all_configs) * len(available_symbols)}"
    )

    completed_total = 0
    started_at = time.perf_counter()
    current_symbol = ""
    rows: list[dict] = []
    baseline_summary: dict[str, Any] = {}
    baseline_score = 0.0
    buy_hold_return: float | None = None

    try:
        for symbol in available_symbols:
            current_symbol = symbol
            print(f"Stage 6A optimizing symbol: {symbol}")
            print(f"{symbol}: {len(all_configs)} configs queued")
            symbol_started_at = time.perf_counter()
            strategy = _build_strategy(config, [symbol])
            run_data = {
                candidate_symbol: frame
                for candidate_symbol, frame in price_data.items()
                if candidate_symbol in collect_required_symbols(strategy)
            }
            buy_hold_return = _buy_hold_return(price_data.get(symbol))
            baseline_run = run_strategy_definition(strategy, run_data)
            baseline_summary = (
                baseline_run.summary.iloc[0].to_dict()
                if not baseline_run.summary.empty
                else {}
            )
            baseline_score = score_summary(baseline_summary, objective)

            rows = []
            for run_number, parameters in enumerate(all_configs, start=1):
                candidate = strategy.clone_with_parameters(parameters)
                run = run_strategy_definition(candidate, run_data)
                summary = run.summary.iloc[0].to_dict() if not run.summary.empty else {}
                total_return = _to_float(summary.get("total_return"))
                max_drawdown = _to_float(summary.get("max_drawdown"))
                row = dict(summary)
                row.update(_side_contribution_for_symbol(run.trades, symbol))
                row.update(
                    {
                        "symbol": symbol,
                        "optimization_mode": optimization_mode,
                        "optimization_run": run_number,
                        "objective": objective,
                        "score": score_summary(summary, objective),
                        "parameters": json.dumps(parameters, sort_keys=True),
                        "buy_hold_return": buy_hold_return,
                        "excess_vs_buy_hold": (
                            total_return - buy_hold_return
                            if buy_hold_return is not None
                            else None
                        ),
                        "profitable": total_return > 0,
                        "beats_buy_hold": (
                            total_return > buy_hold_return
                            if buy_hold_return is not None
                            else False
                        ),
                        "max_drawdown_worse_than_70pct": max_drawdown < -0.70,
                        "baseline_parameter_match": _is_baseline_parameter_set(parameters),
                        "near_baseline_parameter_set": _is_near_baseline_parameter_set(parameters),
                    }
                )
                for parameter in parameter_columns:
                    row[parameter] = _parameter_value_for_table(parameters.get(parameter))
                rows.append(row)
                completed_total += 1
                if run_number % 25 == 0 or run_number == len(all_configs):
                    elapsed = time.perf_counter() - symbol_started_at
                    print(
                        f"{symbol}: completed {run_number}/{len(all_configs)} configs "
                        f"({elapsed:.1f}s)"
                    )
                if completed_total % 50 == 0:
                    _write_optimization_partial(
                        partial_path,
                        all_results,
                        pd.DataFrame(rows),
                        completed_total,
                        interrupted=False,
                    )
                    print(
                        f"Wrote partial optimization results after "
                        f"{completed_total} completed configs: {partial_path}"
                    )

            symbol_results = _finalize_symbol_optimization_results(pd.DataFrame(rows), top_n)
            if symbol_results.empty:
                continue
            all_results.append(symbol_results)
            all_top.append(symbol_results.head(top_n).copy())
            baseline_rows.append(
                _build_baseline_vs_optimized_row(
                    symbol=symbol,
                    baseline_summary=baseline_summary,
                    baseline_score=baseline_score,
                    ranked_results=symbol_results,
                    buy_hold_return=buy_hold_return,
                )
            )
    except KeyboardInterrupt:
        print(
            f"Stage 6 optimization interrupted after {completed_total} completed configs."
        )
        current_results = _finalize_symbol_optimization_results(pd.DataFrame(rows), top_n)
        if not current_results.empty:
            all_results.append(current_results)
            all_top.append(current_results.head(top_n).copy())
            if current_symbol:
                baseline_rows.append(
                    _build_baseline_vs_optimized_row(
                        symbol=current_symbol,
                        baseline_summary=baseline_summary,
                        baseline_score=baseline_score,
                        ranked_results=current_results,
                        buy_hold_return=buy_hold_return,
                    )
                )
        _write_optimization_partial(
            partial_path,
            all_results,
            pd.DataFrame(),
            completed_total,
            interrupted=True,
        )
        print(f"Interrupted partial results saved to: {partial_path}")

    optimization_results = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame()
    )
    top_20_results = pd.concat(all_top, ignore_index=True) if all_top else pd.DataFrame()
    parameter_stability = build_stage6_parameter_stability(
        optimization_results,
        parameter_columns,
    )
    baseline_vs_optimized_summary = pd.DataFrame(baseline_rows)
    runtime_seconds = time.perf_counter() - started_at
    if not baseline_vs_optimized_summary.empty:
        baseline_vs_optimized_summary["stage6_runtime_seconds"] = runtime_seconds
        baseline_vs_optimized_summary["stage6_completed_configs_total"] = completed_total
        baseline_vs_optimized_summary["optimization_mode"] = optimization_mode
    _write_optimization_partial(
        partial_path,
        all_results,
        pd.DataFrame(),
        completed_total,
        interrupted=False,
    )
    print(
        f"Stage 6 optimization finished with {completed_total} completed configs "
        f"in {runtime_seconds:.1f}s."
    )
    return (
        optimization_results,
        top_20_results,
        parameter_stability,
        baseline_vs_optimized_summary,
    )


def run_stage7_permutation_tests(
    config: dict[str, Any],
    symbols: list[str],
    price_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Stage 7 permutation tests per requested symbol."""
    perm_config = config.get("permutation_test", {}) or {}
    objective_metrics = perm_config.get(
        "objective_metrics",
        [
            "total_return",
            "cagr",
            "max_drawdown",
            "profit_factor",
            "return_over_max_drawdown",
            "final_equity",
        ],
    )
    num_runs = int(perm_config.get("num_runs", 200))
    random_seed = int(perm_config.get("random_seed", 0))
    checkpoint_every = int(perm_config.get("checkpoint_every", 25))
    progress_every = int(perm_config.get("progress_every", 25))
    output_dir = Path(output_dir)
    all_results: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for symbol in [item for item in normalize_symbols(symbols) if item in price_data]:
        print(f"Stage 7 permutation symbol: {symbol}")
        strategy = _build_strategy(config, [symbol])
        run_data = {
            candidate_symbol: frame
            for candidate_symbol, frame in price_data.items()
            if candidate_symbol in collect_required_symbols(strategy)
        }
        checkpoint_path = output_dir / f"permutation_results_{symbol}_partial.csv"
        results, summary = run_permutation_test(
            strategy=strategy,
            symbol=symbol,
            price_data=run_data,
            num_runs=num_runs,
            random_seed=random_seed,
            objective_metrics=objective_metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            progress_every=progress_every,
        )
        if not results.empty:
            results = results.copy()
            if "symbol" not in results:
                results.insert(0, "symbol", symbol)
            all_results.append(results)
            pd.concat(all_results, ignore_index=True).to_csv(
                output_dir / "permutation_results_partial.csv",
                index=False,
            )
        if not summary.empty:
            summary = summary.copy()
            if "symbol" not in summary:
                summary.insert(0, "symbol", symbol)
            for idx, row in summary.iterrows():
                metric = str(row["metric"])
                values = (
                    pd.to_numeric(results.get(metric, pd.Series(dtype=float)), errors="coerce")
                    .replace([float("inf"), float("-inf")], pd.NA)
                    .dropna()
                )
                summary.loc[idx, "permutation_max"] = (
                    float(values.max()) if len(values) else None
                )
                summary.loc[idx, "pass_fail_flag"] = _permutation_pass_fail(
                    row.get("quasi_p_value")
                )
                summary.loc[idx, "comparison_rule"] = (
                    "permuted >= real; for max_drawdown, less severe negative drawdown is better"
                    if metric == "max_drawdown"
                    else "permuted >= real"
                )
            all_summary.append(summary)

    permutation_results = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame()
    )
    permutation_summary = (
        pd.concat(all_summary, ignore_index=True)
        if all_summary
        else pd.DataFrame()
    )
    quasi_columns = [
        "symbol",
        "metric",
        "real_value",
        "permutation_p95",
        "permutation_max",
        "quasi_p_value",
        "num_permutations",
        "random_seed_base",
        "pass_fail_flag",
        "comparison_rule",
    ]
    quasi_p_values = (
        permutation_summary[[col for col in quasi_columns if col in permutation_summary]]
        if not permutation_summary.empty
        else pd.DataFrame()
    )
    return permutation_results, permutation_summary, quasi_p_values


def run_final_validation_suite(
    config: dict[str, Any],
    primary_symbol: str,
    price_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    baseline_summary: pd.DataFrame,
    strategy_vs_buy_hold: pd.DataFrame,
) -> dict[str, Any]:
    """Run the requested final-validation checks without optimization."""
    symbol = primary_symbol.upper()
    output_dir = Path(output_dir)
    permutation_results, permutation_summary, quasi_p_values = run_crwd_permutation_1000(
        config,
        symbol,
        price_data,
        output_dir,
    )
    time_split_validation = run_time_split_validation(config, symbol, price_data)
    direction_mode_comparison = run_final_direction_mode_comparison(config, symbol, price_data)
    sizing_stress_summary = run_sizing_stress_summary(config, symbol, price_data)
    verdict_summary = build_final_validation_verdict_summary(
        baseline_summary=baseline_summary,
        strategy_vs_buy_hold=strategy_vs_buy_hold,
        permutation_1000_summary=permutation_summary,
        time_split_validation=time_split_validation,
        direction_mode_comparison=direction_mode_comparison,
        sizing_stress_summary=sizing_stress_summary,
    )
    return {
        "permutation_1000_results": permutation_results,
        "permutation_1000_summary": permutation_summary,
        "quasi_p_values_1000": quasi_p_values,
        "time_split_validation": time_split_validation,
        "direction_mode_comparison": direction_mode_comparison,
        "sizing_stress_summary": sizing_stress_summary,
        "final_verdict_summary": verdict_summary,
        "final_verdict_markdown": build_final_verdict_markdown(verdict_summary),
    }


def run_crwd_permutation_1000(
    config: dict[str, Any],
    symbol: str,
    price_data: dict[str, pd.DataFrame],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the final CRWD 1000-permutation robustness filter."""
    perm_config = config.get("permutation_test", {}) or {}
    metrics = [
        "final_equity",
        "total_return",
        "cagr",
        "profit_factor",
        "return_over_max_drawdown",
        "max_drawdown",
    ]
    num_runs = int(perm_config.get("final_validation_num_runs", 1000))
    random_seed = int(perm_config.get("random_seed", 42))
    checkpoint_every = int(perm_config.get("final_validation_checkpoint_every", 50))
    progress_every = int(perm_config.get("final_validation_progress_every", 50))
    strategy = _build_strategy(config, [symbol])
    run_data = {
        candidate_symbol: frame
        for candidate_symbol, frame in price_data.items()
        if candidate_symbol in collect_required_symbols(strategy)
    }
    results, summary = run_permutation_test(
        strategy=strategy,
        symbol=symbol,
        price_data=run_data,
        num_runs=num_runs,
        random_seed=random_seed,
        objective_metrics=metrics,
        checkpoint_path=output_dir / f"permutation_1000_results_{symbol}_partial.csv",
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
    )
    if not summary.empty:
        summary = summary.copy()
        for idx, row in summary.iterrows():
            metric = str(row["metric"])
            summary.loc[idx, "pass_fail_flag"] = _permutation_pass_fail(
                row.get("quasi_p_value")
            )
            summary.loc[idx, "comparison_rule"] = (
                "permuted >= real; for max_drawdown, less severe negative drawdown is better"
                if metric == "max_drawdown"
                else "permuted >= real"
            )
    quasi_columns = [
        "symbol",
        "metric",
        "real_value",
        "permutation_median",
        "permutation_p95",
        "permutation_max",
        "quasi_p_value",
        "num_permutations",
        "random_seed_base",
        "pass_fail_flag",
        "comparison_rule",
    ]
    quasi = (
        summary[[col for col in quasi_columns if col in summary]]
        if not summary.empty
        else pd.DataFrame()
    )
    return results, summary, quasi


def run_time_split_validation(
    config: dict[str, Any],
    symbol: str,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run the frozen strategy on requested CRWD time splits."""
    prices = _normalized_prices(price_data.get(symbol))
    if prices.empty:
        return pd.DataFrame()
    available_start = prices.index.min()
    available_end = prices.index.max()
    mid_pos = max(len(prices) // 2, 1)
    first_half_end = prices.index[mid_pos - 1]
    second_half_start = prices.index[mid_pos]
    splits = [
        ("ipo_to_2021_12_31", available_start, pd.Timestamp("2021-12-31")),
        ("2022_01_01_to_2023_12_31", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
        ("2024_01_01_to_latest", pd.Timestamp("2024-01-01"), available_end),
        ("first_half_available", available_start, first_half_end),
        ("second_half_available", second_half_start, available_end),
    ]
    rows: list[dict[str, Any]] = []
    for split_name, start, end in splits:
        start = max(pd.Timestamp(start), available_start)
        end = min(pd.Timestamp(end), available_end)
        split_prices = prices.loc[(prices.index >= start) & (prices.index <= end)].copy()
        row = {
            "symbol": symbol,
            "split_name": split_name,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "bars": int(len(split_prices)),
        }
        if len(split_prices) < int((config.get("strategy_parameters", {}) or {}).get("min_bars", 80)):
            row["status"] = "skipped_too_few_bars"
            rows.append(row)
            continue
        run = run_strategy_definition(_build_strategy(config, [symbol]), {symbol: split_prices})
        if run.summary.empty:
            row["status"] = "no_summary"
            rows.append(row)
            continue
        summary = add_trade_contribution_columns(run.summary, run.trades).iloc[0].to_dict()
        buy_hold = _buy_hold_return(split_prices)
        total_return = _to_float(summary.get("total_return"))
        row.update(summary)
        row.update(
            {
                "status": "ok",
                "buy_hold_return": buy_hold,
                "excess_vs_buy_hold": total_return - buy_hold if buy_hold is not None else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_final_direction_mode_comparison(
    config: dict[str, Any],
    symbol: str,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run the requested CRWD direction-mode comparison."""
    mode_specs = [
        (
            "long_short_baseline",
            {
                "direction_mode": "long_short",
                "enable_long_entries": True,
                "enable_short_entries": True,
                "use_opposite_signal_as_exit": True,
                "if_side_disabled_use_signal_only_as_exit": True,
            },
            None,
        ),
        (
            "long_only_short_signal_exit",
            {
                "direction_mode": "long_only",
                "enable_long_entries": True,
                "enable_short_entries": False,
                "use_opposite_signal_as_exit": True,
                "if_side_disabled_use_signal_only_as_exit": True,
            },
            None,
        ),
        (
            "long_only_fixed_bars_only",
            {
                "direction_mode": "long_only",
                "enable_long_entries": True,
                "enable_short_entries": False,
                "use_opposite_signal_as_exit": False,
                "if_side_disabled_use_signal_only_as_exit": False,
            },
            None,
        ),
        (
            "short_only",
            {},
            "short_only_long_signal_exit",
        ),
    ]
    rows: list[dict[str, Any]] = []
    buy_hold = _buy_hold_return(price_data.get(symbol))
    for mode_label, overrides, custom_mode in mode_specs:
        mode_config = deepcopy(config)
        params = dict(mode_config.get("strategy_parameters", {}) or {})
        params.update(overrides)
        mode_config["strategy_parameters"] = params
        strategy = (
            build_short_only_with_long_signal_exit_strategy(mode_config, symbol)
            if custom_mode
            else _build_strategy(mode_config, [symbol])
        )
        run_data = {
            candidate_symbol: frame
            for candidate_symbol, frame in price_data.items()
            if candidate_symbol in collect_required_symbols(strategy)
        }
        run = run_strategy_definition(strategy, run_data)
        if run.summary.empty:
            continue
        summary = add_trade_contribution_columns(run.summary, run.trades).iloc[0].to_dict()
        total_return = _to_float(summary.get("total_return"))
        summary.update(
            {
                "mode_label": mode_label,
                "buy_hold_return": buy_hold,
                "excess_vs_buy_hold": total_return - buy_hold if buy_hold is not None else None,
            }
        )
        rows.append(summary)
    return _front_columns(pd.DataFrame(rows), [
        "symbol",
        "mode_label",
        "total_return",
        "cagr",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "number_of_trades",
        "buy_hold_return",
        "excess_vs_buy_hold",
        "long_trade_count",
        "short_trade_count",
        "long_net_pnl",
        "short_net_pnl",
        "exit_reason_counts",
    ])


def run_sizing_stress_summary(
    config: dict[str, Any],
    symbol: str,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run CRWD baseline at requested percent-equity allocations."""
    rows: list[dict[str, Any]] = []
    buy_hold = _buy_hold_return(price_data.get(symbol))
    for allocation in [100, 50, 25, 10]:
        sizing_config = deepcopy(config)
        sizing_config.setdefault("sizing", {})
        sizing_config["sizing"]["percent_equity"] = allocation
        strategy = _build_strategy(sizing_config, [symbol])
        run_data = {
            candidate_symbol: frame
            for candidate_symbol, frame in price_data.items()
            if candidate_symbol in collect_required_symbols(strategy)
        }
        run = run_strategy_definition(strategy, run_data)
        if run.summary.empty:
            continue
        row = run.summary.iloc[0].to_dict()
        total_return = _to_float(row.get("total_return"))
        row.update(
            {
                "allocation_pct": allocation,
                "buy_hold_return": buy_hold,
                "excess_vs_buy_hold": total_return - buy_hold if buy_hold is not None else None,
            }
        )
        rows.append(row)
    return _front_columns(pd.DataFrame(rows), [
        "symbol",
        "allocation_pct",
        "total_return",
        "cagr",
        "max_drawdown",
        "profit_factor",
        "final_equity",
        "worst_trade",
        "best_trade",
        "number_of_trades",
        "buy_hold_return",
        "excess_vs_buy_hold",
    ])


def run_direction_mode_comparison(
    config: dict[str, Any],
    symbols: list[str],
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare long/short, long-only, and short-only variants without optimizing."""
    mode_specs = [
        (
            "long_short_baseline",
            "Frozen corrected Pine parity: long and short entries enabled.",
            {
                "direction_mode": "long_short",
                "enable_long_entries": True,
                "enable_short_entries": True,
                "use_opposite_signal_as_exit": True,
                "if_side_disabled_use_signal_only_as_exit": True,
            },
        ),
        (
            "long_only_opposite_signal_exit",
            "Long entries only; short signal can exit longs but cannot open shorts.",
            {
                "direction_mode": "long_only",
                "enable_long_entries": True,
                "enable_short_entries": False,
                "use_opposite_signal_as_exit": True,
                "if_side_disabled_use_signal_only_as_exit": True,
            },
        ),
        (
            "long_only_fixed_bars_exit_only",
            "Long entries only; no opposite-signal exit, fixed-bar exit remains.",
            {
                "direction_mode": "long_only",
                "enable_long_entries": True,
                "enable_short_entries": False,
                "use_opposite_signal_as_exit": False,
                "if_side_disabled_use_signal_only_as_exit": False,
            },
        ),
        (
            "short_only_fixed_bars_exit",
            "Short entries only; fixed-bar exit remains.",
            {
                "direction_mode": "short_only",
                "enable_long_entries": False,
                "enable_short_entries": True,
                "use_opposite_signal_as_exit": False,
                "if_side_disabled_use_signal_only_as_exit": False,
            },
        ),
    ]
    rows: list[dict] = []
    for symbol in [item for item in normalize_symbols(symbols) if item in price_data]:
        for mode_label, description, overrides in mode_specs:
            mode_config = deepcopy(config)
            mode_params = dict(mode_config.get("strategy_parameters", {}) or {})
            mode_params.update(overrides)
            mode_config["strategy_parameters"] = mode_params
            strategy = _build_strategy(mode_config, [symbol])
            run_data = {
                candidate_symbol: frame
                for candidate_symbol, frame in price_data.items()
                if candidate_symbol in collect_required_symbols(strategy)
            }
            run = run_strategy_definition(strategy, run_data)
            if run.summary.empty:
                continue
            summary = add_trade_contribution_columns(run.summary, run.trades)
            row = summary.iloc[0].to_dict()
            buy_hold_return = _buy_hold_return(price_data.get(symbol))
            strategy_return = _to_float(row.get("total_return"))
            row.update(
                {
                    "mode_label": mode_label,
                    "mode_description": description,
                    "buy_hold_return": buy_hold_return,
                    "excess_vs_buy_hold": (
                        strategy_return - buy_hold_return
                        if buy_hold_return is not None
                        else None
                    ),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = [
        "symbol",
        "mode_label",
        "mode_description",
        "total_return",
        "cagr",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "number_of_trades",
        "buy_hold_return",
        "excess_vs_buy_hold",
        "long_trade_count",
        "short_trade_count",
        "long_net_pnl",
        "short_net_pnl",
        "exit_reason_counts",
    ]
    return out[[col for col in front if col in out] + [col for col in out.columns if col not in front]]


def build_stage6_parameter_stability(
    optimization_results: pd.DataFrame,
    parameter_columns: list[str],
) -> pd.DataFrame:
    """Summarize median and robustness by parameter value for Stage 6."""
    if optimization_results.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for symbol, symbol_results in optimization_results.groupby("symbol", dropna=False):
        for parameter in parameter_columns:
            if parameter not in symbol_results:
                continue
            for value, group in symbol_results.groupby(parameter, dropna=False):
                total_return = pd.to_numeric(group["total_return"], errors="coerce")
                max_drawdown = pd.to_numeric(group["max_drawdown"], errors="coerce")
                rows.append(
                    {
                        "symbol": symbol,
                        "parameter": parameter,
                        "value": value,
                        "runs": int(len(group)),
                        "median_score": float(pd.to_numeric(group["score"], errors="coerce").median()),
                        "median_total_return": float(total_return.median()),
                        "p25_total_return": float(total_return.quantile(0.25)),
                        "p75_total_return": float(total_return.quantile(0.75)),
                        "median_profit_factor": float(
                            pd.to_numeric(group["profit_factor"], errors="coerce").median()
                        ),
                        "pct_profitable": float(group["profitable"].mean()),
                        "pct_beating_buy_hold": float(group["beats_buy_hold"].mean()),
                        "pct_max_drawdown_worse_than_70pct": float(
                            (max_drawdown < -0.70).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_final_verdict_summary(
    strategy_vs_buy_hold: pd.DataFrame,
    cost_stress_summary: pd.DataFrame,
    monte_carlo_skip_summary: pd.DataFrame,
    trade_sequence_randomization_summary: pd.DataFrame,
    baseline_vs_optimized_summary: pd.DataFrame,
    quasi_p_values: pd.DataFrame,
    direction_mode_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact final-verdict table for the HTML report."""
    rows: list[dict] = []
    if not strategy_vs_buy_hold.empty:
        returns = pd.to_numeric(strategy_vs_buy_hold["strategy_total_return"], errors="coerce")
        excess = pd.to_numeric(strategy_vs_buy_hold["excess_vs_buy_hold"], errors="coerce")
        profitable = int((returns > 0).sum())
        beating = int((excess > 0).sum())
        total = int(len(strategy_vs_buy_hold))
        rows.append(
            {
                "area": "Peer basket",
                "finding": f"{profitable}/{total} symbols were profitable; {beating}/{total} beat buy-and-hold.",
                "verdict": "CRWD is not alone, but broad buy-and-hold outperformance is limited.",
            }
        )
    if not cost_stress_summary.empty:
        crwd_25 = cost_stress_summary[
            (cost_stress_summary["symbol"].astype(str).str.upper() == "CRWD")
            & (pd.to_numeric(cost_stress_summary["slippage_bps_per_side"], errors="coerce") == 25)
        ]
        if not crwd_25.empty:
            row = crwd_25.iloc[0]
            rows.append(
                {
                    "area": "Cost stress",
                    "finding": (
                        f"CRWD total return at 25 bps/side: "
                        f"{_to_float(row.get('total_return')):.4f}; "
                        f"profit factor: {_to_float(row.get('profit_factor')):.4f}."
                    ),
                    "verdict": "The CRWD long-side edge survives aggressive cost stress, while shorts are much thinner.",
                }
            )
    if not monte_carlo_skip_summary.empty:
        skip_10 = monte_carlo_skip_summary[
            pd.to_numeric(monte_carlo_skip_summary["skip_pct"], errors="coerce") == 10
        ]
        if not skip_10.empty:
            row = skip_10.iloc[0]
            rows.append(
                {
                    "area": "Monte Carlo skip",
                    "finding": (
                        f"10% skip median return: {_to_float(row.get('median_total_return')):.4f}; "
                        f"5th percentile: {_to_float(row.get('p05_total_return')):.4f}; "
                        f"beat buy-and-hold: {_to_float(row.get('pct_beating_buy_hold')):.2%}."
                    ),
                    "verdict": "Skipping trades dents the curve but does not immediately erase CRWD robustness.",
                }
            )
    if not trade_sequence_randomization_summary.empty:
        row = trade_sequence_randomization_summary.iloc[0]
        rows.append(
            {
                "area": "Trade sequence",
                "finding": (
                    f"Worst shuffled max drawdown: "
                    f"{_to_float(row.get('worst_max_drawdown')):.4f}; "
                    f"median max drawdown: {_to_float(row.get('median_max_drawdown')):.4f}."
                ),
                "verdict": "Same trade returns can produce materially worse paths; path risk remains high.",
            }
        )
    if not baseline_vs_optimized_summary.empty:
        crwd = baseline_vs_optimized_summary[
            baseline_vs_optimized_summary["symbol"].astype(str).str.upper() == "CRWD"
        ]
        if not crwd.empty:
            row = crwd.iloc[0]
            rows.append(
                {
                    "area": "Parameter stability",
                    "finding": (
                        f"CRWD baseline rank by score: "
                        f"{int(_to_float(row.get('baseline_rank_by_score')))} of "
                        f"{int(_to_float(row.get('parameter_sets_tested')))}; "
                        f"nearby median return: "
                        f"{_to_float(row.get('nearby_median_total_return')):.4f}."
                    ),
                    "verdict": "Treat the optimized grid as a stability diagnostic, not permission to tune yet.",
                }
            )
    if not quasi_p_values.empty:
        crwd = quasi_p_values[quasi_p_values["symbol"].astype(str).str.upper() == "CRWD"]
        if not crwd.empty:
            best = crwd.sort_values("quasi_p_value", na_position="last").iloc[0]
            rows.append(
                {
                    "area": "Permutation",
                    "finding": (
                        f"Strongest CRWD metric: {best.get('metric')} "
                        f"with quasi p-value {_to_float(best.get('quasi_p_value')):.4f}."
                    ),
                    "verdict": "Permutation can flag non-random-looking behavior, but it is still not proof.",
                }
            )
    if not direction_mode_comparison.empty:
        crwd_modes = direction_mode_comparison[
            direction_mode_comparison["symbol"].astype(str).str.upper() == "CRWD"
        ]
        baseline = crwd_modes[crwd_modes["mode_label"] == "long_short_baseline"]
        short_only = crwd_modes[crwd_modes["mode_label"] == "short_only_fixed_bars_exit"]
        if not baseline.empty and not short_only.empty:
            rows.append(
                {
                    "area": "Long/short contribution",
                    "finding": (
                        f"Baseline short PnL: {_to_float(baseline.iloc[0].get('short_net_pnl')):.2f}; "
                        f"short-only total return: {_to_float(short_only.iloc[0].get('total_return')):.4f}."
                    ),
                    "verdict": "Shorts help parity and add some dollars in CRWD, but the main engine is long-side.",
                }
            )
    return pd.DataFrame(rows)


def build_final_validation_verdict_summary(
    baseline_summary: pd.DataFrame,
    strategy_vs_buy_hold: pd.DataFrame,
    permutation_1000_summary: pd.DataFrame,
    time_split_validation: pd.DataFrame,
    direction_mode_comparison: pd.DataFrame,
    sizing_stress_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build final-validation verdict rows."""
    rows: list[dict[str, Any]] = []
    crwd = _row_for_symbol(baseline_summary, "CRWD")
    crwd_bh = _row_for_symbol(strategy_vs_buy_hold, "CRWD")
    if crwd:
        rows.append(
            {
                "area": "CRWD baseline",
                "finding": (
                    f"Total return {_pct(crwd.get('total_return'))}; "
                    f"CAGR {_pct(crwd.get('cagr'))}; "
                    f"max drawdown {_pct(crwd.get('max_drawdown'))}; "
                    f"profit factor {_to_float(crwd.get('profit_factor')):.4f}; "
                    f"trades {int(_to_float(crwd.get('number_of_trades')))}."
                ),
                "classification": "incubation candidate",
                "note": (
                    f"Buy-and-hold return {_pct(crwd_bh.get('buy_hold_return'))}."
                    if crwd_bh
                    else ""
                ),
            }
        )
    rows.extend(
        [
            {
                "area": "Peer basket summary",
                "finding": "Not rerun in final_validation to keep this stage CRWD-only.",
                "classification": "incubation candidate",
                "note": "Review the prior peer basket stage artifact before escalating beyond incubation.",
            },
            {
                "area": "Cost stress summary",
                "finding": "Not rerun in final_validation.",
                "classification": "incubation candidate",
                "note": "Review the prior cost stress artifact before any paper-tracking decision.",
            },
            {
                "area": "Monte Carlo skip summary",
                "finding": "Not rerun in final_validation.",
                "classification": "incubation candidate",
                "note": "Review the prior skip-test artifact; skip robustness is a filter, not proof.",
            },
            {
                "area": "Trade sequence randomization",
                "finding": "Not rerun in final_validation.",
                "classification": "incubation candidate",
                "note": "Prior sequence testing showed path risk can remain severe even with strong returns.",
            },
            {
                "area": "Parameter stability summary",
                "finding": "No optimization was run in final_validation.",
                "classification": "incubation candidate",
                "note": "Stage 6A baseline-neighbor stability should be reviewed separately; no parameters were tuned here.",
            },
        ]
    )
    if not permutation_1000_summary.empty:
        pass_count = int((permutation_1000_summary["pass_fail_flag"] == "pass_interesting").sum())
        borderline_count = int((permutation_1000_summary["pass_fail_flag"] == "borderline").sum())
        fail_count = int((permutation_1000_summary["pass_fail_flag"] == "fail_weak").sum())
        rows.append(
            {
                "area": "CRWD 1000-permutation",
                "finding": (
                    f"{pass_count} pass, {borderline_count} borderline, "
                    f"{fail_count} fail across {len(permutation_1000_summary)} metrics."
                ),
                "classification": "incubation candidate",
                "note": "Permutation is a robustness filter, not proof.",
            }
        )
    if not time_split_validation.empty:
        ok = time_split_validation[time_split_validation.get("status", "") == "ok"]
        profitable = int((pd.to_numeric(ok.get("total_return"), errors="coerce") > 0).sum())
        beat_bh = int((pd.to_numeric(ok.get("excess_vs_buy_hold"), errors="coerce") > 0).sum())
        rows.append(
            {
                "area": "Time split",
                "finding": (
                    f"{profitable}/{len(ok)} splits profitable; "
                    f"{beat_bh}/{len(ok)} beat buy-and-hold."
                ),
                "classification": "watchlist only" if profitable < len(ok) else "incubation candidate",
                "note": "Use split consistency to decide whether the edge is regime-dependent.",
            }
        )
    if not direction_mode_comparison.empty:
        base = direction_mode_comparison[
            direction_mode_comparison["mode_label"].astype(str) == "long_short_baseline"
        ]
        short = direction_mode_comparison[
            direction_mode_comparison["mode_label"].astype(str) == "short_only"
        ]
        short_note = ""
        classification = "incubation candidate"
        if not base.empty and not short.empty:
            short_pnl = _to_float(base.iloc[0].get("short_net_pnl"))
            short_only_return = _to_float(short.iloc[0].get("total_return"))
            short_note = (
                f"Baseline short PnL {short_pnl:.2f}; "
                f"short-only return {_pct(short_only_return)}."
            )
            if short_pnl < 0:
                classification = "watchlist only"
        rows.append(
            {
                "area": "Direction mode",
                "finding": short_note or "Direction-mode comparison completed.",
                "classification": classification,
                "note": "Shorts should be treated as contribution diagnostics, not a reason to tune.",
            }
        )
    if not sizing_stress_summary.empty:
        dd_10 = sizing_stress_summary[
            pd.to_numeric(sizing_stress_summary["allocation_pct"], errors="coerce") == 10
        ]
        note = ""
        classification = "paper-tracking candidate"
        if not dd_10.empty:
            dd = _to_float(dd_10.iloc[0].get("max_drawdown"))
            note = f"10% allocation max drawdown {_pct(dd)}."
            if dd < -0.25:
                classification = "incubation candidate"
        rows.append(
            {
                "area": "Sizing stress",
                "finding": "Sizing stress completed at 100%, 50%, 25%, and 10% equity.",
                "classification": classification,
                "note": note,
            }
        )
    final_classification = _final_classification(rows)
    rows.append(
        {
            "area": "Final research verdict",
            "finding": (
                "Returns remain unusually strong, but drawdown/path risk is severe. "
                "Do not recommend full-account allocation."
            ),
            "classification": final_classification,
            "note": "Not a live-trading candidate.",
        }
    )
    return pd.DataFrame(rows)


def build_final_verdict_markdown(verdict_summary: pd.DataFrame) -> str:
    """Render final verdict rows as markdown."""
    lines = [
        "# Simple Strategy 01 Final Verdict",
        "",
        "Classification scale: reject, watchlist only, incubation candidate, paper-tracking candidate, live-trading candidate.",
        "",
        "This report does not classify the strategy as a live-trading candidate.",
        "",
    ]
    if verdict_summary.empty:
        lines.append("No verdict rows were generated.")
        return "\n".join(lines)
    for _, row in verdict_summary.iterrows():
        lines.extend(
            [
                f"## {row.get('area')}",
                f"- Classification: {row.get('classification')}",
                f"- Finding: {row.get('finding')}",
                f"- Note: {row.get('note')}",
                "",
            ]
        )
    return "\n".join(lines)


def resolve_optimization_parameter_grid(opt_config: dict[str, Any]) -> dict[str, list[Any]]:
    """Resolve the staged optimization grid from config."""
    mode = str(opt_config.get("optimization_mode", "baseline_neighbors")).lower()
    if mode == "baseline_neighbors":
        return baseline_neighbor_parameter_grid()
    if mode == "small_grid":
        return small_parameter_grid()
    if mode == "full_grid":
        return opt_config.get("parameter_grid", default_parameter_grid())
    raise ValueError(
        "optimization_mode must be one of: baseline_neighbors, small_grid, full_grid"
    )


def baseline_neighbor_parameter_grid() -> dict[str, list[Any]]:
    """Return the requested Stage 6A baseline-neighbor grid."""
    return {
        "entry.long_extreme_signal.length": [2, 3],
        "entry.short_extreme_signal.length": [8, 11, 15],
        "entry.volume_filter.length": [10],
        "filters.weekday_skip_wednesday.skip_days": [["Wednesday"], []],
        "entry.short_filters.0.range_mult": [0.8, 1.0, 1.2],
        "exits.fixed_bars.bars": [20, 25, 30],
        "filters.trend_filter.enabled": [False],
        "filters.trend_filter.length": [200],
        "exits.percent_stop_loss.enabled": [False],
        "exits.percent_stop_loss.stop_loss_pct": [25],
    }


def small_parameter_grid() -> dict[str, list[Any]]:
    """A compact grid broader than baseline_neighbors but still bounded."""
    return {
        "entry.long_extreme_signal.length": [2, 3, 4],
        "entry.short_extreme_signal.length": [8, 11, 15],
        "entry.volume_filter.length": [5, 10],
        "filters.weekday_skip_wednesday.skip_days": [["Wednesday"], []],
        "entry.short_filters.0.range_mult": [0.8, 1.0, 1.2],
        "exits.fixed_bars.bars": [20, 25, 30],
        "filters.trend_filter.enabled": [False],
        "filters.trend_filter.length": [200],
        "exits.percent_stop_loss.enabled": [False],
        "exits.percent_stop_loss.stop_loss_pct": [25],
    }


def default_parameter_grid() -> dict[str, list[Any]]:
    """Return the requested small parameter grid."""
    return {
        "entry.long_extreme_signal.length": [2, 3, 4, 5],
        "entry.short_extreme_signal.length": [8, 11, 15, 20],
        "entry.volume_filter.length": [5, 10, 20],
        "filters.weekday_skip_wednesday.skip_days": [["Wednesday"], []],
        "entry.short_filters.0.range_mult": [0.8, 1.0, 1.2],
        "exits.fixed_bars.bars": [15, 20, 25, 30, 40],
        "filters.trend_filter.enabled": [False, True],
        "filters.trend_filter.length": [100, 200],
        "exits.percent_stop_loss.enabled": [False, True],
        "exits.percent_stop_loss.stop_loss_pct": [20, 25, 30],
    }


def _iter_effective_parameter_grid(parameter_grid: dict[str, list[Any]]):
    """Yield grid points, collapsing duplicate disabled stop/trend variants."""
    seen: set[str] = set()
    for raw_parameters in iter_parameter_grid(parameter_grid):
        parameters = dict(raw_parameters)
        weekday_path = "filters.weekday_skip_wednesday.skip_days"
        if weekday_path in parameters:
            parameters["entry.long_filters.0.skip_days"] = parameters[weekday_path]
        if not _as_bool(parameters.get("exits.percent_stop_loss.enabled", False)):
            parameters["exits.percent_stop_loss.stop_loss_pct"] = 25
        if not _as_bool(parameters.get("filters.trend_filter.enabled", False)):
            parameters["filters.trend_filter.length"] = 200
        key = json.dumps(
            {name: _parameter_value_for_table(value) for name, value in parameters.items()},
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        yield parameters


def build_short_only_with_long_signal_exit_strategy(
    config: dict[str, Any],
    symbol: str,
) -> StrategyDefinition:
    """Build requested short-only comparison with raw long signal as exit."""
    params = dict(config.get("strategy_parameters", {}) or {})
    costs = dict(config.get("costs", {}) or {})
    sizing = dict(config.get("sizing", {}) or {})
    long_length = int(params.get("long_length", 2))
    short_length = int(params.get("short_length", 11))
    volume_avg_length = int(params.get("volume_avg_length", 10))
    avoid_wednesday_longs = bool(params.get("avoid_wednesday_longs", True))
    range_avg_length = int(params.get("range_avg_length", 20))
    wide_range_mult = float(params.get("wide_range_mult", 1.0))
    exit_bars = int(params.get("exit_bars", 25))
    stop_loss_enabled = bool(params.get("stop_loss_enabled", False))
    stop_loss_pct = float(params.get("stop_loss_pct", 25.0))
    trend_filter = bool(params.get("trend_filter", False))
    trend_length = int(params.get("trend_length", 200))
    market_regime_filter = bool(params.get("market_regime_filter", False))
    market_symbol = str(params.get("market_symbol", "QQQ"))
    market_trend_length = int(params.get("market_trend_length", 200))

    weekday_filter = WeekdayFilter(
        skip_days=["Wednesday"] if avoid_wednesday_longs else [],
        apply_to_long=True,
        apply_to_short=False,
        label="weekday_skip_wednesday",
    )
    wide_bar_filter = WideBullishBarFilter(
        range_length=range_avg_length,
        range_mult=wide_range_mult,
        block_when_true=True,
        enabled=bool(params.get("wide_bullish_filter", True)),
        apply_to_long=False,
        apply_to_short=True,
        label="wide_bullish_bar",
    )
    trend_module = TrendFilter(
        length=trend_length,
        mode="close_above_sma",
        enabled=trend_filter,
        label="trend_filter",
    )
    market_module = MarketRegimeFilter(
        benchmark_symbol=market_symbol,
        length=market_trend_length,
        mode="close_above_sma",
        enabled=market_regime_filter,
        label="market_regime_filter",
    )
    entry = VolumeFadeReversalEntry(
        long_extreme_signal=LowestCloseSignal(length=long_length),
        short_extreme_signal=HighestCloseSignal(length=short_length),
        volume_filter=VolumeAboveAverageFilter(length=volume_avg_length, multiplier=1.0),
        long_filters=[weekday_filter],
        short_filters=[wide_bar_filter],
        enable_long_entries=False,
        enable_short_entries=True,
    )
    long_exit_signal = VolumeStrengthSignal(
        direction="long",
        extreme_signal=LowestCloseSignal(length=long_length),
        volume_filter=VolumeAboveAverageFilter(length=volume_avg_length, multiplier=1.0),
        filters=[weekday_filter],
    )
    exits = [
        PercentStopLossExit(
            stop_loss_pct=stop_loss_pct,
            enabled=stop_loss_enabled,
            label="percent_stop_loss",
        ),
        OppositeStrengthExit(
            signal=long_exit_signal,
            reverse_on_opposite=False,
            label="opposite_long_strength",
        ),
    ]
    if bool(params.get("fixed_bars_exit", True)):
        exits.append(FixedBarsExit(bars=exit_bars, label="fixed_bars"))
    return StrategyDefinition(
        name="software_volume_fade_reversal",
        symbols=[symbol.upper()],
        entry=entry,
        exit_stack=ExitStack(exits=exits, label="volume_fade_exit_stack"),
        filters=[
            WeekdayFilter(
                skip_days=["Wednesday"] if avoid_wednesday_longs else [],
                apply_to_long=True,
                apply_to_short=False,
                label="weekday_skip_wednesday",
            ),
            trend_module,
            market_module,
        ],
        direction_mode="short_only",
        filter_mode="all",
        initial_capital=float(config.get("initial_capital", 100000.0)),
        sizing=PercentEquitySizing(percent=float(sizing.get("percent_equity", 100.0))),
        costs=StockCostModel(
            commission_percent=float(costs.get("commission_percent", 0.03)),
            slippage_bps_per_side=float(costs.get("slippage_bps_per_side", 0.0)),
        ),
        parameters=dict(params),
        min_bars=int(params.get("min_bars", 80)),
    )


def _finalize_symbol_optimization_results(
    symbol_results: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Sort, rank, and flag one symbol's optimization rows."""
    if symbol_results.empty:
        return symbol_results
    out = symbol_results.sort_values(
        "score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    out["rank_by_score"] = out.index + 1
    out = add_optimization_flags(out)
    return out


def _write_optimization_partial(
    path: Path,
    completed_symbol_results: list[pd.DataFrame],
    current_symbol_rows: pd.DataFrame,
    completed_configs: int,
    interrupted: bool,
) -> None:
    """Write checkpointed optimization rows that are complete so far."""
    frames = [frame for frame in completed_symbol_results if frame is not None and not frame.empty]
    if current_symbol_rows is not None and not current_symbol_rows.empty:
        frames.append(current_symbol_rows.copy())
    partial = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not partial.empty:
        partial["partial_completed_configs"] = completed_configs
        partial["partial_interrupted"] = interrupted
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.to_csv(path, index=False)


def _build_baseline_vs_optimized_row(
    symbol: str,
    baseline_summary: dict[str, Any],
    baseline_score: float,
    ranked_results: pd.DataFrame,
    buy_hold_return: float | None,
) -> dict[str, Any]:
    """Summarize baseline rank and nearby-parameter stability for one symbol."""
    baseline_matches = ranked_results[
        ranked_results["baseline_parameter_match"].astype(bool)
    ]
    nearby = ranked_results[ranked_results["near_baseline_parameter_set"].astype(bool)]
    top = ranked_results.iloc[0] if not ranked_results.empty else pd.Series(dtype=object)
    baseline_grid = (
        baseline_matches.sort_values("rank_by_score").iloc[0]
        if not baseline_matches.empty
        else pd.Series(dtype=object)
    )
    total_return = _to_float(baseline_summary.get("total_return"))
    row = {
        "symbol": symbol,
        "objective": ranked_results["objective"].iloc[0] if "objective" in ranked_results else None,
        "parameter_sets_tested": int(len(ranked_results)),
        "baseline_rank_by_score": (
            int(baseline_grid.get("rank_by_score"))
            if not baseline_grid.empty and pd.notna(baseline_grid.get("rank_by_score"))
            else None
        ),
        "baseline_score": baseline_score,
        "baseline_grid_score": _to_float(baseline_grid.get("score")),
        "baseline_total_return": total_return,
        "baseline_cagr": _to_float(baseline_summary.get("cagr")),
        "baseline_max_drawdown": _to_float(baseline_summary.get("max_drawdown")),
        "baseline_profit_factor": _to_float(baseline_summary.get("profit_factor")),
        "baseline_number_of_trades": _to_float(baseline_summary.get("number_of_trades")),
        "buy_hold_return": buy_hold_return,
        "baseline_excess_vs_buy_hold": (
            total_return - buy_hold_return if buy_hold_return is not None else None
        ),
        "top_rank_score": _to_float(top.get("score")),
        "top_rank_total_return": _to_float(top.get("total_return")),
        "top_rank_cagr": _to_float(top.get("cagr")),
        "top_rank_max_drawdown": _to_float(top.get("max_drawdown")),
        "top_rank_profit_factor": _to_float(top.get("profit_factor")),
        "top_rank_parameters": top.get("parameters"),
        "parameter_sets_profitable": int(ranked_results["profitable"].sum()),
        "parameter_sets_beating_buy_hold": int(ranked_results["beats_buy_hold"].sum()),
        "parameter_sets_max_drawdown_gt_70pct": int(
            ranked_results["max_drawdown_worse_than_70pct"].sum()
        ),
        "nearby_parameter_sets": int(len(nearby)),
        "nearby_median_score": _series_median(nearby, "score"),
        "nearby_median_total_return": _series_median(nearby, "total_return"),
        "nearby_p25_total_return": _series_quantile(nearby, "total_return", 0.25),
        "nearby_p75_total_return": _series_quantile(nearby, "total_return", 0.75),
        "nearby_pct_profitable": _series_mean(nearby, "profitable"),
        "nearby_pct_beating_buy_hold": _series_mean(nearby, "beats_buy_hold"),
    }
    return row


def _side_contribution_for_symbol(trade_log: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Return long/short PnL and exit reason counts for a symbol."""
    out = {
        "long_trade_count": 0,
        "short_trade_count": 0,
        "long_net_pnl": 0.0,
        "short_net_pnl": 0.0,
        "exit_reason_counts": "{}",
    }
    if trade_log.empty or "symbol" not in trade_log:
        return out
    trades = trade_log[trade_log["symbol"].astype(str).str.upper() == symbol.upper()]
    if trades.empty:
        return out
    long_trades = trades[trades["side"].astype(str).str.lower() == "long"]
    short_trades = trades[trades["side"].astype(str).str.lower() == "short"]
    out["long_trade_count"] = int(len(long_trades))
    out["short_trade_count"] = int(len(short_trades))
    out["long_net_pnl"] = (
        float(long_trades["net_pnl"].sum()) if not long_trades.empty else 0.0
    )
    out["short_net_pnl"] = (
        float(short_trades["net_pnl"].sum()) if not short_trades.empty else 0.0
    )
    if "exit_reason" in trades:
        out["exit_reason_counts"] = json.dumps(
            trades["exit_reason"].value_counts().to_dict(),
            sort_keys=True,
        )
    return out


def _is_baseline_parameter_set(parameters: dict[str, Any]) -> bool:
    """Return true when a grid point matches the frozen Pine baseline."""
    return (
        int(parameters.get("entry.long_extreme_signal.length", -1)) == 2
        and int(parameters.get("entry.short_extreme_signal.length", -1)) == 11
        and int(parameters.get("entry.volume_filter.length", -1)) == 10
        and _has_wednesday(parameters.get("filters.weekday_skip_wednesday.skip_days"))
        and abs(float(parameters.get("entry.short_filters.0.range_mult", 0.0)) - 1.0) < 1e-9
        and int(parameters.get("exits.fixed_bars.bars", -1)) == 25
        and not _as_bool(parameters.get("exits.percent_stop_loss.enabled", False))
        and not _as_bool(parameters.get("filters.trend_filter.enabled", False))
    )


def _is_near_baseline_parameter_set(parameters: dict[str, Any]) -> bool:
    """Broad local neighborhood around the frozen baseline settings."""
    return (
        int(parameters.get("entry.long_extreme_signal.length", -1)) in {2, 3}
        and int(parameters.get("entry.short_extreme_signal.length", -1)) in {8, 11, 15}
        and int(parameters.get("entry.volume_filter.length", -1)) in {5, 10, 20}
        and abs(float(parameters.get("entry.short_filters.0.range_mult", 0.0)) - 1.0) <= 0.21
        and int(parameters.get("exits.fixed_bars.bars", -1)) in {20, 25, 30}
    )


def _parameter_value_for_table(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _permutation_pass_fail(quasi_p_value: Any) -> str:
    value = _to_float(quasi_p_value)
    if value != value:
        return "not_available"
    if value <= 0.05:
        return "pass_interesting"
    if value <= 0.10:
        return "borderline"
    return "fail_weak"


def _has_wednesday(value: Any) -> bool:
    if isinstance(value, list):
        return any(str(item).lower() == "wednesday" for item in value)
    return "wednesday" in str(value).lower()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _pct(value: Any) -> str:
    number = _to_float(value)
    if number != number:
        return "N/A"
    return f"{number * 100:.2f}%"


def _row_for_symbol(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    if df.empty or "symbol" not in df:
        return {}
    rows = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _final_classification(rows: list[dict[str, Any]]) -> str:
    classes = {str(row.get("classification", "")) for row in rows}
    if "reject" in classes:
        return "reject"
    if "watchlist only" in classes:
        return "watchlist only"
    if "incubation candidate" in classes:
        return "incubation candidate"
    return "paper-tracking candidate"


def _front_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    front = [col for col in columns if col in df.columns]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]


def _normalized_prices(prices: pd.DataFrame | None) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    out = prices.copy()
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _series_median(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").median())


def _series_quantile(df: pd.DataFrame, column: str, quantile: float) -> float | None:
    if df.empty or column not in df:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").quantile(quantile))


def _series_mean(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _build_strategy(config: dict[str, Any], symbols: list[str]) -> StrategyDefinition:
    params = dict(config.get("strategy_parameters", {}) or {})
    costs = dict(config.get("costs", {}) or {})
    sizing = dict(config.get("sizing", {}) or {})
    return build_software_volume_fade_reversal_strategy(
        symbols=symbols,
        percent_equity=float(sizing.get("percent_equity", params.pop("percent_equity", 100.0))),
        commission_percent=float(costs.get("commission_percent", 0.03)),
        slippage_bps_per_side=float(costs.get("slippage_bps_per_side", 0.0)),
        initial_capital=float(config.get("initial_capital", 100000.0)),
        **params,
    )
