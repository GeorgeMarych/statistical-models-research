"""Run Simple Strategy #1 research lab.

Usage:
    python scripts/run_simple_strategy_01_lab.py
    python scripts/run_simple_strategy_01_lab.py config/simple_strategy_01.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.labs.simple_strategy_01_lab import (
    load_simple_strategy_01_config,
    run_simple_strategy_01_lab,
)
from src.labs.simple_strategy_01_outputs import clean_simple_strategy_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Simple Strategy 01 research stages.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(_ROOT / "config" / "simple_strategy_01.yaml"),
    )
    parser.add_argument(
        "--stage",
        default=None,
        choices=[
            "stage_1_pine_parity",
            "stage_2_peer_baseline",
            "stage_3_cost_stress",
            "stage_4_monte_carlo",
            "stage_5_trade_sequence",
            "stage_2_to_5",
            "stage_6a_baseline_neighbors",
            "stage_6_optimization",
            "stage_7_permutation",
            "stage_6_7",
            "stage_8_combined",
            "final_validation",
            "permutation_1000",
        ],
        help="Override the config stage profile.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Archive old root-level outputs and exit without running research.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if args.clean_output:
        config = load_simple_strategy_01_config(config_path)
        output_path = Path(config.get("output_path", _ROOT / "data" / "results" / "current" / "simple_strategy_01"))
        layout = clean_simple_strategy_output(output_path)
        print("Simple Strategy 01 output cleanup complete.")
        print(f"Base output directory: {layout.base_dir}")
        print(f"Latest directory: {layout.latest_dir}")
        print(f"Runs directory: {layout.base_dir / 'runs'}")
        print(f"Archive directory: {layout.archive_dir}")
        print(f"Archived files: {len(layout.moved_files)}")
        for source, target in layout.moved_files:
            print(f"  file: {source} -> {target}")
        print(f"Archived aborted run folders: {len(layout.moved_dirs)}")
        for source, target in layout.moved_dirs:
            print(f"  folder: {source} -> {target}")
        print(f"Seeded latest artifacts: {len(layout.latest_files)}")
        for source, target in layout.latest_files:
            print(f"  latest: {source} -> {target}")
        return

    result = run_simple_strategy_01_lab(config_path, stage_override=args.stage)
    print("Simple Strategy 01 lab complete.")
    print("Stage:", result.stage)
    print(f"Strategy: {result.strategy.name}")
    print(f"Output directory: {result.output_dir}")
    _print_crwd_metrics(result)
    print("NOTE: This Stage 1 run uses the corrected Pine reference settings: long + short enabled.")
    print("NOTE: Compare these numbers against the PineScript result before trusting further tests.")
    if result.missing_symbols:
        print(f"Missing symbols: {', '.join(result.missing_symbols)}")
    for label, path in result.paths.items():
        print(f"{label}: {path}")


def _print_crwd_metrics(result) -> None:
    summary = result.summary_by_symbol
    if summary.empty or "symbol" not in summary:
        print("CRWD metrics: unavailable (no summary rows).")
        return
    crwd_rows = summary[summary["symbol"].astype(str).str.upper() == "CRWD"]
    if crwd_rows.empty:
        print("CRWD metrics: unavailable (CRWD was not run).")
        return
    row = crwd_rows.iloc[0]
    buy_hold = result.strategy_vs_buy_hold
    bh_return = None
    if not buy_hold.empty and "symbol" in buy_hold:
        bh_rows = buy_hold[buy_hold["symbol"].astype(str).str.upper() == "CRWD"]
        if not bh_rows.empty:
            bh_return = bh_rows.iloc[0].get("buy_hold_return")

    print("CRWD Stage 1 metrics:")
    for label, column in [
        ("total_return", "total_return"),
        ("CAGR", "cagr"),
        ("max_drawdown", "max_drawdown"),
        ("profit_factor", "profit_factor"),
        ("win_rate", "win_rate"),
        ("number_of_trades", "number_of_trades"),
        ("final_equity", "final_equity"),
    ]:
        print(f"  {label}: {_fmt(row.get(column))}")
    print(f"  buy_hold_return: {_fmt(bh_return)}")
    trades = result.trade_log
    if trades.empty:
        print("  long_trade_count: 0")
        print("  short_trade_count: 0")
        print("  long_net_pnl: 0")
        print("  short_net_pnl: 0")
        print("  exit_reason_counts: none")
        return

    crwd_trades = trades[trades["symbol"].astype(str).str.upper() == "CRWD"].copy()
    long_trades = crwd_trades[crwd_trades["side"].astype(str).str.lower() == "long"]
    short_trades = crwd_trades[crwd_trades["side"].astype(str).str.lower() == "short"]
    print(f"  long_trade_count: {len(long_trades)}")
    print(f"  short_trade_count: {len(short_trades)}")
    print(f"  long_net_pnl: {_fmt(long_trades['net_pnl'].sum() if not long_trades.empty else 0.0)}")
    print(f"  short_net_pnl: {_fmt(short_trades['net_pnl'].sum() if not short_trades.empty else 0.0)}")
    if "exit_reason" in crwd_trades:
        print("  exit_reason_counts:")
        counts = crwd_trades["exit_reason"].value_counts()
        for reason, count in counts.items():
            print(f"    {reason}: {int(count)}")


def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if column_is_nan(number):
        return "N/A"
    return f"{number:.6g}"


def column_is_nan(value: float) -> bool:
    return value != value


if __name__ == "__main__":
    main()
