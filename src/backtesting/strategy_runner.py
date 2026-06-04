"""Run complete StrategyDefinition objects."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.backtesting.engine import BacktestEngine, BacktestSettings
from src.strategies.strategy_definition import StrategyDefinition


@dataclass
class StrategyRunResult:
    """Outputs for one complete strategy run across symbols."""

    strategy: StrategyDefinition
    summary: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame


def run_strategy_definition(
    strategy: StrategyDefinition,
    price_data: dict[str, pd.DataFrame],
) -> StrategyRunResult:
    """Run a strategy definition on every configured symbol with available data."""
    summary_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    filter_context = {"market_data": price_data}

    for symbol in strategy.symbols:
        prices = price_data.get(symbol.upper())
        if prices is None or prices.empty or len(prices) < strategy.min_bars:
            continue

        settings = BacktestSettings(
            symbol=symbol,
            initial_capital=strategy.initial_capital,
            allow_short=strategy.allow_short,
            sizing=strategy.sizing,
            costs=strategy.costs,
            periods_per_year=strategy.periods_per_year,
        )
        filtered_entry = strategy.filtered_entry(filter_context)
        result = BacktestEngine(settings).run(
            prices,
            filtered_entry,
            strategy.exit_stack,
        )

        run_id = f"{strategy.name}__{symbol}".lower()
        row = {
            "run_id": run_id,
            "strategy": strategy.name,
            "symbol": symbol,
            "entry": strategy.entry.name,
            "exit_stack": strategy.exit_stack.name,
            "filters": ",".join(filter_module.name for filter_module in strategy.filters),
            "direction_mode": strategy.direction_mode,
            "entry_parameters": json.dumps(strategy.entry.parameters, sort_keys=True),
            "exit_stack_parameters": json.dumps(strategy.exit_stack.parameters, sort_keys=True),
            "filter_parameters": json.dumps(
                {
                    filter_module.name: filter_module.parameters
                    for filter_module in strategy.filters
                },
                sort_keys=True,
            ),
        }
        row.update(result.summary)
        summary_rows.append(row)

        trades = result.trades.copy()
        if not trades.empty:
            trades.insert(0, "run_id", run_id)
            trades.insert(1, "strategy", strategy.name)
            trades.insert(4, "entry", strategy.entry.name)
            trades.insert(5, "exit_stack", strategy.exit_stack.name)
            trade_frames.append(trades)

        equity = result.equity_curve.copy()
        if not equity.empty:
            equity.insert(0, "run_id", run_id)
            equity.insert(1, "strategy", strategy.name)
            equity.insert(4, "entry", strategy.entry.name)
            equity.insert(5, "exit_stack", strategy.exit_stack.name)
            equity_frames.append(equity)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = _order_columns(
            summary,
            [
                "run_id",
                "strategy",
                "symbol",
                "entry",
                "exit_stack",
                "filters",
                "direction_mode",
                "total_return",
                "cagr",
                "max_drawdown",
                "sharpe",
                "profit_factor",
                "win_rate",
                "number_of_trades",
                "exposure_time",
                "average_bars_held",
                "best_trade",
                "worst_trade",
                "final_equity",
                "initial_capital",
            ],
        )
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    return StrategyRunResult(strategy, summary, trades, equity)


def collect_required_symbols(strategy: StrategyDefinition) -> list[str]:
    """Return strategy symbols plus any benchmark symbols used by filters."""
    symbols = list(strategy.symbols)
    for filter_module in strategy.filters:
        benchmark = getattr(filter_module, "benchmark_symbol", None)
        if benchmark:
            symbols.append(str(benchmark).upper())
    seen: set[str] = set()
    out: list[str] = []
    for symbol in symbols:
        symbol = str(symbol).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def aggregate_strategy_summary(summary: pd.DataFrame) -> dict:
    """Aggregate per-symbol rows into one compact optimization row."""
    if summary.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "expectancy_pct": 0.0,
            "number_of_trades": 0,
            "final_equity": 0.0,
            "symbols_tested": 0,
            "symbols_profitable": 0,
        }
    pf = summary["profit_factor"].replace([float("inf")], 10.0)
    return {
        "total_return": float(summary["total_return"].mean()),
        "cagr": float(summary["cagr"].mean()),
        "max_drawdown": float(summary["max_drawdown"].min()),
        "sharpe": float(summary["sharpe"].mean()),
        "profit_factor": float(pf.mean()),
        "win_rate": float(summary["win_rate"].mean()) if "win_rate" in summary else 0.0,
        "expectancy_pct": (
            float(summary["expectancy_pct"].mean())
            if "expectancy_pct" in summary
            else 0.0
        ),
        "number_of_trades": int(summary["number_of_trades"].sum()),
        "final_equity": float(summary["final_equity"].mean()),
        "symbols_tested": int(summary["symbol"].nunique()),
        "symbols_profitable": int((summary["total_return"] > 0).sum()),
    }


def _order_columns(df: pd.DataFrame, front_cols: list[str]) -> pd.DataFrame:
    front = [col for col in front_cols if col in df.columns]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]
