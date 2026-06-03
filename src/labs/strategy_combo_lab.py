"""Strategy entry/exit combination lab."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtesting.combinator import (
    build_entry_modules,
    build_exit_modules,
    iter_entry_exit_combinations,
    safe_run_id,
)
from src.backtesting.costs import TradingCosts
from src.backtesting.engine import BacktestEngine, BacktestSettings, prepare_ohlcv
from src.backtesting.portfolio import PositionSizing
from src.data_loader import download_universe
from src.reports.strategy_report import write_strategy_combo_outputs


@dataclass
class StrategyComboLabResult:
    """Outputs returned by a strategy combo lab run."""

    output_dir: Path
    paths: dict[str, Path]
    summary: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame


def load_strategy_combo_config(path: str | Path) -> dict[str, Any]:
    """Load a strategy combo lab YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw


def run_strategy_combo_lab(config_path: str | Path) -> StrategyComboLabResult:
    """Run every configured symbol, entry, and exit combination."""
    config = load_strategy_combo_config(config_path)
    symbols = _normalize_symbols(config.get("symbols", []))
    if not symbols:
        raise ValueError("strategy combo config must include at least one symbol")

    entries = build_entry_modules(config.get("entries", {}))
    exits = build_exit_modules(config.get("exits", {}))
    if not entries:
        raise ValueError("strategy combo config must enable at least one entry")
    if not exits:
        raise ValueError("strategy combo config must enable at least one exit")

    data = _load_symbol_data(config, symbols)
    if not data:
        raise ValueError("no symbol data available for strategy combo lab")

    initial_capital = float(config.get("initial_capital", 100000.0))
    mode = str(config.get("mode", "long_only")).lower().replace("-", "_")
    allow_short = mode in {"long_short", "long/short", "longshort"}
    sizing = PositionSizing(**(config.get("sizing", {}) or {}))
    costs = TradingCosts(**(config.get("costs", {}) or {}))
    periods_per_year = int(config.get("periods_per_year", 252))
    min_bars = int(config.get("min_bars", 60))

    summary_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        if symbol not in data:
            continue
        prices = data[symbol]
        if len(prices) < min_bars:
            continue

        for entry, exit_module in iter_entry_exit_combinations(entries, exits):
            run_id = safe_run_id(symbol, entry.name, exit_module.name)
            settings = BacktestSettings(
                symbol=symbol,
                initial_capital=initial_capital,
                allow_short=allow_short,
                sizing=sizing,
                costs=costs,
                periods_per_year=periods_per_year,
            )
            result = BacktestEngine(settings).run(prices, entry, exit_module)

            summary = {
                "run_id": run_id,
                "symbol": symbol,
                "entry": entry.name,
                "exit": exit_module.name,
                "entry_parameters": json.dumps(entry.parameters, sort_keys=True),
                "exit_parameters": json.dumps(exit_module.parameters, sort_keys=True),
                "mode": "long_short" if allow_short else "long_only",
            }
            summary.update(result.summary)
            summary_rows.append(summary)

            trades = result.trades.copy()
            if not trades.empty:
                trades.insert(0, "run_id", run_id)
                trades.insert(2, "entry", entry.name)
                trades.insert(3, "exit", exit_module.name)
                trade_frames.append(trades)

            equity = result.equity_curve.copy()
            if not equity.empty:
                equity.insert(0, "run_id", run_id)
                equity.insert(2, "entry", entry.name)
                equity.insert(3, "exit", exit_module.name)
                equity_frames.append(equity)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("no backtest runs completed")

    front_cols = [
        "run_id",
        "symbol",
        "entry",
        "exit",
        "mode",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "win_rate",
        "expectancy_pct",
        "expectancy_dollars",
        "number_of_trades",
        "exposure_time",
        "average_bars_held",
        "best_trade",
        "worst_trade",
        "final_equity",
        "initial_capital",
    ]
    summary_df = _order_columns(summary_df, front_cols)
    summary_df = summary_df.sort_values(
        ["total_return", "sharpe"],
        ascending=[False, False],
        na_position="last",
    )

    trades_df = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame()
    )
    equity_df = (
        pd.concat(equity_frames, ignore_index=True)
        if equity_frames
        else pd.DataFrame()
    )

    output_dir = Path(config.get("output_path", "data/results/current/strategy_combo_lab"))
    paths = write_strategy_combo_outputs(summary_df, trades_df, equity_df, output_dir)
    return StrategyComboLabResult(output_dir, paths, summary_df, trades_df, equity_df)


def _normalize_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _load_symbol_data(config: dict[str, Any], symbols: list[str]) -> dict[str, pd.DataFrame]:
    data_config = config.get("data", {}) or {}
    source = str(data_config.get("source", "cache_or_yfinance")).lower()
    cache_dir = Path(data_config.get("cache_dir", "data/raw"))
    interval = str(config.get("interval", data_config.get("interval", "1d")))
    start = config.get("start_date", data_config.get("start_date", "2010-01-01"))
    end = config.get("end_date", data_config.get("end_date"))

    if source == "cache_only":
        raw_data = _read_cached_parquets(symbols, cache_dir)
    elif source in {"cache_or_yfinance", "yfinance"}:
        raw_data = download_universe(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError("data.source must be cache_only, cache_or_yfinance, or yfinance")

    out: dict[str, pd.DataFrame] = {}
    for symbol, raw in raw_data.items():
        if raw is None or raw.empty:
            continue
        prices = prepare_ohlcv(raw)
        if start:
            prices = prices[prices.index >= pd.Timestamp(start)]
        if end:
            prices = prices[prices.index <= pd.Timestamp(end)]
        if not prices.empty:
            out[symbol.upper()] = prices
    return out


def _read_cached_parquets(symbols: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path)
    return out


def _order_columns(df: pd.DataFrame, front_cols: list[str]) -> pd.DataFrame:
    front = [col for col in front_cols if col in df.columns]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]
