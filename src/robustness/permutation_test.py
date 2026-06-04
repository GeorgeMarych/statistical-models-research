"""Permutation tests for frozen strategy definitions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.engine import prepare_ohlcv
from src.backtesting.strategy_runner import run_strategy_definition
from src.optimization.objective import score_summary
from src.strategies.strategy_definition import StrategyDefinition


def run_permutation_test(
    strategy: StrategyDefinition,
    symbol: str,
    price_data: dict[str, pd.DataFrame],
    num_runs: int = 200,
    random_seed: int = 0,
    objective_metrics: list[str] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 0,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare real strategy metrics with permuted OHLC paths."""
    objective_metrics = objective_metrics or [
        "profit_factor",
        "cagr",
        "return_over_drawdown",
        "final_equity",
    ]
    symbol = symbol.upper()
    if symbol not in price_data or price_data[symbol].empty:
        return pd.DataFrame(), pd.DataFrame()

    primary_strategy = deepcopy(strategy)
    primary_strategy.symbols = [symbol]
    real_run = run_strategy_definition(primary_strategy, price_data)
    if real_run.summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    real_summary = real_run.summary.iloc[0].to_dict()

    rows: list[dict] = []
    original = prepare_ohlcv(price_data[symbol])
    print(
        f"{symbol}: starting permutation test with {num_runs} permutations; "
        f"random_seed={random_seed}"
    )
    for run_number in range(1, num_runs + 1):
        seed = random_seed + run_number
        permuted = permute_ohlc_path(original, seed=seed)
        candidate_data = dict(price_data)
        candidate_data[symbol] = permuted
        perm_run = run_strategy_definition(primary_strategy, candidate_data)
        summary = perm_run.summary.iloc[0].to_dict() if not perm_run.summary.empty else {}
        row = {
            "symbol": symbol,
            "run_number": run_number,
            "seed": seed,
            "random_seed_base": random_seed,
            "permutations_requested": num_runs,
        }
        for metric in objective_metrics:
            row[metric] = _metric_value(summary, metric)
        rows.append(row)
        if progress_every and (run_number % progress_every == 0 or run_number == num_runs):
            print(f"{symbol}: completed {run_number}/{num_runs} permutations")
        if checkpoint_path and checkpoint_every and run_number % checkpoint_every == 0:
            _write_permutation_checkpoint(
                rows,
                checkpoint_path,
                completed_runs=run_number,
                complete=False,
            )
            print(f"{symbol}: wrote permutation checkpoint to {checkpoint_path}")

    results = pd.DataFrame(rows)
    if checkpoint_path:
        _write_permutation_checkpoint(
            rows,
            checkpoint_path,
            completed_runs=len(rows),
            complete=True,
        )
    summary_rows: list[dict] = []
    for metric in objective_metrics:
        real_value = _metric_value(real_summary, metric)
        perm_values = results[metric].replace([np.inf, -np.inf], np.nan).dropna()
        quasi_p = (
            float((perm_values >= real_value).mean())
            if len(perm_values) and pd.notna(real_value)
            else np.nan
        )
        summary_rows.append(
            {
                "symbol": symbol,
                "metric": metric,
                "real_value": real_value,
                "permutation_mean": float(perm_values.mean()) if len(perm_values) else np.nan,
                "permutation_median": float(perm_values.median()) if len(perm_values) else np.nan,
                "permutation_p95": float(perm_values.quantile(0.95)) if len(perm_values) else np.nan,
                "permutation_max": float(perm_values.max()) if len(perm_values) else np.nan,
                "quasi_p_value": quasi_p,
                "num_permutations": int(len(perm_values)),
                "random_seed_base": random_seed,
            }
        )
    return results, pd.DataFrame(summary_rows)


def permute_ohlc_path(data: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Generate a randomized OHLC path from permuted daily returns."""
    prices = prepare_ohlcv(data)
    if len(prices) < 3:
        return prices.copy()

    rng = np.random.default_rng(seed)
    returns = prices["close"].pct_change().iloc[1:].to_numpy()
    permuted_returns = rng.permutation(returns)
    close = [float(prices["close"].iloc[0])]
    for ret in permuted_returns:
        close.append(close[-1] * (1.0 + float(ret)))
    close_series = pd.Series(close, index=prices.index)

    ratios = pd.DataFrame(
        {
            "open_ratio": prices["open"] / prices["close"],
            "high_ratio": prices["high"] / prices["close"],
            "low_ratio": prices["low"] / prices["close"],
            "volume": prices["volume"],
        },
        index=prices.index,
    )
    shuffled_ratios = ratios.iloc[rng.permutation(len(ratios))].set_index(prices.index)

    out = pd.DataFrame(index=prices.index)
    out["close"] = close_series
    out["open"] = close_series * shuffled_ratios["open_ratio"].fillna(1.0)
    out["high"] = close_series * shuffled_ratios["high_ratio"].fillna(1.0)
    out["low"] = close_series * shuffled_ratios["low_ratio"].fillna(1.0)
    out["high"] = pd.concat([out["high"], out["open"], out["close"]], axis=1).max(axis=1)
    out["low"] = pd.concat([out["low"], out["open"], out["close"]], axis=1).min(axis=1)
    out["volume"] = shuffled_ratios["volume"].fillna(prices["volume"].median()).to_numpy()
    return out[["open", "high", "low", "close", "volume"]]


def _metric_value(summary: dict, metric: str) -> float:
    if metric in {"return_over_max_drawdown", "return_over_drawdown"}:
        return score_summary(summary, "return_over_drawdown")
    value = summary.get(metric, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _write_permutation_checkpoint(
    rows: list[dict],
    checkpoint_path: str | Path,
    completed_runs: int,
    complete: bool,
) -> None:
    """Write completed permutation rows so interrupted runs keep useful output."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["checkpoint_completed_runs"] = completed_runs
        out["checkpoint_complete"] = complete
    out.to_csv(path, index=False)
