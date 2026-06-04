"""Trade sequence randomization robustness tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_trade_sequence_randomization(
    trades: pd.DataFrame,
    num_runs: int = 1000,
    random_seed: int = 0,
    initial_capital: float = 100000.0,
) -> pd.DataFrame:
    """Shuffle completed trade returns and measure path-risk distributions."""
    if trades.empty or "net_return" not in trades.columns:
        return pd.DataFrame()

    returns = trades["net_return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    values = returns.to_numpy()
    for run_number in range(1, num_runs + 1):
        shuffled = rng.permutation(values)
        equity = initial_capital * pd.Series(1.0 + shuffled).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        losing = shuffled <= 0
        rows.append(
            {
                "run_number": run_number,
                "final_equity": float(equity.iloc[-1]),
                "total_return": float(equity.iloc[-1] / initial_capital - 1.0),
                "max_drawdown": float(drawdown.min()),
                "longest_losing_streak": _longest_streak(losing),
                "number_of_trades": int(len(shuffled)),
            }
        )
    return pd.DataFrame(rows)


def summarize_trade_sequence_randomization(results: pd.DataFrame) -> pd.DataFrame:
    """Return compact distribution statistics."""
    if results.empty:
        return pd.DataFrame()
    metrics = ["final_equity", "total_return", "max_drawdown", "longest_losing_streak"]
    rows: list[dict] = []
    for metric in metrics:
        values = results[metric].astype(float).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _longest_streak(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)
