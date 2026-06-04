"""Reusable objective functions for strategy optimization."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def score_summary(summary: dict, objective: str = "balanced") -> float:
    """Score an aggregate summary row."""
    objective = str(objective).lower()
    if objective == "net_profit":
        return _clean(summary.get("total_return"))
    if objective == "cagr":
        return _clean(summary.get("cagr"))
    if objective == "profit_factor":
        return min(_clean(summary.get("profit_factor")), 10.0)
    if objective == "sharpe":
        return _clean(summary.get("sharpe"))
    if objective in {"return_over_drawdown", "return_max_drawdown"}:
        dd = abs(_clean(summary.get("max_drawdown")))
        return _clean(summary.get("total_return")) / dd if dd > 0 else 0.0
    if objective == "balanced":
        return balanced_score(summary)
    raise ValueError(f"Unsupported objective: {objective}")


def balanced_score(summary: dict) -> float:
    """
    Default objective.

    This rewards CAGR and risk-adjusted behavior, penalizes drawdown, gives a
    capped profit-factor bonus, and requires a modest number of trades before a
    result gets much credit.
    """
    cagr = _clean(summary.get("cagr"))
    max_drawdown = abs(_clean(summary.get("max_drawdown")))
    sharpe = _clean(summary.get("sharpe"))
    profit_factor = min(max(_clean(summary.get("profit_factor")), 0.0), 5.0)
    trades = max(int(summary.get("number_of_trades", 0) or 0), 0)
    symbols_tested = max(int(summary.get("symbols_tested", 0) or 0), 1)
    symbols_profitable = max(int(summary.get("symbols_profitable", 0) or 0), 0)

    trade_sanity = min(math.log1p(trades) / math.log1p(40), 1.0)
    symbol_breadth = symbols_profitable / symbols_tested

    return float(
        cagr
        - 0.75 * max_drawdown
        + 0.10 * sharpe
        + 0.04 * math.log1p(profit_factor)
        + 0.05 * trade_sanity
        + 0.05 * symbol_breadth
    )


def _clean(value) -> float:
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(out) or np.isinf(out):
        return 0.0
    return out
