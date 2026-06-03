"""Performance and distribution metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate(returns: pd.Series) -> float:
    """Fraction of returns that are strictly positive."""
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    return float((clean > 0).mean())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound-annualize a series of simple (not log) period returns."""
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    compound = (1.0 + clean).prod()
    return float(compound ** (periods_per_year / len(clean)) - 1.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (assumes risk-free rate = 0)."""
    clean = returns.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return np.nan
    return float(clean.mean() / clean.std() * np.sqrt(periods_per_year))
