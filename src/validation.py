"""Statistical validation utilities for the Markov signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def signal_return_correlation(
    df: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "future_5d_return",
) -> dict[str, float]:
    """
    Pearson and Spearman correlation between signal and a forward return.

    Returns a dict with keys: pearson_r, pearson_p, spearman_r, spearman_p, n.
    All values are NaN if fewer than 10 valid rows are available.
    """
    valid = df[[signal_col, return_col]].dropna()
    if len(valid) < 10:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "n": len(valid),
        }

    pearson_r, pearson_p = stats.pearsonr(valid[signal_col], valid[return_col])
    spearman_r, spearman_p = stats.spearmanr(valid[signal_col], valid[return_col])

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": len(valid),
    }


def signal_quantile_returns(
    df: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "future_5d_return",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Split signal into quantiles and compute mean forward return per quantile.

    If the signal has real predictive value, the top quantile should have
    the highest mean return and the bottom quantile the lowest.

    Returns an empty DataFrame when there is insufficient data.
    """
    valid = df[[signal_col, return_col]].dropna()
    if len(valid) < n_quantiles * 10:
        return pd.DataFrame()

    valid = valid.copy()
    valid["quantile"] = pd.qcut(valid[signal_col], n_quantiles, labels=False)
    result = valid.groupby("quantile")[return_col].agg(["mean", "std", "count"])
    result.index.name = "signal_quantile"
    return result


def state_return_summary(
    df: pd.DataFrame,
    return_col: str = "future_5d_return",
) -> pd.DataFrame:
    """
    Mean forward return grouped by current regime state.

    Useful sanity check: Bull state should have higher mean returns than Bear.
    """
    state_labels = {1: "Bull", 0: "Sideways", -1: "Bear"}
    result = (
        df.groupby("state")[return_col]
        .agg(["mean", "std", "count"])
        .rename(index=state_labels)
    )
    result.index.name = "state"
    return result
