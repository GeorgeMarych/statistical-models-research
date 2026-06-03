"""
Legacy simple visible Markov model.

This module contains the project's first experiment: a visible Markov chain over
simple bull/sideways/bear price-action states. It is preserved for existing
validation reports, but it should be treated as legacy research rather than the
center of the project or a standalone strategy.

State encoding:
    Bull      =  1
    Sideways  =  0
    Bear      = -1

Transition matrix layout (row = from-state, col = to-state):
    index 0 = Bull, index 1 = Sideways, index 2 = Bear

This matches the PineScript implementation exactly:
    for i = 1 to trainingWindow:
        fromState = marketState[i]      # i bars ago
        toState   = marketState[i - 1]  # i-1 bars ago (one step later)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Map state value → matrix row/column index
_STATE_TO_IDX: dict[int, int] = {1: 0, 0: 1, -1: 2}


def classify_states(
    close: pd.Series,
    lookback: int,
    bull_threshold: float,
    bear_threshold: float,
) -> pd.Series:
    """
    Classify each bar as Bull (1), Sideways (0), or Bear (-1).

    rolling_return = close[t] / close[t - lookback] - 1

    NaN is returned for the first `lookback` bars (insufficient history).
    """
    rolling_return = close / close.shift(lookback) - 1.0

    states = pd.Series(np.nan, index=close.index, dtype=float, name="state")
    valid = rolling_return.notna()

    states[valid & (rolling_return >= bull_threshold)] = 1.0
    states[valid & (rolling_return <= -bear_threshold)] = -1.0
    states[valid & (rolling_return > -bear_threshold) & (rolling_return < bull_threshold)] = 0.0

    return states


def compute_rolling_markov(
    states: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Compute rolling Markov transition probabilities for each bar.

    For bar t, count every consecutive state pair (states[t-i], states[t-i+1])
    for i = 1 … window. This sliding window is updated incrementally: one
    transition is added and one is removed on each bar, keeping O(n) total cost.

    A signal is only produced once we have a full window of transitions.

    Returns a DataFrame (same index as `states`) with columns:
        p_bull       — P(next = Bull  | current state)
        p_bear       — P(next = Bear  | current state)
        signal       — p_bull - p_bear
        sample_count — transitions available for the current state's matrix row
    """
    state_values = states.to_numpy(dtype=float)
    n = len(state_values)

    p_bull = np.full(n, np.nan)
    p_bear = np.full(n, np.nan)
    signal = np.full(n, np.nan)
    sample_count = np.zeros(n, dtype=int)

    # counts[from_idx, to_idx] — incremental 3×3 transition matrix
    counts = np.zeros((3, 3), dtype=int)

    for t in range(1, n):
        # Add the transition that just entered the window: states[t-1] → states[t]
        sv_prev = state_values[t - 1]
        sv_curr = state_values[t]
        if not np.isnan(sv_prev) and not np.isnan(sv_curr):
            fi = _STATE_TO_IDX.get(int(sv_prev))
            ti = _STATE_TO_IDX.get(int(sv_curr))
            if fi is not None and ti is not None:
                counts[fi, ti] += 1

        # Remove the transition that just left the window
        if t > window:
            sv_old_prev = state_values[t - window - 1]
            sv_old_curr = state_values[t - window]
            if not np.isnan(sv_old_prev) and not np.isnan(sv_old_curr):
                fi_old = _STATE_TO_IDX.get(int(sv_old_prev))
                ti_old = _STATE_TO_IDX.get(int(sv_old_curr))
                if fi_old is not None and ti_old is not None:
                    counts[fi_old, ti_old] -= 1

        # Warm-up: don't produce a signal until we have a full window
        if t < window:
            continue

        sv_t = state_values[t]
        if np.isnan(sv_t):
            continue

        ci = _STATE_TO_IDX.get(int(sv_t))
        if ci is None:
            continue

        row_total = int(counts[ci].sum())
        sample_count[t] = row_total

        if row_total > 0:
            p_bull[t] = counts[ci, 0] / row_total
            p_bear[t] = counts[ci, 2] / row_total
            signal[t] = p_bull[t] - p_bear[t]

    return pd.DataFrame(
        {
            "p_bull": p_bull,
            "p_bear": p_bear,
            "signal": signal,
            "sample_count": sample_count,
        },
        index=states.index,
    )
