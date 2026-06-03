"""
Backtesting skeleton — not yet implemented.

This module will simulate trade entries and exits driven by the Markov signal.
It will be built after the signal validation step confirms predictive value.

Planned inputs:
    - Signal dataset (from features.build_signal_dataset)
    - Entry threshold (cfg.signal_threshold)
    - Optional: market regime filter (SPY/QQQ EMA)
    - Optional: ATR stop
    - Optional: EMA 10/20 exit
    - Optional: earnings exit

Planned outputs:
    - Trade log (entry date, exit date, return, exit reason)
    - Equity curve
    - Summary metrics (strategy return, B&H return, max drawdown, exposure)
"""
from __future__ import annotations

import pandas as pd


def run_backtest(df: pd.DataFrame, signal_threshold: float) -> dict:
    """Placeholder — will simulate long entries when signal >= signal_threshold."""
    raise NotImplementedError(
        "Backtester not yet implemented. "
        "Run run_signal_validation.py first to validate the signal."
    )
