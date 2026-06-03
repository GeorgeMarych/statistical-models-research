"""
Backtest runner — not yet implemented.

This script will simulate long entries/exits based on the Markov signal
once signal validation has confirmed predictive value.

Prerequisite:
    Run run_signal_validation.py first to generate
    data/results/legacy/markov_signal_dataset.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtester import run_backtest  # noqa: F401

if __name__ == "__main__":
    print("Backtester not yet implemented.")
    print("Run scripts/run_signal_validation.py first.")
