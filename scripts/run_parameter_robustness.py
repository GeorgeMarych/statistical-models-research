"""
Parameter robustness sweep — not yet implemented.

This script will grid-search over key parameters (state_lookback,
bull/bear thresholds, training_window) and report how stable the
signal's predictive value is across the parameter space.

Prerequisite:
    Run run_signal_validation.py first with the baseline config.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    print("Parameter robustness sweep not yet implemented.")
    print("Run scripts/run_signal_validation.py first.")
