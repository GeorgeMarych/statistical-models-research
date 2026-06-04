"""Run one configured modular strategy.

Usage:
    python scripts/run_single_strategy_lab.py
    python scripts/run_single_strategy_lab.py config/single_strategy_lab.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.labs.single_strategy_lab import run_single_strategy_lab


def main() -> None:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else _ROOT / "config" / "single_strategy_lab.yaml"
    )
    result = run_single_strategy_lab(config_path)
    print("Single strategy lab complete.")
    print(f"Strategy: {result.strategy.name}")
    print(f"Output directory: {result.output_dir}")
    for label, path in result.paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
