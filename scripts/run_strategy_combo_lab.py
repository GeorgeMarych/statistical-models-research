"""Run the modular strategy entry/exit combination lab.

Usage:
    python scripts/run_strategy_combo_lab.py
    python scripts/run_strategy_combo_lab.py config/strategy_combo_lab.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.labs.strategy_combo_lab import run_strategy_combo_lab


def main() -> None:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else _ROOT / "config" / "strategy_combo_lab.yaml"
    )
    result = run_strategy_combo_lab(config_path)
    print("Strategy combo lab complete.")
    print(f"Output directory: {result.output_dir}")
    for label, path in result.paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
