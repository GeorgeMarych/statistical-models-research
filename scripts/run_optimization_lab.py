"""Run a small parameter-grid optimization for one configured strategy.

Usage:
    python scripts/run_optimization_lab.py
    python scripts/run_optimization_lab.py config/optimization_lab.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.labs.optimization_lab import run_optimization_lab


def main() -> None:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else _ROOT / "config" / "optimization_lab.yaml"
    )
    result = run_optimization_lab(config_path)
    print("Optimization lab complete.")
    print(f"Strategy: {result.strategy.name}")
    print(f"Output directory: {result.output_dir}")
    for label, path in result.paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
