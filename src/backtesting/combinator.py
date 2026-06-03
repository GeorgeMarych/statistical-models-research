"""Factories and utilities for entry/exit combination experiments."""
from __future__ import annotations

from itertools import product
from typing import Iterable

from src.entries import (
    DonchianBreakoutEntry,
    MovingAverageCrossoverEntry,
    RsiBollingerEntry,
)
from src.exits import (
    ATRStopExit,
    FixedBarsExit,
    MiddleBollingerExit,
    OppositeSignalExit,
    TakeProfitStopLossExit,
    UpperBollingerExit,
)

ENTRY_REGISTRY = {
    "rsi_bb_mean_reversion": RsiBollingerEntry,
    "rsi_bb": RsiBollingerEntry,
    "donchian_breakout": DonchianBreakoutEntry,
    "ma_crossover": MovingAverageCrossoverEntry,
    "moving_average_crossover": MovingAverageCrossoverEntry,
}

EXIT_REGISTRY = {
    "fixed_bars": FixedBarsExit,
    "opposite_signal": OppositeSignalExit,
    "atr_stop": ATRStopExit,
    "middle_bollinger": MiddleBollingerExit,
    "bollinger_middle": MiddleBollingerExit,
    "upper_bollinger": UpperBollingerExit,
    "bollinger_upper": UpperBollingerExit,
    "take_profit_stop_loss": TakeProfitStopLossExit,
    "tp_sl": TakeProfitStopLossExit,
}


def build_modules(section: dict, registry: dict, kind: str) -> list:
    """
    Instantiate enabled modules from a YAML section.

    Each item can use the config key as the module name, or provide an explicit
    `module` value when defining multiple variants of the same implementation.
    """
    modules = []
    for config_name, spec in (section or {}).items():
        spec = spec or {}
        if not spec.get("enabled", True):
            continue
        module_name = str(spec.get("module", config_name)).strip()
        if module_name not in registry:
            known = ", ".join(sorted(registry))
            raise ValueError(f"Unknown {kind} module {module_name!r}. Known: {known}")
        parameters = dict(spec.get("parameters", {}) or {})
        parameters.setdefault("label", str(config_name))
        modules.append(registry[module_name](**parameters))
    return modules


def build_entry_modules(section: dict) -> list:
    """Build enabled entry modules from config."""
    return build_modules(section, ENTRY_REGISTRY, "entry")


def build_exit_modules(section: dict) -> list:
    """Build enabled exit modules from config."""
    return build_modules(section, EXIT_REGISTRY, "exit")


def iter_entry_exit_combinations(entries: Iterable, exits: Iterable) -> Iterable[tuple]:
    """Yield every entry/exit pair."""
    yield from product(entries, exits)


def safe_run_id(*parts: str) -> str:
    """Build a filesystem/CSV friendly run id."""
    out = "__".join(str(part).strip().lower() for part in parts if str(part).strip())
    for old, new in {
        " ": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        "|": "_",
        "(": "",
        ")": "",
    }.items():
        out = out.replace(old, new)
    return out
