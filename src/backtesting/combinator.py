"""Factories and utilities for entry/exit combination experiments."""
from __future__ import annotations

from itertools import product
from typing import Iterable

from src.backtesting.exit_stack import ExitStack
from src.entries import (
    CloseBreakoutEntry,
    DonchianBreakoutEntry,
    DuelingMomentumEntry,
    MeanReversionEntry,
    MovingAverageCrossoverEntry,
    PricePatternEntry,
    RsiBollingerEntry,
    StopBreakoutEntry,
    VolumeFadeLongEntry,
    VolumeFadeReversalEntry,
)
from src.exits import (
    ATRStopExit,
    FixedBarsExit,
    MiddleBollingerExit,
    OppositeSignalExit,
    OppositeStrengthExit,
    PercentStopLossExit,
    ProfitTargetExit,
    StopLossExit,
    TakeProfitStopLossExit,
    TrailingProfitExit,
    UpperBollingerExit,
)
from src.filters import (
    DayOfWeekFilter,
    MarketRegimeFilter,
    TrendFilter,
    VolatilityFilter,
    VolumeAboveAverageFilter,
    VolumeFilter,
    WeekdayFilter,
    WideBullishBarFilter,
)

ENTRY_REGISTRY = {
    "close_breakout": CloseBreakoutEntry,
    "stop_breakout": StopBreakoutEntry,
    "mean_reversion": MeanReversionEntry,
    "rsi_bb_mean_reversion": RsiBollingerEntry,
    "rsi_bb": RsiBollingerEntry,
    "donchian_breakout": DonchianBreakoutEntry,
    "ma_crossover": MovingAverageCrossoverEntry,
    "moving_average_crossover": MovingAverageCrossoverEntry,
    "dueling_momentum": DuelingMomentumEntry,
    "price_pattern": PricePatternEntry,
    "volume_fade_long": VolumeFadeLongEntry,
    "volume_fade_entry": VolumeFadeLongEntry,
    "volume_fade_reversal": VolumeFadeReversalEntry,
}

EXIT_REGISTRY = {
    "fixed_bars": FixedBarsExit,
    "opposite_signal": OppositeSignalExit,
    "opposite_strength": OppositeStrengthExit,
    "stop_loss": StopLossExit,
    "percent_stop_loss": PercentStopLossExit,
    "profit_target": ProfitTargetExit,
    "atr_stop": ATRStopExit,
    "trailing_profit": TrailingProfitExit,
    "middle_bollinger": MiddleBollingerExit,
    "bollinger_middle": MiddleBollingerExit,
    "upper_bollinger": UpperBollingerExit,
    "bollinger_upper": UpperBollingerExit,
    "take_profit_stop_loss": TakeProfitStopLossExit,
    "tp_sl": TakeProfitStopLossExit,
}

FILTER_REGISTRY = {
    "trend_filter": TrendFilter,
    "trend": TrendFilter,
    "volatility_filter": VolatilityFilter,
    "volatility": VolatilityFilter,
    "volume_filter": VolumeFilter,
    "volume": VolumeFilter,
    "volume_above_average": VolumeAboveAverageFilter,
    "volume_above_average_filter": VolumeAboveAverageFilter,
    "market_regime_filter": MarketRegimeFilter,
    "market_regime": MarketRegimeFilter,
    "day_of_week_filter": DayOfWeekFilter,
    "day_of_week": DayOfWeekFilter,
    "weekday_filter": WeekdayFilter,
    "weekday": WeekdayFilter,
    "wide_bullish_bar_filter": WideBullishBarFilter,
    "wide_bullish_bar": WideBullishBarFilter,
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


def build_filter_modules(section: dict) -> list:
    """Build enabled filter modules from config."""
    return build_modules(section, FILTER_REGISTRY, "filter")


def build_exit_stack(section: dict | list, label: str = "exit_stack") -> ExitStack:
    """Build an ExitStack from a dict or list config section."""
    if isinstance(section, list):
        exits = []
        for index, spec in enumerate(section):
            spec = spec or {}
            if not spec.get("enabled", True):
                continue
            module_name = str(spec.get("module", spec.get("name", ""))).strip()
            if module_name not in EXIT_REGISTRY:
                known = ", ".join(sorted(EXIT_REGISTRY))
                raise ValueError(f"Unknown exit module {module_name!r}. Known: {known}")
            parameters = dict(spec.get("parameters", {}) or {})
            parameters.setdefault("label", str(spec.get("label", spec.get("name", f"exit_{index}"))))
            exits.append(EXIT_REGISTRY[module_name](**parameters))
        return ExitStack(exits=exits, label=label)

    exits = build_exit_modules(section or {})
    return ExitStack(exits=exits, label=label)


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
