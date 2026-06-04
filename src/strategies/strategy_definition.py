"""Complete strategy definitions built from reusable modules."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.backtesting.combinator import (
    ENTRY_REGISTRY,
    FILTER_REGISTRY,
    build_exit_stack,
)
from src.backtesting.costs import TradingCosts
from src.backtesting.exit_stack import ExitStack
from src.backtesting.portfolio import PositionSizing
from src.entries.base import EntryProtocol
from src.filters.base import FilterProtocol
from src.strategies.base import FilteredEntry


@dataclass
class StrategyDefinition:
    """One complete backtestable strategy definition."""

    name: str
    symbols: list[str]
    entry: EntryProtocol
    exit_stack: ExitStack
    filters: list[FilterProtocol] = field(default_factory=list)
    direction_mode: str = "long_only"
    filter_mode: str = "all"
    initial_capital: float = 100000.0
    sizing: PositionSizing = field(default_factory=PositionSizing)
    costs: TradingCosts = field(default_factory=TradingCosts)
    parameters: dict[str, Any] = field(default_factory=dict)
    periods_per_year: int = 252
    min_bars: int = 80

    def filtered_entry(self, filter_context: dict | None = None) -> FilteredEntry:
        """Return an entry adapter with strategy filters applied."""
        return FilteredEntry(
            entry=self.entry,
            filters=self.filters,
            filter_mode=self.filter_mode,
            direction_mode=self.direction_mode,
            filter_context=filter_context,
        )

    @property
    def allow_short(self) -> bool:
        mode = str(self.direction_mode).lower().replace("-", "_")
        return mode in {"short_only", "long_short", "long/short", "longshort"}

    def clone_with_parameter(self, path: str, value: Any) -> "StrategyDefinition":
        """Return a copy with one dotted parameter path changed."""
        clone = deepcopy(self)
        clone.set_parameter(path, value)
        return clone

    def clone_with_parameters(self, parameters: dict[str, Any]) -> "StrategyDefinition":
        """Return a copy with multiple dotted parameter paths changed."""
        clone = deepcopy(self)
        for path, value in parameters.items():
            clone.set_parameter(path, value)
        return clone

    def set_parameter(self, path: str, value: Any) -> None:
        """Set a dotted parameter path on entry, filters, exits, sizing, or costs."""
        parts = path.split(".")
        if not parts:
            raise ValueError("empty parameter path")
        root = parts[0]
        if root == "entry":
            _set_attr(self.entry, parts[1:], value)
        elif root in {"exit", "exits", "exit_stack"}:
            exit_module = _select_named_or_indexed(self.exit_stack.exits, parts[1], "exit")
            _set_attr(exit_module, parts[2:], value)
        elif root == "filters":
            filter_module = _select_named_or_indexed(self.filters, parts[1], "filter")
            _set_attr(filter_module, parts[2:], value)
        elif root == "sizing":
            _set_attr(self.sizing, parts[1:], value)
        elif root == "costs":
            _set_attr(self.costs, parts[1:], value)
        elif root == "strategy":
            _set_attr(self, parts[1:], value)
        else:
            raise ValueError(f"Unsupported parameter root: {root}")
        self.parameters[path] = value


def strategy_from_config(config: dict[str, Any]) -> StrategyDefinition:
    """Build a StrategyDefinition from a YAML-style config dictionary."""
    raw = config.get("strategy", config)
    name = str(raw.get("name", "strategy"))
    symbols = _normalize_symbols(raw.get("symbols", config.get("symbols", [])))
    if not symbols:
        raise ValueError("strategy config must include symbols")

    entry = _build_one(raw.get("entry", {}), ENTRY_REGISTRY, "entry", "entry")
    filters = [
        _build_one(spec, FILTER_REGISTRY, "filter", f"filter_{i}")
        for i, spec in enumerate(raw.get("filters", []) or [])
        if (spec or {}).get("enabled", True)
    ]
    exit_stack = build_exit_stack(
        raw.get("exit_stack", raw.get("exits", [])),
        label=str(raw.get("exit_stack_name", "exit_stack")),
    )
    if not exit_stack:
        raise ValueError("strategy config must include at least one exit")

    sizing = PositionSizing(**(raw.get("sizing", config.get("sizing", {})) or {}))
    costs = TradingCosts(**(raw.get("costs", config.get("costs", {})) or {}))

    return StrategyDefinition(
        name=name,
        symbols=symbols,
        entry=entry,
        exit_stack=exit_stack,
        filters=filters,
        direction_mode=str(raw.get("direction_mode", raw.get("mode", "long_only"))),
        filter_mode=str(raw.get("filter_mode", "all")),
        initial_capital=float(raw.get("initial_capital", config.get("initial_capital", 100000.0))),
        sizing=sizing,
        costs=costs,
        parameters=dict(raw.get("parameters", {}) or {}),
        periods_per_year=int(raw.get("periods_per_year", config.get("periods_per_year", 252))),
        min_bars=int(raw.get("min_bars", config.get("min_bars", 80))),
    )


def _build_one(spec: dict[str, Any], registry: dict, kind: str, default_label: str):
    spec = spec or {}
    module_name = str(spec.get("module", spec.get("name", ""))).strip()
    if module_name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} module {module_name!r}. Known: {known}")
    parameters = dict(spec.get("parameters", {}) or {})
    parameters.setdefault("label", str(spec.get("label", spec.get("name", default_label))))
    return registry[module_name](**parameters)


def _select_named_or_indexed(modules: list, key: str, kind: str):
    if key.isdigit():
        return modules[int(key)]
    for module in modules:
        if getattr(module, "name", None) == key:
            return module
    raise KeyError(f"{kind} {key!r} not found")


def _set_attr(target, parts: list[str], value: Any) -> None:
    if not parts:
        raise ValueError("missing attribute name in parameter path")
    obj = target
    for part in parts[:-1]:
        if isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, list) and last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def _normalize_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out
