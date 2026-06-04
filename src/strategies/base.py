"""Strategy helper classes."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.entries.base import SIGNAL_COLUMN, EntryProtocol
from src.filters.base import ALLOW_LONG, ALLOW_SHORT, FilterProtocol, validate_filter_mask


@dataclass
class FilteredEntry:
    """Entry adapter that applies filters and direction mode."""

    entry: EntryProtocol
    filters: list[FilterProtocol] = field(default_factory=list)
    filter_mode: str = "all"
    direction_mode: str = "long_only"
    filter_context: dict | None = None

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def parameters(self) -> dict:
        return self.entry.parameters

    @property
    def min_lookback(self) -> int:
        return getattr(self.entry, "min_lookback", 0)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = self.entry.generate_signals(data).copy()
        allow_long, allow_short = self._combined_filter_masks(data)

        signals.loc[signals[SIGNAL_COLUMN] == 1, SIGNAL_COLUMN] = (
            signals.loc[signals[SIGNAL_COLUMN] == 1, SIGNAL_COLUMN]
            .where(allow_long.loc[signals[SIGNAL_COLUMN] == 1], 0)
        )
        signals.loc[signals[SIGNAL_COLUMN] == -1, SIGNAL_COLUMN] = (
            signals.loc[signals[SIGNAL_COLUMN] == -1, SIGNAL_COLUMN]
            .where(allow_short.loc[signals[SIGNAL_COLUMN] == -1], 0)
        )

        mode = str(self.direction_mode).lower().replace("-", "_")
        if mode == "long_only":
            signals.loc[signals[SIGNAL_COLUMN] < 0, SIGNAL_COLUMN] = 0
        elif mode == "short_only":
            signals.loc[signals[SIGNAL_COLUMN] > 0, SIGNAL_COLUMN] = 0
        elif mode in {"long_short", "long/short", "longshort"}:
            pass
        else:
            raise ValueError(f"Unsupported direction mode: {self.direction_mode}")

        return signals

    def _combined_filter_masks(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if not self.filters:
            return (
                pd.Series(True, index=data.index),
                pd.Series(True, index=data.index),
            )

        long_masks: list[pd.Series] = []
        short_masks: list[pd.Series] = []
        context = self.filter_context or {}
        for filter_module in self.filters:
            mask = filter_module.generate_mask(data, context)
            validate_filter_mask(data, mask, filter_module.name)
            long_masks.append(mask[ALLOW_LONG].astype(bool))
            short_masks.append(mask[ALLOW_SHORT].astype(bool))

        filter_mode = str(self.filter_mode).lower()
        if filter_mode == "all":
            allow_long = pd.concat(long_masks, axis=1).all(axis=1)
            allow_short = pd.concat(short_masks, axis=1).all(axis=1)
        elif filter_mode == "any":
            allow_long = pd.concat(long_masks, axis=1).any(axis=1)
            allow_short = pd.concat(short_masks, axis=1).any(axis=1)
        else:
            raise ValueError("filter_mode must be 'all' or 'any'")

        return allow_long, allow_short
