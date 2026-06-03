"""Base interfaces for entry modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd

SIGNAL_COLUMN = "entry_signal"


@runtime_checkable
class EntryProtocol(Protocol):
    """Protocol implemented by strategy entry modules."""

    @property
    def name(self) -> str:
        """Human-readable entry name."""

    @property
    def parameters(self) -> dict:
        """Serializable parameter dictionary."""

    @property
    def min_lookback(self) -> int:
        """Minimum bars expected before this entry can emit a signal."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed like data with an entry_signal column."""


class BaseEntry(ABC):
    """Base class for concrete entry modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable entry name."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Serializable parameter dictionary."""

    @property
    def min_lookback(self) -> int:
        """Minimum bars expected before this entry can emit a signal."""
        return 0

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed like data with an entry_signal column."""


def make_signal_frame(
    data: pd.DataFrame,
    signal: pd.Series,
    extra_columns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build a normalized entry signal DataFrame."""
    out = pd.DataFrame(index=data.index)
    clean_signal = signal.reindex(data.index).fillna(0).astype(int)
    clean_signal = clean_signal.clip(lower=-1, upper=1)
    out[SIGNAL_COLUMN] = clean_signal
    if extra_columns:
        for name, values in extra_columns.items():
            out[name] = values.reindex(data.index)
    return out


def validate_entry_signals(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    entry_name: str,
) -> None:
    """Validate shape and values of an entry signal frame."""
    if not signals.index.equals(data.index):
        raise ValueError(f"{entry_name}: signals must have the same index as data")
    if SIGNAL_COLUMN not in signals.columns:
        raise ValueError(f"{entry_name}: missing required {SIGNAL_COLUMN!r} column")
    clean = signals[SIGNAL_COLUMN].dropna()
    bad_values = sorted(set(clean.unique()) - {-1, 0, 1})
    if bad_values:
        raise ValueError(
            f"{entry_name}: {SIGNAL_COLUMN} values must be -1, 0, or 1; "
            f"got {bad_values}"
        )
