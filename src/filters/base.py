"""Base interfaces for reusable strategy filters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pandas as pd

ALLOW_LONG = "allow_long"
ALLOW_SHORT = "allow_short"


@runtime_checkable
class FilterProtocol(Protocol):
    """Protocol implemented by filters that gate entry signals."""

    @property
    def name(self) -> str:
        """Human-readable filter name."""

    @property
    def parameters(self) -> dict:
        """Serializable parameter dictionary."""

    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        """Return allow_long and allow_short columns aligned with data."""


class BaseFilter(ABC):
    """Base class for concrete filters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Serializable parameter dictionary."""

    @abstractmethod
    def generate_mask(
        self,
        data: pd.DataFrame,
        context: dict | None = None,
    ) -> pd.DataFrame:
        """Return allow_long and allow_short columns aligned with data."""


def make_filter_frame(
    data: pd.DataFrame,
    allow_long: pd.Series | bool,
    allow_short: pd.Series | bool | None = None,
    extra_columns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build a normalized filter mask frame."""
    out = pd.DataFrame(index=data.index)
    if isinstance(allow_long, bool):
        out[ALLOW_LONG] = allow_long
    else:
        out[ALLOW_LONG] = allow_long.reindex(data.index).fillna(False).astype(bool)

    if allow_short is None:
        allow_short = allow_long
    if isinstance(allow_short, bool):
        out[ALLOW_SHORT] = allow_short
    else:
        out[ALLOW_SHORT] = allow_short.reindex(data.index).fillna(False).astype(bool)

    if extra_columns:
        for name, values in extra_columns.items():
            out[name] = values.reindex(data.index)
    return out


def validate_filter_mask(data: pd.DataFrame, mask: pd.DataFrame, filter_name: str) -> None:
    """Validate a filter mask frame."""
    if not mask.index.equals(data.index):
        raise ValueError(f"{filter_name}: mask must have the same index as data")
    missing = [col for col in [ALLOW_LONG, ALLOW_SHORT] if col not in mask.columns]
    if missing:
        raise ValueError(f"{filter_name}: missing required columns {missing}")
