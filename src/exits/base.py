"""Base interfaces for reusable exit modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

ExitTiming = Literal["intrabar", "next_open"]


@dataclass
class PositionContext:
    """Position details exposed to exit modules."""

    symbol: str
    side: int
    quantity: float
    entry_price: float
    entry_index: int
    signal_index: int
    entry_date: pd.Timestamp
    signal_date: pd.Timestamp
    bars_held: int


@dataclass
class ExitDecision:
    """Decision returned by an exit module for the current bar."""

    should_exit: bool = False
    reason: str = ""
    timing: ExitTiming = "next_open"
    exit_price: float | None = None
    reverse_to_side: int | None = None

    @classmethod
    def no_action(cls) -> "ExitDecision":
        return cls(False)


@runtime_checkable
class ExitProtocol(Protocol):
    """Protocol implemented by exit modules."""

    @property
    def name(self) -> str:
        """Human-readable exit name."""

    @property
    def parameters(self) -> dict:
        """Serializable parameter dictionary."""

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Precompute trailing exit indicators for a single run."""

    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        """Evaluate the exit on the current bar."""


class BaseExit(ABC):
    """Base class for concrete exit modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable exit name."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Serializable parameter dictionary."""

    def prepare(
        self,
        data: pd.DataFrame,
        entry_signals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Precompute trailing exit indicators for a single run."""
        return pd.DataFrame(index=data.index)

    @abstractmethod
    def on_bar(
        self,
        data: pd.DataFrame,
        index: int,
        position: PositionContext,
        context: dict,
    ) -> ExitDecision:
        """Evaluate the exit on the current bar."""
