"""Composable exit stacks for complete strategy definitions."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from src.exits.base import ExitProtocol


@dataclass
class ExitStack(Sequence):
    """
    Ordered collection of exit modules.

    The engine evaluates exits in this order. The first exit that triggers closes
    the trade, so emergency stops should usually come before slower indicator or
    time-based exits.
    """

    exits: list[ExitProtocol] = field(default_factory=list)
    label: str = "exit_stack"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {
            "exits": [
                {
                    "name": exit_module.name,
                    "parameters": exit_module.parameters,
                }
                for exit_module in self.exits
            ]
        }

    def __iter__(self) -> Iterator[ExitProtocol]:
        return iter(self.exits)

    def __len__(self) -> int:
        return len(self.exits)

    def __getitem__(self, index: int) -> ExitProtocol:
        return self.exits[index]

    def find(self, name: str) -> ExitProtocol:
        """Return the first exit whose name matches name."""
        for exit_module in self.exits:
            if exit_module.name == name:
                return exit_module
        raise KeyError(f"Exit {name!r} not found in {self.name}")
