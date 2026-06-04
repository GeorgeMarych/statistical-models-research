"""Small parameter-grid utilities."""
from __future__ import annotations

from itertools import product
from typing import Any, Iterable


def iter_parameter_grid(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    """Yield every parameter combination from a dotted-path grid."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def grid_size(grid: dict[str, list[Any]]) -> int:
    """Return the number of combinations in a grid."""
    size = 1
    for values in grid.values():
        size *= len(values)
    return size
