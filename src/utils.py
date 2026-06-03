"""General utilities: logging setup and filesystem helpers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple console logger."""
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (and parents) if they do not already exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
