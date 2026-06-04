"""Objective helpers for Simple Strategy #1."""
from __future__ import annotations

from src.optimization.objective import balanced_score, score_summary


def simple_strategy_01_score(summary: dict) -> float:
    """Balanced score used for the software volume-fade lab."""
    return score_summary(summary, "balanced")


__all__ = ["balanced_score", "score_summary", "simple_strategy_01_score"]
