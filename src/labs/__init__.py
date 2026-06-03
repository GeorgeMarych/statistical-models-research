"""Research lab modules.

Labs compare candidate signals, state definitions, and outcomes. They are
research/validation surfaces, not production strategies.
"""
"""Research lab entry points."""

from src.labs.strategy_combo_lab import StrategyComboLabResult, run_strategy_combo_lab

__all__ = ["StrategyComboLabResult", "run_strategy_combo_lab"]
