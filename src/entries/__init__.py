"""Entry modules for modular strategy research."""
from src.entries.base import EntryProtocol, BaseEntry, validate_entry_signals
from src.entries.donchian_breakout_entry import DonchianBreakoutEntry
from src.entries.ma_crossover_entry import MovingAverageCrossoverEntry
from src.entries.rsi_bb_entry import RsiBollingerEntry

__all__ = [
    "BaseEntry",
    "EntryProtocol",
    "validate_entry_signals",
    "RsiBollingerEntry",
    "DonchianBreakoutEntry",
    "MovingAverageCrossoverEntry",
]
