"""Entry modules for modular strategy research."""
from src.entries.base import EntryProtocol, BaseEntry, validate_entry_signals
from src.entries.close_breakout import CloseBreakoutEntry
from src.entries.donchian_breakout_entry import DonchianBreakoutEntry
from src.entries.dueling_momentum import DuelingMomentumEntry
from src.entries.ma_crossover_entry import MovingAverageCrossoverEntry
from src.entries.mean_reversion import MeanReversionEntry
from src.entries.price_extreme_signals import HighestCloseSignal, LowestCloseSignal
from src.entries.price_pattern import PricePatternEntry
from src.entries.rsi_bb_entry import RsiBollingerEntry
from src.entries.stop_breakout import StopBreakoutEntry
from src.entries.volume_fade_entry import VolumeFadeLongEntry, VolumeFadeReversalEntry

__all__ = [
    "BaseEntry",
    "EntryProtocol",
    "validate_entry_signals",
    "CloseBreakoutEntry",
    "StopBreakoutEntry",
    "MeanReversionEntry",
    "RsiBollingerEntry",
    "DonchianBreakoutEntry",
    "MovingAverageCrossoverEntry",
    "DuelingMomentumEntry",
    "PricePatternEntry",
    "LowestCloseSignal",
    "HighestCloseSignal",
    "VolumeFadeLongEntry",
    "VolumeFadeReversalEntry",
]
