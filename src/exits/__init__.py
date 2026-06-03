"""Reusable exit modules for modular strategy research."""
from src.exits.atr_stop_exit import ATRStopExit
from src.exits.base import BaseExit, ExitDecision, ExitProtocol, PositionContext
from src.exits.bollinger_exit import MiddleBollingerExit, UpperBollingerExit
from src.exits.fixed_bars_exit import FixedBarsExit
from src.exits.opposite_signal_exit import OppositeSignalExit
from src.exits.take_profit_stop_loss_exit import TakeProfitStopLossExit

__all__ = [
    "BaseExit",
    "ExitDecision",
    "ExitProtocol",
    "PositionContext",
    "FixedBarsExit",
    "OppositeSignalExit",
    "ATRStopExit",
    "MiddleBollingerExit",
    "UpperBollingerExit",
    "TakeProfitStopLossExit",
]
