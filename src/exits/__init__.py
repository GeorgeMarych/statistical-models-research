"""Reusable exit modules for modular strategy research."""
from src.exits.atr_stop_exit import ATRStopExit
from src.exits.base import BaseExit, ExitDecision, ExitProtocol, PositionContext
from src.exits.bollinger_exit import MiddleBollingerExit, UpperBollingerExit
from src.exits.fixed_bars_exit import FixedBarsExit
from src.exits.opposite_strength_exit import OppositeStrengthExit, VolumeStrengthSignal
from src.exits.opposite_signal_exit import OppositeSignalExit
from src.exits.percent_stop_loss_exit import PercentStopLossExit
from src.exits.profit_target_exit import ProfitTargetExit
from src.exits.stop_loss_exit import StopLossExit
from src.exits.take_profit_stop_loss_exit import TakeProfitStopLossExit
from src.exits.trailing_profit_exit import TrailingProfitExit

__all__ = [
    "BaseExit",
    "ExitDecision",
    "ExitProtocol",
    "PositionContext",
    "FixedBarsExit",
    "OppositeSignalExit",
    "OppositeStrengthExit",
    "VolumeStrengthSignal",
    "StopLossExit",
    "PercentStopLossExit",
    "ProfitTargetExit",
    "ATRStopExit",
    "TrailingProfitExit",
    "MiddleBollingerExit",
    "UpperBollingerExit",
    "TakeProfitStopLossExit",
]
