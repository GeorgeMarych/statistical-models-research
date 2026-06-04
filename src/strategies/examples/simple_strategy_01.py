"""Simple Strategy #1: software volume-fade reversal."""
from __future__ import annotations

from src.backtesting.exit_stack import ExitStack
from src.costs.stock_cost_model import StockCostModel
from src.entries.price_extreme_signals import HighestCloseSignal, LowestCloseSignal
from src.entries.volume_fade_entry import VolumeFadeLongEntry, VolumeFadeReversalEntry
from src.exits.fixed_bars_exit import FixedBarsExit
from src.exits.opposite_strength_exit import OppositeStrengthExit, VolumeStrengthSignal
from src.exits.opposite_signal_exit import OppositeSignalExit
from src.exits.percent_stop_loss_exit import PercentStopLossExit
from src.filters.market_regime_filter import MarketRegimeFilter
from src.filters.trend_filter import TrendFilter
from src.filters.volume_filter import VolumeAboveAverageFilter
from src.filters.weekday_filter import WeekdayFilter
from src.filters.wide_bar_filter import WideBullishBarFilter
from src.sizing.percent_equity import PercentEquitySizing
from src.strategies.strategy_definition import StrategyDefinition

PRIMARY_SYMBOL = "CRWD"

CYBERSECURITY_SYMBOLS = [
    "CRWD",
    "PANW",
    "FTNT",
    "ZS",
    "NET",
    "S",
    "OKTA",
    "CYBR",
    "TENB",
    "VRNS",
    "RPD",
]

HIGH_GROWTH_SOFTWARE_SYMBOLS = [
    "DDOG",
    "MDB",
    "TEAM",
    "SNOW",
    "PLTR",
    "APP",
    "NOW",
    "SHOP",
    "HUBS",
    "CFLT",
]

PEER_UNIVERSE = CYBERSECURITY_SYMBOLS + HIGH_GROWTH_SOFTWARE_SYMBOLS


def build_software_volume_fade_reversal_strategy(
    symbols: list[str] | None = None,
    long_length: int = 2,
    short_length: int = 11,
    volume_avg_length: int = 10,
    avoid_wednesday_longs: bool = True,
    trend_filter: bool = False,
    trend_length: int = 200,
    market_regime_filter: bool = False,
    market_symbol: str = "QQQ",
    market_trend_length: int = 200,
    wide_bullish_filter: bool = True,
    range_avg_length: int = 10,
    wide_range_mult: float = 1.0,
    fixed_bars_exit: bool = True,
    exit_bars: int = 25,
    stop_loss_enabled: bool = False,
    stop_loss_pct: float = 20.0,
    percent_equity: float = 100.0,
    commission_percent: float = 0.03,
    slippage_bps_per_side: float = 0.0,
    initial_capital: float = 100000.0,
    direction_mode: str = "long_only",
    enable_long_entries: bool = True,
    enable_short_entries: bool | None = None,
    use_opposite_signal_as_exit: bool = True,
    if_side_disabled_use_signal_only_as_exit: bool = True,
    min_bars: int = 80,
) -> StrategyDefinition:
    """Compose the strategy from reusable modules."""
    symbols = _normalize_symbols(symbols or [PRIMARY_SYMBOL])
    mode = direction_mode.lower().replace("-", "_")
    short_entries_enabled = (
        enable_short_entries
        if enable_short_entries is not None
        else mode in {"long_short", "long/short", "longshort"}
    )

    weekday_filter = WeekdayFilter(
        skip_days=["Wednesday"] if avoid_wednesday_longs else [],
        apply_to_long=True,
        apply_to_short=False,
        label="weekday_skip_wednesday",
    )
    trend_module = TrendFilter(
        length=trend_length,
        mode="close_above_sma",
        enabled=trend_filter,
        label="trend_filter",
    )
    market_module = MarketRegimeFilter(
        benchmark_symbol=market_symbol,
        length=market_trend_length,
        mode="close_above_sma",
        enabled=market_regime_filter,
        label="market_regime_filter",
    )
    wide_bar_filter = WideBullishBarFilter(
        range_length=range_avg_length,
        range_mult=wide_range_mult,
        block_when_true=True,
        enabled=wide_bullish_filter,
        apply_to_long=False,
        apply_to_short=True,
        label="wide_bullish_bar",
    )

    shared_filters = [
        trend_module,
        market_module,
    ]
    if short_entries_enabled or mode in {"long_short", "long/short", "longshort"}:
        entry = VolumeFadeReversalEntry(
            long_extreme_signal=LowestCloseSignal(length=long_length),
            short_extreme_signal=HighestCloseSignal(length=short_length),
            volume_filter=VolumeAboveAverageFilter(
                length=volume_avg_length,
                multiplier=1.0,
            ),
            long_filters=[weekday_filter],
            short_filters=[wide_bar_filter],
            enable_long_entries=enable_long_entries,
            enable_short_entries=bool(short_entries_enabled),
        )
    else:
        entry = VolumeFadeLongEntry(
            extreme_signal=LowestCloseSignal(length=long_length),
            volume_filter=VolumeAboveAverageFilter(
                length=volume_avg_length,
                multiplier=1.0,
            ),
        )

    filters = [
        WeekdayFilter(
            skip_days=["Wednesday"] if avoid_wednesday_longs else [],
            apply_to_long=True,
            apply_to_short=False,
            label="weekday_skip_wednesday",
        ),
        trend_module,
        market_module,
    ]

    opposite_strength_signal = VolumeStrengthSignal(
        direction="short",
        extreme_signal=HighestCloseSignal(length=short_length),
        volume_filter=VolumeAboveAverageFilter(
            length=volume_avg_length,
            multiplier=1.0,
        ),
        filters=[wide_bar_filter],
    )

    exits = [
        PercentStopLossExit(
            stop_loss_pct=stop_loss_pct,
            enabled=stop_loss_enabled,
            label="percent_stop_loss",
        )
    ]
    if use_opposite_signal_as_exit and short_entries_enabled:
        exits.append(
            OppositeSignalExit(
                reverse_on_opposite=True,
                label="opposite_signal",
            )
        )
    elif use_opposite_signal_as_exit or if_side_disabled_use_signal_only_as_exit:
        exits.append(
            OppositeStrengthExit(
                signal=opposite_strength_signal,
                reverse_on_opposite=False,
                label="opposite_strength",
            )
        )
    if fixed_bars_exit:
        exits.append(FixedBarsExit(bars=exit_bars, label="fixed_bars"))

    return StrategyDefinition(
        name="software_volume_fade_reversal",
        symbols=symbols,
        entry=entry,
        exit_stack=ExitStack(exits=exits, label="volume_fade_exit_stack"),
        filters=filters,
        direction_mode=direction_mode,
        filter_mode="all",
        initial_capital=initial_capital,
        sizing=PercentEquitySizing(percent=percent_equity),
        costs=StockCostModel(
            commission_percent=commission_percent,
            slippage_bps_per_side=slippage_bps_per_side,
        ),
        parameters={
            "long_length": long_length,
            "short_length": short_length,
            "volume_avg_length": volume_avg_length,
            "avoid_wednesday_longs": avoid_wednesday_longs,
            "wide_bullish_filter": wide_bullish_filter,
            "wide_range_mult": wide_range_mult,
            "fixed_bars_exit": fixed_bars_exit,
            "exit_bars": exit_bars,
            "trend_filter": trend_filter,
            "trend_length": trend_length,
            "market_regime_filter": market_regime_filter,
            "stop_loss_enabled": stop_loss_enabled,
            "stop_loss_pct": stop_loss_pct,
            "percent_equity": percent_equity,
            "commission_percent": commission_percent,
            "slippage_bps_per_side": slippage_bps_per_side,
            "enable_long_entries": enable_long_entries,
            "enable_short_entries": bool(short_entries_enabled),
            "use_opposite_signal_as_exit": use_opposite_signal_as_exit,
            "if_side_disabled_use_signal_only_as_exit": if_side_disabled_use_signal_only_as_exit,
        },
        min_bars=min_bars,
    )


def _normalize_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out
