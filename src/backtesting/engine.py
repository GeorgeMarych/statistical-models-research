"""Simple event-driven backtest engine for entry/exit combinations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.backtesting.costs import TradingCosts
from src.backtesting.metrics import summarize_backtest
from src.backtesting.portfolio import Position, PositionSizing, mark_to_market
from src.entries.base import SIGNAL_COLUMN, EntryProtocol, validate_entry_signals
from src.exits.base import ExitDecision, ExitProtocol, PositionContext

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
FUTURE_COLUMN_TOKENS = (
    "future",
    "forward",
    "mfe",
    "mae",
    "outcome",
    "target_before_stop",
)

TRADE_COLUMNS = [
    "symbol",
    "side",
    "entry_date",
    "exit_date",
    "signal_date",
    "entry_index",
    "exit_index",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "net_pnl",
    "gross_return",
    "net_return",
    "entry_commission",
    "exit_commission",
    "bars_held",
    "exit_reason",
]


@dataclass
class BacktestSettings:
    """Settings for one symbol/entry/exit run."""

    symbol: str
    initial_capital: float = 100000.0
    allow_short: bool = False
    sizing: PositionSizing = field(default_factory=PositionSizing)
    costs: TradingCosts = field(default_factory=TradingCosts)
    force_flat_at_end: bool = True
    periods_per_year: int = 252


@dataclass
class BacktestResult:
    """Backtest outputs for one run."""

    summary: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    signals: pd.DataFrame


def find_future_columns(columns: Sequence[str]) -> list[str]:
    """Return columns whose names look like future/outcome labels."""
    bad: list[str] = []
    for column in columns:
        clean = str(column).lower()
        if any(token in clean for token in FUTURE_COLUMN_TOKENS):
            bad.append(str(column))
    return bad


def prepare_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a price frame for strategy modules.

    Only OHLCV columns are retained, which prevents entry modules from touching
    legacy future-return or outcome columns even if a wide research panel is
    accidentally supplied.
    """
    if data.empty:
        raise ValueError("data is empty")

    out = data.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    if "close" not in out.columns:
        raise ValueError("data must include a close column")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "open" not in out.columns:
        out["open"] = out["close"]
    if "high" not in out.columns:
        out["high"] = pd.concat([out["open"], out["close"]], axis=1).max(axis=1)
    if "low" not in out.columns:
        out["low"] = pd.concat([out["open"], out["close"]], axis=1).min(axis=1)
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[PRICE_COLUMNS].copy()
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index.name = "date"
    return out


class BacktestEngine:
    """One-position-at-a-time event-driven backtest engine."""

    def __init__(self, settings: BacktestSettings) -> None:
        self.settings = settings

    def run(
        self,
        data: pd.DataFrame,
        entry: EntryProtocol,
        exit_modules: ExitProtocol | Sequence[ExitProtocol],
    ) -> BacktestResult:
        """Run one entry against one or more exits on a single symbol."""
        prices = prepare_ohlcv(data)
        exits = self._normalize_exits(exit_modules)

        signals = entry.generate_signals(prices)
        validate_entry_signals(prices, signals, entry.name)

        prepared_exits = {
            exit_module.name: exit_module.prepare(prices, signals)
            for exit_module in exits
        }

        cash = float(self.settings.initial_capital)
        position: Position | None = None
        trades: list[dict] = []
        equity_records: list[dict] = []
        pending_entry_side: int | None = None
        pending_entry_signal_index: int | None = None
        pending_exit: ExitDecision | None = None

        for index, (date, row) in enumerate(prices.iterrows()):
            open_price = float(row["open"])
            close_price = float(row["close"])

            if position is not None and pending_exit is not None:
                position, cash = self._close_position(
                    position=position,
                    raw_price=open_price,
                    exit_index=index,
                    exit_date=date,
                    cash=cash,
                    reason=pending_exit.reason,
                    trades=trades,
                )
                reverse_side = pending_exit.reverse_to_side
                pending_exit = None
                if position is None and reverse_side is not None and self._side_allowed(reverse_side):
                    position, cash = self._open_position(
                        side=reverse_side,
                        raw_price=open_price,
                        entry_index=index,
                        signal_index=max(index - 1, 0),
                        cash=cash,
                        prices=prices,
                    )

            if position is None and pending_entry_side is not None:
                if self._side_allowed(pending_entry_side):
                    position, cash = self._open_position(
                        side=pending_entry_side,
                        raw_price=open_price,
                        entry_index=index,
                        signal_index=pending_entry_signal_index or max(index - 1, 0),
                        cash=cash,
                        prices=prices,
                    )
                pending_entry_side = None
                pending_entry_signal_index = None

            if position is not None:
                position_context = self._position_context(position, prices, index)
                context = self._context(
                    phase="intrabar",
                    prepared_exits=prepared_exits,
                    signals=signals,
                )
                for exit_module in exits:
                    decision = exit_module.on_bar(prices, index, position_context, context)
                    if decision.should_exit and decision.timing == "intrabar":
                        raw_exit_price = decision.exit_price
                        if raw_exit_price is None or pd.isna(raw_exit_price):
                            raw_exit_price = close_price
                        position, cash = self._close_position(
                            position=position,
                            raw_price=float(raw_exit_price),
                            exit_index=index,
                            exit_date=date,
                            cash=cash,
                            reason=decision.reason,
                            trades=trades,
                        )
                        break

            equity = mark_to_market(cash, position, close_price)
            equity_records.append(
                {
                    "date": date,
                    "symbol": self.settings.symbol,
                    "cash": cash,
                    "close": close_price,
                    "equity": equity,
                    "position_side": position.side if position is not None else 0,
                    "position_quantity": (
                        position.quantity if position is not None else 0.0
                    ),
                }
            )

            pending_exit = None
            if position is not None:
                position_context = self._position_context(position, prices, index)
                context = self._context(
                    phase="close",
                    prepared_exits=prepared_exits,
                    signals=signals,
                )
                for exit_module in exits:
                    decision = exit_module.on_bar(prices, index, position_context, context)
                    if decision.should_exit and decision.timing == "next_open":
                        pending_exit = decision
                        break

            if position is None and index < len(prices) - 1:
                signal = int(signals[SIGNAL_COLUMN].iloc[index])
                if signal != 0 and self._side_allowed(signal):
                    pending_entry_side = signal
                    pending_entry_signal_index = index

        if position is not None and self.settings.force_flat_at_end:
            last_index = len(prices) - 1
            last_date = prices.index[last_index]
            last_close = float(prices["close"].iloc[last_index])
            position, cash = self._close_position(
                position=position,
                raw_price=last_close,
                exit_index=last_index,
                exit_date=last_date,
                cash=cash,
                reason="end_of_data",
                trades=trades,
            )
            if equity_records:
                equity_records[-1]["cash"] = cash
                equity_records[-1]["equity"] = cash
                equity_records[-1]["position_side"] = 0
                equity_records[-1]["position_quantity"] = 0.0

        equity_curve = pd.DataFrame(equity_records)
        if not equity_curve.empty:
            peak = equity_curve["equity"].cummax()
            equity_curve["drawdown"] = equity_curve["equity"] / peak - 1.0
            equity_curve["period_return"] = equity_curve["equity"].pct_change()

        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        summary = summarize_backtest(
            equity_curve=equity_curve,
            trades=trades_df,
            initial_capital=self.settings.initial_capital,
            periods_per_year=self.settings.periods_per_year,
        )
        return BacktestResult(summary, trades_df, equity_curve, signals)

    def _normalize_exits(
        self,
        exit_modules: ExitProtocol | Sequence[ExitProtocol],
    ) -> list[ExitProtocol]:
        if isinstance(exit_modules, Sequence) and not isinstance(exit_modules, (str, bytes)):
            exits = list(exit_modules)
        else:
            exits = [exit_modules]  # type: ignore[list-item]
        if not exits:
            raise ValueError("at least one exit module is required")
        return exits

    def _side_allowed(self, side: int) -> bool:
        return side == 1 or (side == -1 and self.settings.allow_short)

    def _open_position(
        self,
        side: int,
        raw_price: float,
        entry_index: int,
        signal_index: int,
        cash: float,
        prices: pd.DataFrame,
    ) -> tuple[Position | None, float]:
        order_side = 1 if side == 1 else -1
        fill_price = self.settings.costs.price_with_costs(raw_price, order_side)
        equity = float(cash)
        quantity = self.settings.sizing.quantity(fill_price, equity)
        if quantity <= 0:
            return None, cash

        notional = fill_price * quantity
        commission = self.settings.costs.commission(notional)
        if side == 1:
            cash -= notional + commission
        else:
            cash += notional - commission

        signal_index = max(min(signal_index, len(prices) - 1), 0)
        return (
            Position(
                symbol=self.settings.symbol,
                side=side,
                quantity=quantity,
                entry_price=fill_price,
                entry_index=entry_index,
                signal_index=signal_index,
                entry_date=prices.index[entry_index],
                signal_date=prices.index[signal_index],
                entry_commission=commission,
            ),
            cash,
        )

    def _close_position(
        self,
        position: Position,
        raw_price: float,
        exit_index: int,
        exit_date: pd.Timestamp,
        cash: float,
        reason: str,
        trades: list[dict],
    ) -> tuple[None, float]:
        order_side = -1 if position.side == 1 else 1
        fill_price = self.settings.costs.price_with_costs(raw_price, order_side)
        notional = fill_price * position.quantity
        exit_commission = self.settings.costs.commission(notional)

        if position.side == 1:
            cash += notional - exit_commission
        else:
            cash -= notional + exit_commission

        gross_pnl = position.side * (fill_price - position.entry_price) * position.quantity
        net_pnl = gross_pnl - position.entry_commission - exit_commission
        entry_notional = position.entry_price * position.quantity
        gross_return = gross_pnl / entry_notional if entry_notional else np.nan
        net_return = net_pnl / entry_notional if entry_notional else np.nan
        bars_held = max(exit_index - position.entry_index, 1)

        trades.append(
            {
                "symbol": position.symbol,
                "side": "long" if position.side == 1 else "short",
                "entry_date": position.entry_date,
                "exit_date": exit_date,
                "signal_date": position.signal_date,
                "entry_index": position.entry_index,
                "exit_index": exit_index,
                "entry_price": position.entry_price,
                "exit_price": fill_price,
                "quantity": position.quantity,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "gross_return": gross_return,
                "net_return": net_return,
                "entry_commission": position.entry_commission,
                "exit_commission": exit_commission,
                "bars_held": bars_held,
                "exit_reason": reason,
            }
        )
        return None, cash

    def _position_context(
        self,
        position: Position,
        prices: pd.DataFrame,
        index: int,
    ) -> PositionContext:
        bars_held = max(index - position.entry_index + 1, 1)
        return PositionContext(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            entry_index=position.entry_index,
            signal_index=position.signal_index,
            entry_date=position.entry_date,
            signal_date=position.signal_date,
            bars_held=bars_held,
        )

    def _context(
        self,
        phase: str,
        prepared_exits: dict[str, pd.DataFrame],
        signals: pd.DataFrame,
    ) -> dict:
        return {
            "phase": phase,
            "prepared_exits": prepared_exits,
            "entry_signals": signals,
            "allow_short": self.settings.allow_short,
        }
