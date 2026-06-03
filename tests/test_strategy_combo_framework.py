from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine, BacktestSettings, prepare_ohlcv
from src.backtesting.portfolio import PositionSizing
from src.entries import (
    DonchianBreakoutEntry,
    MovingAverageCrossoverEntry,
    RsiBollingerEntry,
)
from src.entries.base import SIGNAL_COLUMN
from src.exits import FixedBarsExit


def synthetic_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=140, freq="B")
    close = np.concatenate(
        [
            np.full(35, 100.0),
            np.linspace(100.0, 130.0, 45),
            np.linspace(130.0, 92.0, 60),
        ]
    )
    open_ = pd.Series(close, index=dates).shift(1).fillna(close[0]).to_numpy()
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000000,
        },
        index=dates,
    )


class StrategyComboFrameworkTests(unittest.TestCase):
    def test_entry_signals_match_input_index(self) -> None:
        data = synthetic_ohlcv()
        entries = [
            RsiBollingerEntry(),
            DonchianBreakoutEntry(lookback=20),
            MovingAverageCrossoverEntry(fast_window=5, slow_window=20),
        ]
        for entry in entries:
            signals = entry.generate_signals(data)
            self.assertTrue(signals.index.equals(data.index), entry.name)
            self.assertIn(SIGNAL_COLUMN, signals.columns, entry.name)
            self.assertTrue(set(signals[SIGNAL_COLUMN].unique()).issubset({-1, 0, 1}))

    def test_donchian_does_not_signal_before_history(self) -> None:
        data = synthetic_ohlcv()
        entry = DonchianBreakoutEntry(lookback=20)
        signals = entry.generate_signals(data)
        self.assertTrue((signals[SIGNAL_COLUMN].iloc[:20] == 0).all())

    def test_prepare_ohlcv_drops_future_columns(self) -> None:
        data = synthetic_ohlcv()
        data["future_20d_return"] = data["close"].shift(-20) / data["close"] - 1.0
        prepared = prepare_ohlcv(data)
        self.assertNotIn("future_20d_return", prepared.columns)
        self.assertEqual(list(prepared.columns), ["open", "high", "low", "close", "volume"])

    def test_backtest_trade_log_and_summary_when_trades_occur(self) -> None:
        data = synthetic_ohlcv()
        entry = MovingAverageCrossoverEntry(fast_window=5, slow_window=20)
        exit_module = FixedBarsExit(max_bars=8)
        settings = BacktestSettings(
            symbol="TEST",
            initial_capital=100000,
            allow_short=False,
            sizing=PositionSizing(mode="percent_equity", value=0.95),
        )
        result = BacktestEngine(settings).run(data, entry, exit_module)
        self.assertGreater(result.summary["number_of_trades"], 0)
        self.assertFalse(result.trades.empty)
        self.assertFalse(result.equity_curve.empty)
        self.assertGreaterEqual(result.trades["entry_index"].min(), entry.min_lookback)


if __name__ == "__main__":
    unittest.main()
