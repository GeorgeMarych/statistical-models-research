# Strategy Combo Lab

The strategy combo lab is the first modular backtesting layer in this research
repo. It keeps entry logic, exit logic, orchestration, costs, metrics, and
reports separate so new ideas can be tested as combinations:

- Entry A plus Exit A
- Entry A plus Exit B
- Entry B plus Exit A
- Entry B plus Exit C

This is still a research/backtesting framework. It is not a live trading bot
and does not contain brokerage execution.

## Run The Lab

```bash
python scripts/run_strategy_combo_lab.py
```

The default config is:

```text
config/strategy_combo_lab.yaml
```

Outputs are written to:

```text
data/results/current/strategy_combo_lab/
```

Standard outputs:

- `summary.csv`
- `trade_log.csv`
- `equity_curve.csv`
- `strategy_combo_report.html`

## Execution Assumptions

Entry modules generate signals on the signal bar using OHLCV data available on
that bar. The engine fills entry signals on the next bar's open.

Close-based exits also fill on the next bar's open. Intrabar stop and target
exits use levels that are known before or during the bar. If a stop and target
are both touched on the same bar, the stop is treated as occurring first.

The first version supports one position at a time per symbol/run. Multi-symbol
portfolio allocation is intentionally left for a later layer.

## Add A New Entry

1. Create a file in `src/entries/`.
2. Inherit from `BaseEntry` or implement `EntryProtocol`.
3. Return a DataFrame with the same index as the input data and an
   `entry_signal` column:
   - `1` means long entry signal.
   - `-1` means short or opposite-side signal.
   - `0` means no new entry.
4. Use only current-or-prior bars. If you need a breakout over prior highs,
   compute the rolling high and then shift it by one bar.
5. Register the class in `src/backtesting/combinator.py`.
6. Enable it in `config/strategy_combo_lab.yaml`.

Minimal shape:

```python
from dataclasses import dataclass

from src.entries.base import BaseEntry, make_signal_frame


@dataclass
class MyEntry(BaseEntry):
    lookback: int = 20
    label: str = "my_entry"

    @property
    def name(self) -> str:
        return self.label

    @property
    def parameters(self) -> dict:
        return {"lookback": self.lookback}

    def generate_signals(self, data):
        close = data["close"]
        prior_level = close.rolling(self.lookback, min_periods=self.lookback).max().shift(1)
        signal = (close > prior_level).astype(int)
        return make_signal_frame(data, signal)
```

## Add A New Exit

1. Create a file in `src/exits/`.
2. Inherit from `BaseExit` or implement `ExitProtocol`.
3. Use `prepare()` for rolling indicators that should be computed once per run.
4. Use `on_bar()` to return an `ExitDecision`.
5. Register the class in `src/backtesting/combinator.py`.
6. Enable it in `config/strategy_combo_lab.yaml`.

Use `timing="next_open"` for close-based indicator exits and
`timing="intrabar"` only when the exit level is known before the intrabar
breach is checked.

## Current Starter Modules

Entries:

- RSI plus Bollinger Band mean reversion
- Donchian breakout
- Moving average crossover

Exits:

- Fixed bars
- Opposite signal
- Middle Bollinger Band
- Upper Bollinger Band touch/close
- ATR stop
- Take-profit / stop-loss
