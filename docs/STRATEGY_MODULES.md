# Strategy Modules

This project now supports complete modular research strategies built from:

- one entry module
- zero or more filters
- one exit stack containing one or more exits
- position sizing settings
- cost and slippage settings

It remains a research/backtesting lab. There is no live trading, broker
execution, webhook handling, or paper-trading bridge in this layer.

## Entries

Entries live in `src/entries/`.

An entry receives an OHLCV DataFrame and returns a DataFrame with the same index
and an `entry_signal` column:

- `1` means long signal.
- `-1` means short or opposite-side signal.
- `0` means no entry.

Entries should use only current-or-prior bars. For breakout levels, compute the
rolling level and shift it by one bar when the level must have been known before
the signal bar.

Starter entries include:

- `close_breakout`
- `stop_breakout`
- `mean_reversion`
- `rsi_bb_mean_reversion`
- `donchian_breakout`
- `ma_crossover`
- `dueling_momentum`
- `price_pattern`

## Filters

Filters live in `src/filters/`.

A filter returns `allow_long` and `allow_short` boolean masks aligned with the
input data. Strategy definitions combine filters with `filter_mode: all` by
default, meaning every filter must pass for an entry to be allowed.

Starter filters include:

- trend filter
- volatility filter
- volume filter
- market regime filter
- day-of-week filter

## Exits And Exit Stacks

Exits live in `src/exits/`.

An exit can trigger intrabar when its level is known, or at the next open after
a close-based signal. An `ExitStack` is an ordered list of exits. The engine
evaluates them in order and the first triggered exit closes the trade.

Put emergency exits first:

```python
ExitStack([
    ATRStopExit(atr_length=14, atr_multiple=3.0),
    ProfitTargetExit(take_profit_pct=0.06),
    FixedBarsExit(bars=10),
])
```

Starter exits include:

- fixed bars
- opposite signal
- stop loss
- profit target
- ATR stop
- trailing profit
- Bollinger exits
- combined take-profit / stop-loss

## Strategy Definitions

Strategy definitions live in `src/strategies/`.

A `StrategyDefinition` represents one complete strategy:

- `name`
- `symbols`
- `direction_mode`: `long_only`, `short_only`, or `long_short`
- one entry
- one exit stack
- zero or more filters
- sizing
- costs

The CLI config format is shown in `config/single_strategy_lab.yaml`.

Run a single strategy:

```bash
python scripts/run_single_strategy_lab.py
```

Outputs are saved under:

```text
data/results/current/single_strategy_lab/
```

## Add A Module

To add an entry, exit, or filter:

1. Add a small class in the matching package.
2. Keep parameters serializable.
3. Register the class in `src/backtesting/combinator.py`.
4. Add it to a YAML strategy config.
5. Add one focused test if the logic has branching behavior.

## Backtest Assumptions

- Entry signal generated on the signal bar.
- Default fill is next open.
- Stop/target exits may trigger intrabar if the level is known.
- One position at a time per symbol.
- Same-bar stop/target ambiguity is conservative: stop first when using the
  combined target/stop exit.
- Trade logs include dates, prices, PnL, return, bars held, symbol, strategy,
  and exit reason.
