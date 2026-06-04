# Optimization Workflow

The optimization layer is intentionally small. It is for probing parameter
ranges, not for running huge searches or declaring a strategy robust because one
parameter set looked good.

Run the default optimization:

```bash
python scripts/run_optimization_lab.py
```

The default config is:

```text
config/optimization_lab.yaml
```

Outputs are saved under:

```text
data/results/current/optimization_lab/
```

Files:

- `optimization_results.csv`
- `top_results.csv`
- `parameter_stability.csv`
- `optimization_report.html`

## Parameter Paths

Parameter grids use dotted paths:

```yaml
optimization:
  objective: balanced
  top_n: 20
  parameter_grid:
    entry.length: [10, 20, 30, 40]
    exits.atr_stop.atr_multiple: [2.0, 3.0, 4.0]
    exits.fixed_bars.bars: [5, 10, 15]
```

Supported roots:

- `entry`
- `exits`
- `filters`
- `sizing`
- `costs`
- `strategy`

Exits and filters can be referenced by index or by their configured label.

## Objective

The default objective is `balanced`.

It rewards CAGR and Sharpe, penalizes drawdown, adds a capped profit-factor
bonus, and adds a small sanity bonus for trade count and symbol breadth. It does
not blindly maximize net profit.

Available objectives:

- `balanced`
- `net_profit`
- `cagr`
- `profit_factor`
- `sharpe`
- `return_over_drawdown`

## Stability Checks

The optimizer writes `parameter_stability.csv`, which groups results by each
parameter value and reports:

- mean score
- median score
- best score
- mean CAGR
- mean drawdown
- mean trade count
- share of runs in the top quintile

Healthy candidates should show decent results across neighboring values, not
one isolated spike.

## Overfitting Discipline

Keep grids small. Prefer broad, sensible ranges over dense searches. Do not
optimize every parameter at once. After a promising result, inspect:

- equity curves by symbol
- trade count
- cost sensitivity
- whether one symbol dominates results
- whether neighboring parameters are also acceptable

Walk-forward, Monte Carlo, permutation, and live/paper trading checks are
intentionally not part of this layer yet.
