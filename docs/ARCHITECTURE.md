# Architecture Notes

`markov-research` is moving from a single-signal validation project into a broad
statistical market research lab. The old simple Markov model is one legacy lab,
not the center of the architecture.

The existing pipeline still builds a wide research dataset and report. That
functionality should remain stable while the architecture is gradually split
into cleaner panels.

Long-term model families can live as separate labs: visible Markov chains,
Hidden Markov Models, Monte Carlo simulation, Bayesian/MCMC models,
RSI/Bollinger mean reversion, trend following, breakouts, pullbacks, relative
strength, volatility compression/expansion, sector rotation, macro regimes,
path probability, and risk or position-sizing simulations.

## Panel Separation

### state_panel

The `state_panel` should contain only information known at the signal date.

Examples:

- Market regime features available on or before the date.
- Sector regime features available on or before the date.
- Stock state labels available on or before the date.
- Stock personality labels based only on trailing data.
- Technical descriptors such as EMA filters, RSI, Bollinger state, volatility,
  drawdown from past highs, or trend state.

The `state_panel` must not contain future returns, future highs/lows, future
path metrics, or target-before-stop labels.

### transition_panel

The `transition_panel` should contain visible state transition summaries.

Examples:

- Visible Markov transition probabilities between current and next observed
  states.
- Rolling transition counts and sample sizes.
- Transition features by market regime, sector regime, or stock state.
- Later: transition statistics over richer state definitions than the legacy
  bull/sideways/bear return states.

This panel is for studying state movement. It should be built only from current
and historical state observations.

### outcome_panel

The `outcome_panel` should contain future validation labels.

Examples:

- Forward close-to-close returns.
- Excess returns versus SPY, QQQ, sector ETF, and universe average.
- Maximum favorable excursion.
- Maximum adverse excursion.
- Target-before-stop labels.
- Later: realized path/risk labels used for Monte Carlo calibration.

The `outcome_panel` is not a feature source. It exists to evaluate whether
signal-date states or transitions had useful forward information.

### research_panel

The `research_panel` is the validation-only join of state, transition, and
outcome panels.

This is the right place for:

- Report generation.
- Statistical validation.
- Slicing by regime, sector, stock personality, and path outcome.
- Comparing candidate state definitions.
- Exploratory lab summaries.

It is not the right place to define features for a model unless outcome columns
are explicitly excluded.

## Leakage Control

Panel separation matters because the most useful validation labels are also the
most dangerous leakage sources.

Future returns, future high/low excursions, and target-before-stop labels are
computed after the signal date. They are valid outcomes for research, but they
must never influence:

- Market regime labels.
- Sector regime labels.
- Stock state definitions.
- Visible Markov transition inputs.
- Hidden Markov Model training inputs.
- Any future strategy entry filter.

Keeping `state_panel`, `transition_panel`, and `outcome_panel` separate makes it
clear which columns are observable at decision time and which columns are only
available later for validation.

## Migration Approach

Do not rewrite the working pipeline all at once.

Recommended migration path:

1. Keep existing output names and reports stable.
2. Add new state/regime logic in dedicated modules.
3. Build explicit panel outputs in `data/processed` when useful.
4. Keep the current wide dataset as a compatibility `research_panel`.
5. Move reporting and lab analysis out of the monolithic reporting module over
   time.
6. Add HMM and Monte Carlo research only after the basic panels are clean.

## Output Organization

Current research outputs live under `data/results/current/`.

These files are the front door for the new research direction:

- `market_regime_daily.csv`
- `market_regime_summary.csv`
- `sector_regime_daily.csv`
- `sector_regime_summary.csv`
- `research_dashboard.html`

Legacy reference outputs live under `data/results/legacy/`.

These preserve the original simple Markov, RSI/Bollinger, excess-return,
sector/personality, and path-analysis labs:

- `markov_signal_dataset.parquet`
- `markov_signal_dataset.csv`
- `legacy_markov_report.html`
- `summary_*.csv`

Legacy outputs are intentionally kept for comparison and auditability, but new
research layers should not add more files to the legacy area unless they are
part of the old Markov/RSI reference labs.
