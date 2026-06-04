"""Reports for Simple Strategy #1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RESEARCH_WARNINGS = [
    "This Stage 1 run uses the corrected Pine reference settings: long + short enabled.",
    "Compare these numbers against the PineScript result before trusting further tests.",
    "CRWD is a single stock and has limited post-IPO history.",
    "A huge CRWD result may be real behavior or may be selection bias.",
    "Testing many stocks and choosing the best creates selection bias.",
    "Optimization can create data mining bias.",
    "Monte Carlo skip survival does not prove edge.",
    "Permutation p-value is a filter, not proof.",
    "This is not live-trading ready.",
    "This needs incubation / paper tracking.",
]


def write_simple_strategy_01_outputs(
    output_dir: str | Path,
    summary_by_symbol: pd.DataFrame,
    trade_log: pd.DataFrame,
    equity_curves: pd.DataFrame,
    strategy_vs_buy_hold: pd.DataFrame,
    monte_carlo_skip_results: pd.DataFrame,
    monte_carlo_skip_summary: pd.DataFrame,
    trade_sequence_randomization: pd.DataFrame,
    permutation_results: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    optimization_results: pd.DataFrame,
    top_20_results: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    baseline_vs_optimized_summary: pd.DataFrame | None = None,
    quasi_p_values: pd.DataFrame | None = None,
    permutation_1000_results: pd.DataFrame | None = None,
    permutation_1000_summary: pd.DataFrame | None = None,
    quasi_p_values_1000: pd.DataFrame | None = None,
    time_split_validation: pd.DataFrame | None = None,
    direction_mode_comparison: pd.DataFrame | None = None,
    sizing_stress_summary: pd.DataFrame | None = None,
    final_verdict_summary: pd.DataFrame | None = None,
    final_verdict_markdown: str | None = None,
    trade_sequence_randomization_summary: pd.DataFrame | None = None,
    cost_stress_summary: pd.DataFrame | None = None,
    missing_symbols: list[str] | None = None,
    config_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write all requested CSV and HTML artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_by_symbol": out / "summary_by_symbol.csv",
        "key_metrics_summary": out / "key_metrics_summary.csv",
        "trade_log": out / "trade_log.csv",
        "equity_curves": out / "equity_curves.csv",
        "strategy_vs_buy_hold": out / "strategy_vs_buy_hold.csv",
        "cost_stress_summary": out / "cost_stress_summary.csv",
        "monte_carlo_skip_results": out / "monte_carlo_skip_results.csv",
        "monte_carlo_skip_summary": out / "monte_carlo_skip_summary.csv",
        "trade_sequence_randomization": out / "trade_sequence_randomization.csv",
        "trade_sequence_randomization_summary": out / "trade_sequence_randomization_summary.csv",
        "permutation_results": out / "permutation_results.csv",
        "permutation_results_partial": out / "permutation_results_partial.csv",
        "permutation_results_CRWD_partial": out / "permutation_results_CRWD_partial.csv",
        "permutation_results_SHOP_partial": out / "permutation_results_SHOP_partial.csv",
        "permutation_summary": out / "permutation_summary.csv",
        "permutation_1000_results": out / "permutation_1000_results.csv",
        "permutation_1000_summary": out / "permutation_1000_summary.csv",
        "optimization_results": out / "optimization_results.csv",
        "optimization_results_partial": out / "optimization_results_partial.csv",
        "top_20_results": out / "top_20_results.csv",
        "parameter_stability": out / "parameter_stability.csv",
        "baseline_vs_optimized_summary": out / "baseline_vs_optimized_summary.csv",
        "quasi_p_values": out / "quasi_p_values.csv",
        "quasi_p_values_1000": out / "quasi_p_values_1000.csv",
        "time_split_validation": out / "time_split_validation.csv",
        "direction_mode_comparison": out / "direction_mode_comparison.csv",
        "sizing_stress_summary": out / "sizing_stress_summary.csv",
        "final_verdict_summary": out / "final_verdict_summary.csv",
        "final_verdict_markdown": out / "final_verdict_summary.md",
        "html_report": out / "simple_strategy_01_report.html",
        "peer_baseline_summary": out / "peer_baseline_summary.csv",
        "peer_trade_log": out / "peer_trade_log.csv",
        "peer_equity_curves": out / "peer_equity_curves.csv",
    }
    trade_sequence_randomization_summary = (
        trade_sequence_randomization_summary
        if trade_sequence_randomization_summary is not None
        else pd.DataFrame()
    )
    baseline_vs_optimized_summary = (
        baseline_vs_optimized_summary
        if baseline_vs_optimized_summary is not None
        else pd.DataFrame()
    )
    quasi_p_values = quasi_p_values if quasi_p_values is not None else pd.DataFrame()
    permutation_1000_results = (
        permutation_1000_results
        if permutation_1000_results is not None
        else pd.DataFrame()
    )
    permutation_1000_summary = (
        permutation_1000_summary
        if permutation_1000_summary is not None
        else pd.DataFrame()
    )
    quasi_p_values_1000 = (
        quasi_p_values_1000
        if quasi_p_values_1000 is not None
        else pd.DataFrame()
    )
    time_split_validation = (
        time_split_validation
        if time_split_validation is not None
        else pd.DataFrame()
    )
    direction_mode_comparison = (
        direction_mode_comparison
        if direction_mode_comparison is not None
        else pd.DataFrame()
    )
    sizing_stress_summary = (
        sizing_stress_summary
        if sizing_stress_summary is not None
        else pd.DataFrame()
    )
    final_verdict_summary = (
        final_verdict_summary
        if final_verdict_summary is not None
        else pd.DataFrame()
    )
    key_metrics_summary = build_key_metrics_summary(summary_by_symbol, strategy_vs_buy_hold)
    summary_by_symbol.to_csv(paths["summary_by_symbol"], index=False)
    key_metrics_summary.to_csv(paths["key_metrics_summary"], index=False)
    trade_log.to_csv(paths["trade_log"], index=False)
    equity_curves.to_csv(paths["equity_curves"], index=False)
    summary_by_symbol.to_csv(paths["peer_baseline_summary"], index=False)
    trade_log.to_csv(paths["peer_trade_log"], index=False)
    equity_curves.to_csv(paths["peer_equity_curves"], index=False)
    strategy_vs_buy_hold.to_csv(paths["strategy_vs_buy_hold"], index=False)
    cost_stress_summary = cost_stress_summary if cost_stress_summary is not None else pd.DataFrame()
    cost_stress_summary.to_csv(paths["cost_stress_summary"], index=False)
    monte_carlo_skip_results.to_csv(paths["monte_carlo_skip_results"], index=False)
    monte_carlo_skip_summary.to_csv(paths["monte_carlo_skip_summary"], index=False)
    trade_sequence_randomization.to_csv(paths["trade_sequence_randomization"], index=False)
    trade_sequence_randomization_summary.to_csv(
        paths["trade_sequence_randomization_summary"],
        index=False,
    )
    permutation_results.to_csv(paths["permutation_results"], index=False)
    permutation_summary.to_csv(paths["permutation_summary"], index=False)
    permutation_1000_results.to_csv(paths["permutation_1000_results"], index=False)
    permutation_1000_summary.to_csv(paths["permutation_1000_summary"], index=False)
    optimization_results.to_csv(paths["optimization_results"], index=False)
    top_20_results.to_csv(paths["top_20_results"], index=False)
    parameter_stability.to_csv(paths["parameter_stability"], index=False)
    baseline_vs_optimized_summary.to_csv(
        paths["baseline_vs_optimized_summary"],
        index=False,
    )
    quasi_p_values.to_csv(paths["quasi_p_values"], index=False)
    quasi_p_values_1000.to_csv(paths["quasi_p_values_1000"], index=False)
    time_split_validation.to_csv(paths["time_split_validation"], index=False)
    direction_mode_comparison.to_csv(paths["direction_mode_comparison"], index=False)
    sizing_stress_summary.to_csv(paths["sizing_stress_summary"], index=False)
    final_verdict_summary.to_csv(paths["final_verdict_summary"], index=False)
    paths["final_verdict_markdown"].write_text(final_verdict_markdown or "", encoding="utf-8")
    generate_simple_strategy_01_html(
        paths["html_report"],
        summary_by_symbol,
        trade_log,
        equity_curves,
        strategy_vs_buy_hold,
        cost_stress_summary,
        monte_carlo_skip_summary,
        trade_sequence_randomization,
        trade_sequence_randomization_summary,
        permutation_summary,
        quasi_p_values,
        permutation_1000_summary,
        quasi_p_values_1000,
        baseline_vs_optimized_summary,
        top_20_results,
        parameter_stability,
        time_split_validation,
        direction_mode_comparison,
        sizing_stress_summary,
        final_verdict_summary,
        final_verdict_markdown or "",
        missing_symbols or [],
        config_summary or {},
    )
    return paths


def generate_simple_strategy_01_html(
    out_path: str | Path,
    summary_by_symbol: pd.DataFrame,
    trade_log: pd.DataFrame,
    equity_curves: pd.DataFrame,
    strategy_vs_buy_hold: pd.DataFrame,
    cost_stress_summary: pd.DataFrame,
    monte_carlo_skip_summary: pd.DataFrame,
    trade_sequence_randomization: pd.DataFrame,
    trade_sequence_randomization_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    quasi_p_values: pd.DataFrame,
    permutation_1000_summary: pd.DataFrame,
    quasi_p_values_1000: pd.DataFrame,
    baseline_vs_optimized_summary: pd.DataFrame,
    top_20_results: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    time_split_validation: pd.DataFrame,
    direction_mode_comparison: pd.DataFrame,
    sizing_stress_summary: pd.DataFrame,
    final_verdict_summary: pd.DataFrame,
    final_verdict_markdown: str,
    missing_symbols: list[str],
    config_summary: dict[str, Any],
) -> Path:
    """Write a compact HTML report."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    crwd = summary_by_symbol[
        summary_by_symbol.get("symbol", pd.Series(dtype=str)).astype(str).str.upper() == "CRWD"
    ]
    recent_trades = trade_log.copy()
    if not recent_trades.empty and "entry_date" in recent_trades:
        recent_trades = recent_trades.sort_values("entry_date", ascending=False).head(75)
    side_summary = _side_summary(trade_log)
    exit_reason_counts = _exit_reason_counts(trade_log)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simple Strategy 01 Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f6f8fb; color: #17202a; }}
    header {{ background: #121826; color: white; padding: 24px 32px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 18px; margin-bottom: 20px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .muted {{ color: #677489; }}
    .warning li {{ margin-bottom: 6px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th {{ background: #edf2f7; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  </style>
</head>
<body>
  <header>
    <h1>Simple Strategy 01: software_volume_fade_reversal</h1>
    <p class="muted">Daily research lab for high-volume weakness reversal and opposite strength in software/cybersecurity stocks.</p>
    <p><strong>Corrected parity note:</strong> This Stage 1 run uses the corrected Pine reference settings: long + short enabled.</p>
    <p><strong>Stage note:</strong> Compare these numbers against the PineScript result before trusting further tests.</p>
  </header>
  <main>
    <section>
      <h2>CRWD Main Result</h2>
      <div class="table-wrap">{_table(crwd)}</div>
    </section>
    <section>
      <h2>Peer Basket Summary</h2>
      <div class="table-wrap">{_table(summary_by_symbol)}</div>
    </section>
    <section>
      <h2>Strategy vs Buy-and-Hold</h2>
      <div class="table-wrap">{_table(strategy_vs_buy_hold)}</div>
    </section>
    <section>
      <h2>Long / Short Contribution</h2>
      <div class="table-wrap">{_table(side_summary)}</div>
    </section>
    <section>
      <h2>Exit Reason Counts</h2>
      <div class="table-wrap">{_table(exit_reason_counts)}</div>
    </section>
    <section>
      <h2>Execution Cost Stress</h2>
      <div class="table-wrap">{_table(cost_stress_summary)}</div>
    </section>
    <section>
      <h2>Monte Carlo Skip Summary</h2>
      <div class="table-wrap">{_table(monte_carlo_skip_summary)}</div>
    </section>
    <section>
      <h2>Trade Sequence Randomization</h2>
      <div class="table-wrap">{_table(trade_sequence_randomization_summary)}</div>
    </section>
    <section>
      <h2>Stage 6 Baseline vs Optimized Summary</h2>
      <div class="table-wrap">{_table(baseline_vs_optimized_summary)}</div>
    </section>
    <section>
      <h2>Stage 6 Top 20 Optimization Configs</h2>
      <div class="table-wrap">{_table(top_20_results)}</div>
    </section>
    <section>
      <h2>Stage 6 Parameter Stability</h2>
      <div class="table-wrap">{_table(parameter_stability)}</div>
    </section>
    <section>
      <h2>Stage 7 Permutation Summary</h2>
      <div class="table-wrap">{_table(permutation_summary)}</div>
    </section>
    <section>
      <h2>Stage 7 Quasi P-Values</h2>
      <div class="table-wrap">{_table(quasi_p_values)}</div>
    </section>
    <section>
      <h2>CRWD 1000-Permutation Summary</h2>
      <div class="table-wrap">{_table(permutation_1000_summary)}</div>
    </section>
    <section>
      <h2>CRWD 1000-Permutation Quasi P-Values</h2>
      <div class="table-wrap">{_table(quasi_p_values_1000)}</div>
    </section>
    <section>
      <h2>Time-Split Validation</h2>
      <div class="table-wrap">{_table(time_split_validation)}</div>
    </section>
    <section>
      <h2>Direction Mode Comparison</h2>
      <div class="table-wrap">{_table(direction_mode_comparison)}</div>
    </section>
    <section>
      <h2>Sizing Stress</h2>
      <div class="table-wrap">{_table(sizing_stress_summary)}</div>
    </section>
    <section>
      <h2>Final Verdict</h2>
      <pre>{_escape(final_verdict_markdown)}</pre>
      <div class="table-wrap">{_table(final_verdict_summary)}</div>
    </section>
    <section>
      <h2>Recent Trades</h2>
      <div class="table-wrap">{_table(recent_trades)}</div>
    </section>
    <section>
      <h2>Warnings / Limitations</h2>
      <ul class="warning">{''.join(f'<li>{warning}</li>' for warning in RESEARCH_WARNINGS)}</ul>
      <p><strong>Missing symbols:</strong> {', '.join(missing_symbols) if missing_symbols else 'none'}</p>
      <p><strong>Config:</strong> {_escape(str(config_summary))}</p>
    </section>
  </main>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def _table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p>No rows.</p>"
    return df.head(max_rows).to_html(index=False, border=0, escape=True)


def build_key_metrics_summary(
    summary_by_symbol: pd.DataFrame,
    strategy_vs_buy_hold: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact first-look metrics table."""
    if summary_by_symbol.empty:
        return pd.DataFrame()
    columns = [
        "symbol",
        "total_return",
        "cagr",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "number_of_trades",
        "final_equity",
        "long_trade_count",
        "short_trade_count",
        "long_net_pnl",
        "short_net_pnl",
    ]
    out = summary_by_symbol[[col for col in columns if col in summary_by_symbol]].copy()
    if not strategy_vs_buy_hold.empty and "symbol" in strategy_vs_buy_hold:
        compare_cols = [
            col
            for col in ["symbol", "buy_hold_return", "excess_vs_buy_hold"]
            if col in strategy_vs_buy_hold
        ]
        out = out.merge(
            strategy_vs_buy_hold[compare_cols],
            on="symbol",
            how="left",
        )
    return out


def _side_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty or "side" not in trade_log:
        return pd.DataFrame()
    return (
        trade_log.groupby("side", dropna=False)
        .agg(
            trade_count=("side", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_net_return=("net_return", "mean"),
            win_rate=("net_return", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )


def _exit_reason_counts(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty or "exit_reason" not in trade_log:
        return pd.DataFrame()
    return (
        trade_log["exit_reason"]
        .value_counts(dropna=False)
        .rename_axis("exit_reason")
        .reset_index(name="count")
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
