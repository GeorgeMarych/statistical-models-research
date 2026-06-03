"""HTML report generation for strategy combination runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _format_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    table = df.copy()
    if columns is not None:
        table = table[[col for col in columns if col in table.columns]]
    return table.to_html(index=False, classes="data-table", border=0, escape=True)


def _equity_plot_html(summary: pd.DataFrame, equity: pd.DataFrame, top_n: int = 10) -> str:
    if summary.empty or equity.empty or "run_id" not in equity.columns:
        return "<p>No equity curves available.</p>"
    try:
        import plotly.graph_objects as go
    except Exception:
        return "<p>Plotly is unavailable; see equity_curve.csv for curves.</p>"

    top = summary.sort_values("total_return", ascending=False).head(top_n)
    run_ids = top["run_id"].tolist()
    plot_data = equity[equity["run_id"].isin(run_ids)].copy()
    if plot_data.empty:
        return "<p>No equity curves available.</p>"

    fig = go.Figure()
    for run_id, group in plot_data.groupby("run_id", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["equity"],
                mode="lines",
                name=str(run_id),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 40, "r": 20, "t": 35, "b": 35},
        title=f"Top {min(top_n, len(run_ids))} Equity Curves by Total Return",
        xaxis_title="Date",
        yaxis_title="Equity",
        legend_title="Run",
    )
    return fig.to_html(full_html=False, include_plotlyjs=True)


def generate_strategy_combo_html(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    out_path: str | Path,
    title: str = "Strategy Combo Lab",
) -> Path:
    """Write a self-contained HTML dashboard."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary_sorted = summary.copy()
    if not summary_sorted.empty and "total_return" in summary_sorted.columns:
        summary_sorted = summary_sorted.sort_values("total_return", ascending=False)

    summary_cols = [
        "run_id",
        "symbol",
        "entry",
        "exit",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "win_rate",
        "expectancy_pct",
        "number_of_trades",
        "exposure_time",
        "average_bars_held",
        "best_trade",
        "worst_trade",
    ]
    trade_cols = [
        "run_id",
        "symbol",
        "side",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "net_return",
        "net_pnl",
        "bars_held",
        "exit_reason",
    ]

    top_trades = trades.copy()
    if not top_trades.empty and "entry_date" in top_trades.columns:
        top_trades = top_trades.sort_values("entry_date", ascending=False).head(100)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 0;
      color: #1f2933;
      background: #f5f7fa;
    }}
    header {{
      background: #111827;
      color: white;
      padding: 24px 32px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    section {{
      margin-bottom: 28px;
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 20px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .muted {{
      color: #6b7280;
      margin: 0;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table.data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    table.data-table th, table.data-table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    table.data-table th:first-child, table.data-table td:first-child,
    table.data-table th:nth-child(2), table.data-table td:nth-child(2),
    table.data-table th:nth-child(3), table.data-table td:nth-child(3),
    table.data-table th:nth-child(4), table.data-table td:nth-child(4) {{
      text-align: left;
    }}
    table.data-table th {{
      background: #eef2f7;
      color: #111827;
      position: sticky;
      top: 0;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="muted">Modular entry/exit research backtests. Entries are filled on the next open after the signal bar.</p>
  </header>
  <main>
    <section>
      {_equity_plot_html(summary_sorted, equity)}
    </section>
    <section>
      <h2>Summary</h2>
      <div class="table-wrap">{_format_table(summary_sorted, summary_cols)}</div>
    </section>
    <section>
      <h2>Recent Trades</h2>
      <div class="table-wrap">{_format_table(top_trades, trade_cols)}</div>
    </section>
  </main>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out
