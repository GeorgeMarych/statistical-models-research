"""Compact HTML dashboard for the current regime research layer."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.regime_diagnostics import summarize_transition_matrix


def generate_research_dashboard_html(
    market_panel: pd.DataFrame,
    market_summary: pd.DataFrame,
    sector_panel: pd.DataFrame,
    sector_summary: pd.DataFrame,
    missing_symbols: dict[str, str],
    out_path: str | Path,
    market_diagnostics: pd.DataFrame | None = None,
    sector_diagnostics: pd.DataFrame | None = None,
    market_transition_5d: pd.DataFrame | None = None,
    market_transition_20d: pd.DataFrame | None = None,
    sector_transition_5d: pd.DataFrame | None = None,
    sector_transition_20d: pd.DataFrame | None = None,
) -> Path:
    """Write a compact regime-focused research dashboard."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    market_counts = _counts_table(market_panel, "market_regime")
    sector_counts = _counts_table(sector_panel, "sector_regime")
    latest_market = _latest_label(market_panel, "market_regime")
    latest_sector_counts = _latest_sector_counts(sector_panel)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Research Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f6f7f9;
    }}
    header {{
      background: #12212f;
      color: white;
      padding: 28px 36px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1, h2 {{
      margin: 0 0 10px;
    }}
    h1 {{
      font-size: 30px;
      font-weight: 700;
    }}
    h2 {{
      font-size: 20px;
      margin-top: 28px;
    }}
    p {{
      line-height: 1.5;
    }}
    .muted {{
      color: #66788a;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      padding: 16px;
    }}
    .label {{
      color: #66788a;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .value {{
      font-size: 22px;
      font-weight: 700;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      overflow: hidden;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #edf1f5;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #eef3f7;
      font-weight: 650;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .warning {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #7c2d12;
      border-radius: 8px;
      padding: 12px 14px;
      margin-top: 12px;
    }}
    code {{
      background: #eef3f7;
      padding: 1px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Research Dashboard</h1>
    <p>Compact regime foundation dashboard. Generated {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}.</p>
  </header>
  <main>
    <p class="muted">
      This dashboard is focused on the new market/sector regime layer. The old
      <code>legacy_markov_report.html</code> remains a legacy report for the
      simple visible Markov, RSI/Bollinger, sector/personality, and path-analysis labs.
    </p>

    <div class="grid">
      <div class="card">
        <div class="label">Market Rows</div>
        <div class="value">{len(market_panel):,}</div>
      </div>
      <div class="card">
        <div class="label">Market Date Range</div>
        <div class="value">{_date_range(market_panel)}</div>
      </div>
      <div class="card">
        <div class="label">Latest Market Regime</div>
        <div class="value">{html.escape(latest_market)}</div>
      </div>
      <div class="card">
        <div class="label">Sector Rows</div>
        <div class="value">{len(sector_panel):,}</div>
      </div>
    </div>

    <h2>Market Regime Counts</h2>
    {_table_html(market_counts)}

    <h2>Market Forward Returns By Regime</h2>
    {_table_html(_format_numeric_table(market_summary))}

    <h2>Sector Regime Counts</h2>
    {_table_html(sector_counts)}

    <h2>Latest Sector Regimes</h2>
    {_table_html(latest_sector_counts)}

    <h2>Sector Forward Returns By Regime</h2>
    {_table_html(_format_numeric_table(sector_summary))}

    <h2>Regime Diagnostics</h2>
    <p class="muted">
      Diagnostics validate labels against future outcomes and transitions. They
      are not used by the regime label rules.
    </p>
    <h3>Market Forward Return Diagnostics</h3>
    {_table_html(_format_numeric_table(_market_diagnostics_table(market_diagnostics)))}
    <h3>Sector Forward Return Diagnostics</h3>
    {_table_html(_format_numeric_table(_sector_diagnostics_table(sector_diagnostics)))}
    <h3>Market 5D Transition Summary</h3>
    {_table_html(_format_numeric_table(_transition_summary_table(market_transition_5d)))}
    <h3>Market 20D Transition Summary</h3>
    {_table_html(_format_numeric_table(_transition_summary_table(market_transition_20d)))}
    <h3>Sector 5D Transition Summary</h3>
    {_table_html(_format_numeric_table(_transition_summary_table(sector_transition_5d)))}
    <h3>Sector 20D Transition Summary</h3>
    {_table_html(_format_numeric_table(_transition_summary_table(sector_transition_20d)))}

    <h2>Missing Or Unavailable Inputs</h2>
    {_missing_html(missing_symbols)}
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def _date_range(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return "n/a"
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    return f"{dates.min().date()} to {dates.max().date()}"


def _latest_label(df: pd.DataFrame, label_col: str) -> str:
    if df.empty or "date" not in df.columns or label_col not in df.columns:
        return "n/a"
    ordered = df.sort_values("date").dropna(subset=[label_col])
    if ordered.empty:
        return "n/a"
    row = ordered.iloc[-1]
    return f"{row[label_col]} ({pd.to_datetime(row['date']).date()})"


def _counts_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(columns=[label_col, "rows", "share"])
    counts = df[label_col].value_counts(dropna=False).rename_axis(label_col).reset_index(name="rows")
    counts["share"] = counts["rows"] / counts["rows"].sum()
    return _format_numeric_table(counts)


def _latest_sector_counts(sector_panel: pd.DataFrame) -> pd.DataFrame:
    if sector_panel.empty or not {"date", "sector_etf", "sector_regime"}.issubset(sector_panel.columns):
        return pd.DataFrame(columns=["sector_regime", "rows", "share"])
    ordered = sector_panel.sort_values("date")
    latest_date = ordered["date"].max()
    latest = ordered[ordered["date"] == latest_date]
    return _counts_table(latest, "sector_regime")


def _format_numeric_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if (
            "return" in col
            or "rate" in col
            or "probability" in col
            or col == "share"
            or col.startswith("pct_")
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x * 100:.2f}%"
            )
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="muted">No data available.</p>'
    return df.to_html(index=False, escape=True, border=0)


def _market_diagnostics_table(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [
        "market_regime",
        "rows",
        "pct_total_rows",
        "spy_avg_forward_5d_return",
        "spy_avg_forward_10d_return",
        "spy_avg_forward_20d_return",
        "spy_forward_20d_win_rate",
        "qqq_avg_forward_20d_return",
        "rsp_avg_forward_20d_return",
    ]
    return _select_existing(df, cols)


def _sector_diagnostics_table(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [
        "sector_regime",
        "rows",
        "pct_total_rows",
        "sector_avg_forward_5d_return",
        "sector_avg_forward_10d_return",
        "sector_avg_forward_20d_return",
        "sector_forward_20d_win_rate",
    ]
    return _select_existing(df, cols)


def _transition_summary_table(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [
        "current_regime",
        "transition_rows",
        "self_transition_probability",
        "most_likely_future_regime",
        "most_likely_future_probability",
    ]
    return _select_existing(summarize_transition_matrix(df), cols)


def _select_existing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [col for col in cols if col in df.columns]
    return df[existing].copy() if existing else pd.DataFrame()


def _missing_html(missing_symbols: dict[str, str]) -> str:
    if not missing_symbols:
        return '<p class="muted">No missing configured symbols.</p>'
    rows = pd.DataFrame(
        [{"symbol": symbol, "reason": reason} for symbol, reason in missing_symbols.items()]
    )
    return '<div class="warning">Some configured inputs were unavailable; the dashboard continued with available data.</div>' + _table_html(rows)
