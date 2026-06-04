"""Reports for optimization labs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_optimization_outputs(
    results: pd.DataFrame,
    top_results: pd.DataFrame,
    stability: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write optimization CSVs and HTML report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "optimization_results_csv": out / "optimization_results.csv",
        "top_results_csv": out / "top_results.csv",
        "parameter_stability_csv": out / "parameter_stability.csv",
        "html_report": out / "optimization_report.html",
    }
    results.to_csv(paths["optimization_results_csv"], index=False)
    top_results.to_csv(paths["top_results_csv"], index=False)
    stability.to_csv(paths["parameter_stability_csv"], index=False)
    _write_html(results, top_results, stability, paths["html_report"])
    return paths


def _write_html(
    results: pd.DataFrame,
    top_results: pd.DataFrame,
    stability: pd.DataFrame,
    out_path: Path,
) -> None:
    plot_html = _score_plot(results)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Optimization Report</title>
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
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      margin-bottom: 24px;
      padding: 20px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    th {{
      background: #eef2f7;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Optimization Report</h1>
    <p>Small-grid strategy parameter research with stability checks.</p>
  </header>
  <main>
    <section>{plot_html}</section>
    <section>
      <h2>Top Results</h2>
      <div class="table-wrap">{top_results.to_html(index=False, border=0, escape=True)}</div>
    </section>
    <section>
      <h2>Parameter Stability</h2>
      <div class="table-wrap">{stability.to_html(index=False, border=0, escape=True)}</div>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _score_plot(results: pd.DataFrame) -> str:
    if results.empty:
        return "<p>No optimization results.</p>"
    try:
        import plotly.express as px
    except Exception:
        return "<p>Plotly unavailable; see optimization_results.csv.</p>"
    frame = results.sort_values("optimization_run")
    fig = px.scatter(
        frame,
        x="optimization_run",
        y="score",
        hover_data=["parameters", "cagr", "max_drawdown", "number_of_trades"],
        title="Objective Score By Parameter Set",
    )
    fig.update_layout(template="plotly_white", height=440)
    return fig.to_html(full_html=False, include_plotlyjs=True)
