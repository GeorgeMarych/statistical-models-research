"""Reports for single strategy labs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reports.html_report import generate_strategy_combo_html


def write_single_strategy_outputs(
    strategy_name: str,
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write CSV and HTML outputs for one strategy definition."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": out / "summary_metrics.csv",
        "trade_log_csv": out / "trade_log.csv",
        "equity_curve_csv": out / "equity_curve.csv",
        "html_report": out / "single_strategy_report.html",
    }
    summary.to_csv(paths["summary_csv"], index=False)
    trades.to_csv(paths["trade_log_csv"], index=False)
    equity.to_csv(paths["equity_curve_csv"], index=False)
    html_summary = summary.rename(columns={"exit_stack": "exit"})
    html_trades = trades.rename(columns={"exit_stack": "exit"})
    html_equity = equity.rename(columns={"exit_stack": "exit"})
    generate_strategy_combo_html(
        html_summary,
        html_trades,
        html_equity,
        paths["html_report"],
        title=f"Single Strategy Report: {strategy_name}",
    )
    return paths
