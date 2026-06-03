"""CSV and HTML writers for strategy combination research."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reports.html_report import generate_strategy_combo_html


def write_strategy_combo_outputs(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write standard strategy lab outputs and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_csv": out / "summary.csv",
        "trade_log_csv": out / "trade_log.csv",
        "equity_curve_csv": out / "equity_curve.csv",
        "html_report": out / "strategy_combo_report.html",
    }
    summary.to_csv(paths["summary_csv"], index=False)
    trades.to_csv(paths["trade_log_csv"], index=False)
    equity.to_csv(paths["equity_curve_csv"], index=False)
    generate_strategy_combo_html(summary, trades, equity, paths["html_report"])
    return paths
