"""Parameter stability summaries."""
from __future__ import annotations

import pandas as pd


def build_parameter_stability(
    results: pd.DataFrame,
    parameter_columns: list[str],
    score_column: str = "score",
) -> pd.DataFrame:
    """
    Summarize performance by each individual parameter value.

    This is not a full surface analysis, but it quickly shows whether good
    scores are spread across neighboring values or concentrated in one setting.
    """
    rows: list[dict] = []
    if results.empty:
        return pd.DataFrame()

    top_cutoff = results[score_column].quantile(0.80)
    for parameter in parameter_columns:
        if parameter not in results.columns:
            continue
        grouped = results.groupby(parameter, dropna=False)
        for value, group in grouped:
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "count": len(group),
                    "mean_score": group[score_column].mean(),
                    "median_score": group[score_column].median(),
                    "best_score": group[score_column].max(),
                    "mean_cagr": group["cagr"].mean() if "cagr" in group else None,
                    "mean_max_drawdown": (
                        group["max_drawdown"].mean()
                        if "max_drawdown" in group
                        else None
                    ),
                    "mean_trade_count": (
                        group["number_of_trades"].mean()
                        if "number_of_trades" in group
                        else None
                    ),
                    "top_quintile_share": float((group[score_column] >= top_cutoff).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["parameter", "value"])
    return out
