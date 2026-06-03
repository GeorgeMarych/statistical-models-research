"""Diagnostics for current market and sector regime research outputs.

These helpers validate existing regime labels against future outcomes. They do
not participate in regime classification and should remain outcome-side
research utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MARKET_REGIME_ORDER = [
    "risk_on_broad",
    "risk_on_narrow",
    "neutral",
    "fragile",
    "risk_off",
    "panic",
]
SECTOR_REGIME_ORDER = ["strong", "improving", "neutral", "weakening", "weak"]
DEFAULT_FORWARD_DAYS = [5, 10, 20]
DEFAULT_TRANSITION_DAYS = [1, 5, 20]


@dataclass
class RegimeDiagnosticsResult:
    """Container for regime diagnostic tables."""

    market_diagnostics: pd.DataFrame
    sector_diagnostics: pd.DataFrame
    market_transitions: dict[int, pd.DataFrame]
    sector_transitions: dict[int, pd.DataFrame]


def build_regime_diagnostics(
    market_panel: pd.DataFrame,
    sector_panel: pd.DataFrame,
    forward_days: list[int] | None = None,
    transition_days: list[int] | None = None,
) -> RegimeDiagnosticsResult:
    """Build all current-regime diagnostic tables from state panels."""
    forward_days = forward_days or DEFAULT_FORWARD_DAYS
    transition_days = transition_days or DEFAULT_TRANSITION_DAYS

    market_diagnostics = build_market_regime_diagnostics(market_panel, forward_days)
    sector_diagnostics = build_sector_regime_diagnostics(sector_panel, forward_days)
    market_transitions = {
        days: build_transition_matrix(
            market_panel,
            label_col="market_regime",
            days=days,
            label_order=MARKET_REGIME_ORDER,
        )
        for days in transition_days
    }
    sector_transitions = {
        days: build_transition_matrix(
            sector_panel,
            label_col="sector_regime",
            days=days,
            label_order=SECTOR_REGIME_ORDER,
            group_col="sector_etf",
        )
        for days in transition_days
    }

    return RegimeDiagnosticsResult(
        market_diagnostics=market_diagnostics,
        sector_diagnostics=sector_diagnostics,
        market_transitions=market_transitions,
        sector_transitions=sector_transitions,
    )


def write_regime_diagnostics(
    result: RegimeDiagnosticsResult,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write diagnostic CSVs and return their output paths."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["market_regime_diagnostics"] = out_path / "market_regime_diagnostics.csv"
    paths["sector_regime_diagnostics"] = out_path / "sector_regime_diagnostics.csv"
    result.market_diagnostics.to_csv(paths["market_regime_diagnostics"], index=False)
    result.sector_diagnostics.to_csv(paths["sector_regime_diagnostics"], index=False)

    for days, matrix in result.market_transitions.items():
        key = f"market_regime_transition_{days}d"
        paths[key] = out_path / f"{key}.csv"
        matrix.to_csv(paths[key], index=False)

    for days, matrix in result.sector_transitions.items():
        key = f"sector_regime_transition_{days}d"
        paths[key] = out_path / f"{key}.csv"
        matrix.to_csv(paths[key], index=False)

    return paths


def load_current_regime_outputs(current_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current market and sector regime panels from CSV outputs."""
    current_path = Path(current_dir)
    market = pd.read_csv(current_path / "market_regime_daily.csv", parse_dates=["date"])
    sector = pd.read_csv(current_path / "sector_regime_daily.csv", parse_dates=["date"])
    return market, sector


def build_market_regime_diagnostics(
    market_panel: pd.DataFrame,
    days: list[int] | None = None,
) -> pd.DataFrame:
    """Summarize forward index returns by market regime."""
    days = days or DEFAULT_FORWARD_DAYS
    if market_panel.empty or "market_regime" not in market_panel.columns:
        return pd.DataFrame()

    panel = market_panel.sort_values("date").copy()
    total_rows = len(panel)

    for symbol in ["SPY", "QQQ", "RSP"]:
        prefix = symbol.lower()
        close_col = f"{prefix}_close"
        if close_col not in panel.columns:
            continue
        close = pd.to_numeric(panel[close_col], errors="coerce")
        for day in days:
            panel[f"{prefix}_forward_{day}d_return"] = close.shift(-day) / close - 1.0

    rows: list[dict[str, Any]] = []
    for regime in _ordered_labels(panel["market_regime"], MARKET_REGIME_ORDER):
        subset = panel[panel["market_regime"] == regime]
        row: dict[str, Any] = {
            "market_regime": regime,
            "rows": len(subset),
            "pct_total_rows": len(subset) / total_rows if total_rows else np.nan,
        }
        for symbol in ["SPY", "QQQ", "RSP"]:
            prefix = symbol.lower()
            for day in days:
                col = f"{prefix}_forward_{day}d_return"
                if col not in subset.columns:
                    continue
                values = pd.to_numeric(subset[col], errors="coerce").dropna()
                row[f"{prefix}_avg_forward_{day}d_return"] = _mean(values)
                row[f"{prefix}_median_forward_{day}d_return"] = _median(values)
                row[f"{prefix}_forward_{day}d_win_rate"] = _win_rate(values)

        spy20 = pd.to_numeric(
            subset.get("spy_forward_20d_return", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        row["spy_worst_forward_20d_return"] = float(spy20.min()) if len(spy20) else np.nan
        row["spy_best_forward_20d_return"] = float(spy20.max()) if len(spy20) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_sector_regime_diagnostics(
    sector_panel: pd.DataFrame,
    days: list[int] | None = None,
) -> pd.DataFrame:
    """Summarize forward sector ETF returns by sector regime."""
    days = days or DEFAULT_FORWARD_DAYS
    if sector_panel.empty or "sector_regime" not in sector_panel.columns:
        return pd.DataFrame()

    panel = sector_panel.sort_values(["sector_etf", "date"]).copy()
    total_rows = len(panel)
    for day in days:
        panel[f"forward_{day}d_return"] = (
            panel.groupby("sector_etf")["close"].shift(-day) / panel["close"] - 1.0
        )

    rows: list[dict[str, Any]] = []
    for regime in _ordered_labels(panel["sector_regime"], SECTOR_REGIME_ORDER):
        subset = panel[panel["sector_regime"] == regime]
        row: dict[str, Any] = {
            "sector_regime": regime,
            "rows": len(subset),
            "pct_total_rows": len(subset) / total_rows if total_rows else np.nan,
            "sector_count": subset["sector_etf"].nunique()
            if "sector_etf" in subset.columns
            else np.nan,
        }
        for day in days:
            col = f"forward_{day}d_return"
            values = pd.to_numeric(subset[col], errors="coerce").dropna()
            row[f"sector_avg_forward_{day}d_return"] = _mean(values)
            row[f"sector_median_forward_{day}d_return"] = _median(values)
            row[f"sector_forward_{day}d_win_rate"] = _win_rate(values)

        fwd20 = pd.to_numeric(
            subset.get("forward_20d_return", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        row["sector_worst_forward_20d_return"] = float(fwd20.min()) if len(fwd20) else np.nan
        row["sector_best_forward_20d_return"] = float(fwd20.max()) if len(fwd20) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_transition_matrix(
    panel: pd.DataFrame,
    label_col: str,
    days: int,
    label_order: list[str],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Build a count/probability matrix for current label to future label."""
    if panel.empty or label_col not in panel.columns:
        return pd.DataFrame()

    sort_cols = [group_col, "date"] if group_col and group_col in panel.columns else ["date"]
    work = panel.sort_values(sort_cols).copy()
    future_col = "_future_regime"
    if group_col and group_col in work.columns:
        work[future_col] = work.groupby(group_col)[label_col].shift(-days)
    else:
        work[future_col] = work[label_col].shift(-days)

    work = work.dropna(subset=[label_col, future_col])
    if work.empty:
        return pd.DataFrame()

    labels = _ordered_labels(
        pd.concat([work[label_col], work[future_col]], ignore_index=True),
        label_order,
    )
    counts = pd.crosstab(work[label_col], work[future_col]).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )
    totals = counts.sum(axis=1)
    probabilities = counts.div(totals.replace(0, np.nan), axis=0)

    rows: list[dict[str, Any]] = []
    for current in labels:
        row: dict[str, Any] = {
            "current_regime": current,
            "future_horizon_days": days,
            "transition_rows": int(totals.loc[current]),
        }
        for future in labels:
            safe_future = _safe_label(future)
            row[f"to_{safe_future}_count"] = int(counts.loc[current, future])
            probability = probabilities.loc[current, future]
            row[f"to_{safe_future}_probability"] = (
                float(probability) if not pd.isna(probability) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_transition_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return compact transition summary rows for dashboard and console output."""
    if matrix.empty or "current_regime" not in matrix.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    prob_cols = [col for col in matrix.columns if col.startswith("to_") and col.endswith("_probability")]
    for _, row in matrix.iterrows():
        current = row["current_regime"]
        probabilities = {
            _unsafed_label(col.removeprefix("to_").removesuffix("_probability")): row[col]
            for col in prob_cols
            if not pd.isna(row[col])
        }
        if probabilities:
            most_likely, most_likely_prob = max(probabilities.items(), key=lambda item: item[1])
        else:
            most_likely, most_likely_prob = "", np.nan

        rows.append(
            {
                "current_regime": current,
                "transition_rows": row.get("transition_rows", np.nan),
                "self_transition_probability": probabilities.get(current, np.nan),
                "most_likely_future_regime": most_likely,
                "most_likely_future_probability": most_likely_prob,
            }
        )

    return pd.DataFrame(rows)


def build_regime_diagnostics_console_summary(
    market_diagnostics: pd.DataFrame,
    sector_diagnostics: pd.DataFrame,
    market_transition_1d: pd.DataFrame,
    market_transition_5d: pd.DataFrame,
    sector_transition_1d: pd.DataFrame,
    sector_transition_5d: pd.DataFrame,
    missing_symbols: dict[str, str] | None = None,
) -> list[str]:
    """Answer the main diagnostic questions in compact console-ready lines."""
    lines: list[str] = []

    broad = _label_value(market_diagnostics, "market_regime", "risk_on_broad", "spy_avg_forward_20d_return")
    neutral = _label_value(market_diagnostics, "market_regime", "neutral", "spy_avg_forward_20d_return")
    fragile = _label_value(market_diagnostics, "market_regime", "fragile", "spy_avg_forward_20d_return")
    risk_off = _label_value(market_diagnostics, "market_regime", "risk_off", "spy_avg_forward_20d_return")
    comparison = [value for value in [neutral, fragile, risk_off] if not pd.isna(value)]
    if pd.isna(broad) or not comparison:
        lines.append("1. risk_on_broad comparison: insufficient rows.")
    else:
        verdict = "yes" if broad > max(comparison) else "no/mixed"
        lines.append(
            "1. risk_on_broad vs neutral/fragile/risk_off: "
            f"{verdict} on SPY 20D avg "
            f"({ _fmt_pct(broad) } vs neutral { _fmt_pct(neutral) }, "
            f"fragile { _fmt_pct(fragile) }, risk_off { _fmt_pct(risk_off) })."
        )

    broad_rsp = _label_value(market_diagnostics, "market_regime", "risk_on_broad", "rsp_avg_forward_20d_return")
    narrow_rsp = _label_value(market_diagnostics, "market_regime", "risk_on_narrow", "rsp_avg_forward_20d_return")
    if pd.isna(broad_rsp) or pd.isna(narrow_rsp):
        lines.append("2. risk_on_narrow vs risk_on_broad: insufficient RSP rows.")
    else:
        verdict = "worse" if narrow_rsp < broad_rsp else "not worse"
        lines.append(
            "2. risk_on_narrow vs risk_on_broad: "
            f"{verdict} for RSP 20D avg ({_fmt_pct(narrow_rsp)} vs {_fmt_pct(broad_rsp)})."
        )

    panic_rows = _label_value(market_diagnostics, "market_regime", "panic", "rows")
    panic_spy = _label_value(market_diagnostics, "market_regime", "panic", "spy_avg_forward_20d_return")
    panic_win = _label_value(market_diagnostics, "market_regime", "panic", "spy_forward_20d_win_rate")
    if pd.isna(panic_rows) or panic_rows == 0:
        lines.append("3. panic: no panic rows in the current sample, so no danger/rebound read yet.")
    else:
        behavior = "rebound-leaning" if panic_spy > 0 and panic_win >= 0.5 else "danger-continuation leaning"
        lines.append(
            "3. panic: "
            f"{behavior} (SPY 20D avg {_fmt_pct(panic_spy)}, win rate {_fmt_pct(panic_win)}, rows {int(panic_rows)})."
        )

    sector_order = [
        (
            label,
            _label_value(
                sector_diagnostics,
                "sector_regime",
                label,
                "sector_avg_forward_20d_return",
            ),
        )
        for label in SECTOR_REGIME_ORDER
    ]
    available_sector = [(label, value) for label, value in sector_order if not pd.isna(value)]
    if len(available_sector) < 2:
        lines.append("4. sector ordering: insufficient rows.")
    else:
        logical = all(
            available_sector[i][1] >= available_sector[i + 1][1]
            for i in range(len(available_sector) - 1)
        )
        verdict = "yes" if logical else "no/mixed"
        order_text = ", ".join(f"{label} {_fmt_pct(value)}" for label, value in available_sector)
        lines.append(f"4. sector ordering by 20D avg: {verdict} ({order_text}).")

    market_stick_1d = _weighted_self_transition(summarize_transition_matrix(market_transition_1d))
    market_stick_5d = _weighted_self_transition(summarize_transition_matrix(market_transition_5d))
    sector_stick_1d = _weighted_self_transition(summarize_transition_matrix(sector_transition_1d))
    sector_stick_5d = _weighted_self_transition(summarize_transition_matrix(sector_transition_5d))
    stickiness = _stickiness_verdict(market_stick_1d, market_stick_5d, sector_stick_1d, sector_stick_5d)
    lines.append(
        "5. label stickiness/noise: "
        f"{stickiness} (market self 1D {_fmt_pct(market_stick_1d)}, 5D {_fmt_pct(market_stick_5d)}; "
        f"sector self 1D {_fmt_pct(sector_stick_1d)}, 5D {_fmt_pct(sector_stick_5d)})."
    )

    lines.append("6. suspicious distributions: " + _distribution_notes(market_diagnostics, sector_diagnostics))
    lines.append("7. missing inputs to fix first: " + _missing_priority_note(missing_symbols or {}))
    return lines


def _ordered_labels(series: pd.Series, preferred_order: list[str]) -> list[str]:
    values = [str(value) for value in series.dropna().unique()]
    ordered = [label for label in preferred_order if label in values]
    ordered.extend(sorted(label for label in values if label not in ordered))
    return ordered


def _safe_label(label: Any) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _unsafed_label(label: str) -> str:
    return label


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def _median(values: pd.Series) -> float:
    return float(values.median()) if len(values) else np.nan


def _win_rate(values: pd.Series) -> float:
    return float((values > 0).mean()) if len(values) else np.nan


def _label_value(df: pd.DataFrame, label_col: str, label: str, value_col: str) -> float:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return np.nan
    subset = df[df[label_col] == label]
    if subset.empty:
        return np.nan
    value = pd.to_numeric(subset.iloc[0:1][value_col], errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def _weighted_self_transition(summary: pd.DataFrame) -> float:
    if summary.empty or "self_transition_probability" not in summary.columns:
        return np.nan
    weights = pd.to_numeric(summary.get("transition_rows"), errors="coerce").fillna(0)
    values = pd.to_numeric(summary["self_transition_probability"], errors="coerce")
    valid = values.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _stickiness_verdict(
    market_1d: float,
    market_5d: float,
    sector_1d: float,
    sector_5d: float,
) -> str:
    values = [market_1d, market_5d, sector_1d, sector_5d]
    if all(pd.isna(value) for value in values):
        return "insufficient transition data"
    if any((not pd.isna(value)) and value < 0.35 for value in [market_1d, sector_1d]):
        return "potentially noisy"
    if any((not pd.isna(value)) and value > 0.95 for value in [market_1d, sector_1d]):
        return "very sticky at 1D"
    if any((not pd.isna(value)) and value > 0.80 for value in [market_5d, sector_5d]):
        return "sticky beyond one week"
    return "reasonable daily persistence"


def _distribution_notes(market_diagnostics: pd.DataFrame, sector_diagnostics: pd.DataFrame) -> str:
    notes: list[str] = []
    if not market_diagnostics.empty:
        present = set(market_diagnostics["market_regime"].astype(str))
        missing = [label for label in MARKET_REGIME_ORDER if label not in present]
        if missing:
            notes.append("missing market labels " + ", ".join(missing))
        small = market_diagnostics[
            pd.to_numeric(market_diagnostics["pct_total_rows"], errors="coerce") < 0.01
        ]
        if not small.empty:
            labels = ", ".join(small["market_regime"].astype(str))
            notes.append(f"market labels below 1%: {labels}")
    if not sector_diagnostics.empty:
        small = sector_diagnostics[
            pd.to_numeric(sector_diagnostics["pct_total_rows"], errors="coerce") < 0.01
        ]
        if not small.empty:
            labels = ", ".join(small["sector_regime"].astype(str))
            notes.append(f"sector labels below 1%: {labels}")
    return "; ".join(notes) if notes else "no obvious count/distribution bug."


def _missing_priority_note(missing_symbols: dict[str, str]) -> str:
    if not missing_symbols:
        return "none from configured universe."

    missing = set(missing_symbols)
    priority: list[str] = []
    breadth = [symbol for symbol in ["MMFI", "MMTH", "HIGN", "LOWN"] if symbol in missing]
    if breadth:
        priority.append(
            ", ".join(breadth)
            + " for real breadth/new-high-new-low confirmation"
        )
    fear = [symbol for symbol in ["PCC"] if symbol in missing]
    if fear:
        priority.append(", ".join(fear) + " for sentiment/fear confirmation")
    rates = [symbol for symbol in ["US02Y", "T10Y2"] if symbol in missing]
    if rates:
        priority.append(", ".join(rates) + " for yield-curve context")
    other = sorted(missing - set(breadth) - set(fear) - set(rates))
    if other:
        priority.append(", ".join(other))
    return "; then ".join(priority) + "."


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:+.2f}%"
