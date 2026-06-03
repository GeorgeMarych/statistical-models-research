"""
Regenerate the legacy Markov signal HTML report from the saved validation dataset.

This does not download data or rebuild signals. It only reads the existing
legacy dataset and writes legacy_markov_report.html.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Resolve project root so 'src' is importable when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config import load_config
from src.reporting import (
    build_excess_console_summary,
    build_path_console_summary,
    build_personality_console_summary,
    build_research_console_summary,
    build_rsi_bb_console_summary,
    export_excess_summary_csvs,
    generate_signal_report_html,
)


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _load_saved_dataset(
    legacy_results_dir: Path,
    fallback_results_root: Path,
) -> tuple[pd.DataFrame, Path]:
    candidates = [
        legacy_results_dir / "markov_signal_dataset.parquet",
        legacy_results_dir / "markov_signal_dataset.csv",
        fallback_results_root / "markov_signal_dataset.parquet",
        fallback_results_root / "markov_signal_dataset.csv",
    ]

    source_path: Path | None = None
    df: pd.DataFrame | None = None
    for path in candidates:
        if not path.exists():
            continue
        source_path = path
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["date"])
        break

    if df is None or source_path is None:
        expected = " or ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No saved legacy signal dataset found. Expected {expected}.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df, source_path


def main() -> None:
    cfg = load_config(_ROOT / "config.yaml")
    results_root = _resolve_project_path(cfg.paths.results_root)
    legacy_results_dir = _resolve_project_path(cfg.paths.legacy_results_dir)
    legacy_results_dir.mkdir(parents=True, exist_ok=True)

    dataset, source_path = _load_saved_dataset(legacy_results_dir, results_root)
    if dataset.empty:
        raise SystemExit(f"Saved dataset is empty: {source_path}")

    tickers = sorted(dataset["ticker"].dropna().unique().tolist())
    start_date = cfg.get_start_date()
    cfg_summary = {
        "active_universe": cfg.active_universe or "legacy_flat_tickers",
        "tickers": ", ".join(tickers),
        "data_window": f"{start_date} -> today",
        "state_lookback": f"{cfg.state_lookback} bars",
        "bull_threshold": f"+{cfg.bull_threshold_pct}%",
        "bear_threshold": f"-{cfg.bear_threshold_pct}%",
        "training_window": f"{cfg.training_window} bars",
        "signal_threshold": f"{cfg.signal_threshold_pct}%",
        "forward_returns": str(cfg.forward_return_days),
        "interval": cfg.interval,
    }

    html_path = legacy_results_dir / "legacy_markov_report.html"
    generate_signal_report_html(
        dataset,
        html_path,
        cfg_summary=cfg_summary,
        ticker_groups=cfg.ticker_groups,
    )
    summary_paths = export_excess_summary_csvs(dataset, legacy_results_dir)
    research_summary = build_research_console_summary(dataset, cfg.ticker_groups)

    print(f"Rows loaded : {len(dataset):,}")
    print(
        "Date range  : "
        f"{dataset['date'].min().date()} -> {dataset['date'].max().date()}"
    )
    print(f"Tickers     : {', '.join(tickers)}")
    print(f"Source path : {source_path}")
    print(f"Output path : {html_path}")
    print("Summary CSVs:")
    for name, path in summary_paths.items():
        print(f"  {name}: {path}")
    print()
    print("Research summary:")
    print(f"  Best ticker by high-signal avg 20D return : {research_summary['best_ticker_high_avg']}")
    print(f"  Best ticker by high-signal 20D win rate   : {research_summary['best_ticker_high_win']}")
    print(f"  Worst ticker by high-signal avg 20D return: {research_summary['worst_ticker_high_avg']}")
    print(f"  Low-signal better than high-signal tickers: {research_summary['low_better_tickers']}")
    print(f"  Best ticker with market_healthy_either    : {research_summary['best_ticker_market_avg']}")
    print(f"  Best group by high-signal avg 20D return  : {research_summary['best_group_high_avg']}")
    print(f"  Best group by high-signal 20D win rate    : {research_summary['best_group_high_win']}")
    print()
    print("Baseline-adjusted edge summary:")
    for line in build_excess_console_summary(dataset):
        print(f"  {line}")
    print()
    print("Stock personality summary:")
    for line in build_personality_console_summary(dataset):
        print(f"  {line}")
    print()
    print("Forward path / tradability summary:")
    for line in build_path_console_summary(dataset):
        print(f"  {line}")
    print()
    print("RSI/Bollinger vs Markov summary:")
    for line in build_rsi_bb_console_summary(dataset):
        print(f"  {line}")


if __name__ == "__main__":
    main()
