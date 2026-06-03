"""
Signal validation pipeline — the primary entry point for the research.

Workflow:
    1. Load config.yaml
    2. Download daily OHLCV for all tickers (cached to data/raw/)
    3. Download SPY/QQQ for market regime features (cached to data/raw/)
    4. Classify states and compute rolling Markov signal per ticker
    5. Compute forward returns, forward path labels, and stock trend filters
    6. Join market regime columns to dataset on date
    7. Export current regime outputs to data/results/current/
    8. Export legacy Markov/RSI outputs to data/results/legacy/
    9. Generate legacy HTML report with signal, market, and trend filter analysis

Usage:
    cd markov-research
    python scripts/run_signal_validation.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Resolve project root so 'src' is importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.config import load_config
from src.data_loader import download_market_regime, download_universe
from src.features import (
    add_baseline_return_features,
    build_signal_dataset,
    ensure_market_forward_returns,
)
from src.reporting import (
    build_sector_console_summary,
    build_excess_console_summary,
    build_path_console_summary,
    build_personality_console_summary,
    build_ticker_summary,
    build_rsi_bb_console_summary,
    dataset_overview,
    export_excess_summary_csvs,
    generate_signal_report_html,
    print_filter_summary,
)
from src.regime_dashboard import generate_research_dashboard_html
from src.regime_diagnostics import (
    build_regime_diagnostics,
    build_regime_diagnostics_console_summary,
    write_regime_diagnostics,
)
from src.states.market_regime import (
    build_market_regime_console_summary,
    build_market_regime_daily,
    build_market_regime_summary,
    load_market_regime_symbol_data,
    normalize_market_regime_universe,
)
from src.states.sector_regime import (
    build_sector_regime_daily,
    build_sector_regime_summary,
)
from src.universe import filter_valid, get_tickers
from src.utils import ensure_dirs, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    cfg = load_config(_ROOT / "config.yaml")
    tickers = get_tickers(cfg.get_active_tickers())
    ticker_metadata = cfg.get_ticker_metadata()
    start_date = cfg.get_start_date()

    logger.info("=" * 60)
    logger.info("Markov Signal Validation")
    logger.info("=" * 60)
    logger.info(f"Active universe: {cfg.active_universe or 'legacy_flat_tickers'}")
    logger.info(f"Requested syms : {len(tickers)}")
    logger.info(f"Universe       : {tickers}")
    logger.info(f"Data window    : {start_date} -> today")
    logger.info(f"State lookback : {cfg.state_lookback} bars")
    logger.info(f"Bull/Bear thr  : ±{cfg.bull_threshold_pct}%")
    logger.info(f"Training window: {cfg.training_window} bars")
    logger.info(f"Forward returns: {cfg.forward_return_days} days")
    logger.info("=" * 60)

    ensure_dirs(
        cfg.paths.raw_data,
        cfg.paths.processed_data,
        cfg.paths.results_root,
        cfg.paths.current_results_dir,
        cfg.paths.legacy_results_dir,
    )
    current_dir = Path(cfg.paths.current_results_dir)
    legacy_dir = Path(cfg.paths.legacy_results_dir)

    # Download (or load from cache)
    raw_data = download_universe(
        tickers=tickers,
        start=start_date,
        interval=cfg.interval,
        cache_dir=cfg.paths.raw_data,
    )

    logger.info("Downloading market regime (SPY/QQQ) ...")
    regime = download_market_regime(start_date, cache_dir=cfg.paths.raw_data)
    if not regime.empty:
        regime = ensure_market_forward_returns(regime, cfg.forward_return_days)

    # Additive market/sector regime research panels. These use only current or
    # trailing data for labels and do not feed the legacy Markov signal.
    market_universe = normalize_market_regime_universe(cfg.market_regime_universe)
    market_load = load_market_regime_symbol_data(
        universe=market_universe,
        start=start_date,
        interval=cfg.interval,
        cache_dir=cfg.paths.raw_data,
        existing_data=raw_data,
    )
    market_panel = build_market_regime_daily(market_universe, market_load.data)
    market_summary = build_market_regime_summary(
        market_panel,
        days=cfg.forward_return_days,
    )

    sector_names = {
        meta["sector_etf"]: meta["sector"]
        for meta in ticker_metadata.values()
        if meta.get("sector_etf")
    }
    sector_data = {
        symbol: market_load.data.get(symbol, raw_data.get(symbol))
        for symbol in market_universe.get("sectors", [])
    }
    sector_panel = build_sector_regime_daily(
        sector_symbols=market_universe.get("sectors", []),
        sector_data=sector_data,
        market_panel=market_panel,
        sector_names=sector_names,
    )
    sector_summary = build_sector_regime_summary(
        sector_panel,
        days=cfg.forward_return_days,
    )

    market_panel_path = current_dir / "market_regime_daily.csv"
    sector_panel_path = current_dir / "sector_regime_daily.csv"
    market_summary_path = current_dir / "market_regime_summary.csv"
    sector_summary_path = current_dir / "sector_regime_summary.csv"
    dashboard_path = current_dir / "research_dashboard.html"
    diagnostics_dir = current_dir / "diagnostics"
    market_panel.to_csv(market_panel_path, index=False)
    sector_panel.to_csv(sector_panel_path, index=False)
    market_summary.to_csv(market_summary_path, index=False)
    sector_summary.to_csv(sector_summary_path, index=False)
    regime_diagnostics = build_regime_diagnostics(
        market_panel=market_panel,
        sector_panel=sector_panel,
        forward_days=cfg.forward_return_days,
        transition_days=[1, 5, 20],
    )
    diagnostic_paths = write_regime_diagnostics(regime_diagnostics, diagnostics_dir)
    generate_research_dashboard_html(
        market_panel=market_panel,
        market_summary=market_summary,
        sector_panel=sector_panel,
        sector_summary=sector_summary,
        missing_symbols=market_load.missing,
        out_path=dashboard_path,
        market_diagnostics=regime_diagnostics.market_diagnostics,
        sector_diagnostics=regime_diagnostics.sector_diagnostics,
        market_transition_5d=regime_diagnostics.market_transitions.get(5),
        market_transition_20d=regime_diagnostics.market_transitions.get(20),
        sector_transition_5d=regime_diagnostics.sector_transitions.get(5),
        sector_transition_20d=regime_diagnostics.sector_transitions.get(20),
    )
    logger.info(f"Saved market regime panel : {market_panel_path}")
    logger.info(f"Saved sector regime panel : {sector_panel_path}")
    logger.info(f"Saved market regime summary: {market_summary_path}")
    logger.info(f"Saved sector regime summary: {sector_summary_path}")
    for name, path in diagnostic_paths.items():
        logger.info(f"Saved regime diagnostic {name}: {path}")
    logger.info(f"Saved research dashboard   : {dashboard_path}")
    if market_load.missing:
        logger.warning(f"Market regime unavailable symbols: {market_load.missing}")

    # Drop tickers with insufficient history
    min_bars = cfg.training_window + cfg.state_lookback + 50
    valid_tickers = filter_valid(raw_data, min_bars=min_bars)

    if not valid_tickers:
        logger.error("No tickers passed the minimum bar filter. Exiting.")
        sys.exit(1)

    logger.info(f"Valid tickers ({len(valid_tickers)}): {valid_tickers}")

    # Build signal dataset ticker by ticker
    all_frames: list[pd.DataFrame] = []
    processed_tickers: list[str] = []

    for ticker in valid_tickers:
        logger.info(f"Processing {ticker} ...")
        ohlcv = raw_data[ticker]
        df = build_signal_dataset(ticker, ohlcv, cfg)

        if df.empty:
            logger.warning(f"{ticker}: no rows after warm-up - skipping")
            continue

        logger.info(f"  {ticker}: {len(df):,} rows with signal")
        meta = ticker_metadata.get(
            ticker,
            {
                "sector": "unclassified",
                "sector_etf": "",
                "instrument_type": "stock",
            },
        )
        df["sector"] = meta["sector"]
        df["sector_etf"] = meta["sector_etf"]
        df["instrument_type"] = meta["instrument_type"]
        all_frames.append(df)
        processed_tickers.append(ticker)

    if not all_frames:
        logger.error("No data to save after processing. Exiting.")
        sys.exit(1)

    dataset = pd.concat(all_frames, ignore_index=True)
    skipped_tickers = [ticker for ticker in tickers if ticker not in processed_tickers]
    if skipped_tickers:
        logger.warning(f"Skipped symbols ({len(skipped_tickers)}): {skipped_tickers}")
    else:
        logger.info("Skipped symbols: none")

    # ── Join market regime columns ────────────────────────────────────────────
    if not regime.empty:
        regime_flat = regime.reset_index()
        dataset = dataset.merge(regime_flat, on="date", how="left")
        logger.info(f"Market regime joined ({len(regime_flat.columns)} columns)")
    else:
        logger.warning("Market regime unavailable - filter sections will be omitted from report")

    dataset = add_baseline_return_features(dataset, raw_data, cfg.forward_return_days)
    logger.info("Baseline and excess-return columns added")

    # ── Dataset summary ───────────────────────────────────────────────────────
    logger.info("\n" + dataset_overview(dataset))

    logger.info("\nPer-ticker summary:")
    logger.info("\n" + build_ticker_summary(dataset).to_string())

    # ── Export parquet + CSV ─────────────────────────────────────────────────
    parquet_path = legacy_dir / "markov_signal_dataset.parquet"
    csv_path = legacy_dir / "markov_signal_dataset.csv"

    dataset.to_parquet(parquet_path, index=False)
    dataset.to_csv(csv_path, index=False)
    summary_paths = export_excess_summary_csvs(dataset, legacy_dir)

    logger.info(f"\nSaved parquet : {parquet_path}")
    logger.info(f"Saved CSV     : {csv_path}")
    for name, path in summary_paths.items():
        logger.info(f"Saved {name:18}: {path}")

    # ── Generate HTML report ─────────────────────────────────────────────────
    cfg_summary = {
        "active_universe": cfg.active_universe or "legacy_flat_tickers",
        "tickers": ", ".join(valid_tickers),
        "data_window": f"{start_date} -> today",
        "state_lookback": f"{cfg.state_lookback} bars",
        "bull_threshold": f"+{cfg.bull_threshold_pct}%",
        "bear_threshold": f"-{cfg.bear_threshold_pct}%",
        "training_window": f"{cfg.training_window} bars",
        "signal_threshold": f"{cfg.signal_threshold_pct}%",
        "forward_returns": str(cfg.forward_return_days),
        "interval": cfg.interval,
    }

    html_path = legacy_dir / "legacy_markov_report.html"
    logger.info("Generating legacy Markov HTML report ...")
    generate_signal_report_html(
        dataset,
        html_path,
        cfg_summary=cfg_summary,
        ticker_groups=cfg.ticker_groups,
    )
    logger.info(f"Saved legacy Markov report: {html_path}")
    logger.info(f"Current dashboard         : {dashboard_path}")

    # ── Console filter summary ────────────────────────────────────────────────
    print_filter_summary(dataset)
    sector_lines = build_sector_console_summary(
        dataset,
        active_universe=cfg.active_universe or "legacy_flat_tickers",
        requested_symbols=tickers,
        processed_symbols=processed_tickers,
        skipped_symbols=skipped_tickers,
    )
    print("\nSector universe summary:")
    for line in sector_lines:
        print(f"  {line}")
    print("\nBaseline-adjusted edge summary:")
    for line in build_excess_console_summary(dataset):
        print(f"  {line}")
    print("\nStock personality summary:")
    for line in build_personality_console_summary(dataset):
        print(f"  {line}")
    print("\nForward path / tradability summary:")
    for line in build_path_console_summary(dataset):
        print(f"  {line}")
    print("\nRSI/Bollinger vs Markov summary:")
    for line in build_rsi_bb_console_summary(dataset):
        print(f"  {line}")
    print("\nMarket regime layer summary:")
    for line in build_market_regime_console_summary(market_panel, market_load.missing):
        print(f"  {line}")
    print("\nRegime diagnostics summary:")
    for line in build_regime_diagnostics_console_summary(
        market_diagnostics=regime_diagnostics.market_diagnostics,
        sector_diagnostics=regime_diagnostics.sector_diagnostics,
        market_transition_1d=regime_diagnostics.market_transitions.get(1, pd.DataFrame()),
        market_transition_5d=regime_diagnostics.market_transitions.get(5, pd.DataFrame()),
        sector_transition_1d=regime_diagnostics.sector_transitions.get(1, pd.DataFrame()),
        sector_transition_5d=regime_diagnostics.sector_transitions.get(5, pd.DataFrame()),
        missing_symbols=market_load.missing,
    ):
        print(f"  {line}")
    print("\nOutput locations:")
    print(f"  Current dashboard: {dashboard_path}")
    print(f"  Legacy Markov report: {html_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
