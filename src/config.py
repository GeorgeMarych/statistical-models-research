"""Configuration loader — maps config.yaml to a typed dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml


@dataclass
class Paths:
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    results_root: str = "data/results"
    current_results_dir: str = "data/results/current"
    legacy_results_dir: str = "data/results/legacy"

    @property
    def results(self) -> str:
        """Backward-compatible alias for the results root."""
        return self.results_root


@dataclass
class Config:
    tickers: list[str]
    years_back: int
    interval: str
    state_lookback: int
    bull_threshold_pct: float
    bear_threshold_pct: float
    training_window: int
    signal_threshold_pct: float
    forward_return_days: list[int]
    start_date: str | None = None
    active_universe: str | None = None
    universes: dict[str, dict] = field(default_factory=dict)
    ticker_groups: dict[str, list[str]] = field(default_factory=dict)
    market_regime_universe: dict[str, list[str]] = field(default_factory=dict)
    paths: Paths = field(default_factory=Paths)

    # Convenience converters from percentage to fraction

    @property
    def bull_threshold(self) -> float:
        return self.bull_threshold_pct / 100.0

    @property
    def bear_threshold(self) -> float:
        return self.bear_threshold_pct / 100.0

    @property
    def signal_threshold(self) -> float:
        return self.signal_threshold_pct / 100.0

    def get_start_date(self) -> str:
        """Return start date string; computes from years_back when start_date is null."""
        if self.start_date:
            return self.start_date
        dt = datetime.today() - timedelta(days=self.years_back * 365)
        return dt.strftime("%Y-%m-%d")

    def get_universe_members(self) -> list[dict[str, str]]:
        """
        Return active universe members with sector metadata.

        If active_universe points to a structured universe, the configured sector
        ETFs and stocks are flattened. Otherwise the legacy flat tickers list is
        returned as unclassified stock metadata.
        """
        members: list[dict[str, str]] = []

        if self.active_universe and self.active_universe in self.universes:
            universe = self.universes[self.active_universe] or {}
            for sector, spec in universe.items():
                sector_name = str(sector)
                sector_etf = str((spec or {}).get("etf", "")).strip().upper()
                if sector_etf:
                    members.append(
                        {
                            "ticker": sector_etf,
                            "sector": sector_name,
                            "sector_etf": sector_etf,
                            "instrument_type": "sector_etf",
                        }
                    )
                for raw in (spec or {}).get("stocks", []) or []:
                    ticker = str(raw).strip().upper()
                    if not ticker:
                        continue
                    members.append(
                        {
                            "ticker": ticker,
                            "sector": sector_name,
                            "sector_etf": sector_etf,
                            "instrument_type": "stock",
                        }
                    )
        else:
            for raw in self.tickers:
                ticker = str(raw).strip().upper()
                if not ticker:
                    continue
                members.append(
                    {
                        "ticker": ticker,
                        "sector": "unclassified",
                        "sector_etf": "",
                        "instrument_type": "stock",
                    }
                )

        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for member in members:
            ticker = member["ticker"]
            if ticker in seen:
                continue
            seen.add(ticker)
            deduped.append(member)
        return deduped

    def get_active_tickers(self) -> list[str]:
        """Return the ticker symbols for the active universe."""
        return [member["ticker"] for member in self.get_universe_members()]

    def get_ticker_metadata(self) -> dict[str, dict[str, str]]:
        """Return ticker -> sector metadata for the active universe."""
        return {member["ticker"]: member for member in self.get_universe_members()}


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and parse config.yaml into a Config dataclass."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw.get("paths", {})
    results_root = paths_raw.get(
        "results_root",
        paths_raw.get("results", "data/results"),
    )
    paths = Paths(
        raw_data=paths_raw.get("raw_data", "data/raw"),
        processed_data=paths_raw.get("processed_data", "data/processed"),
        results_root=results_root,
        current_results_dir=paths_raw.get(
            "current_results_dir",
            str(Path(results_root) / "current"),
        ),
        legacy_results_dir=paths_raw.get(
            "legacy_results_dir",
            str(Path(results_root) / "legacy"),
        ),
    )

    groups_raw = raw.get("ticker_groups", {}) or {}
    ticker_groups = {
        str(group): [str(t).strip().upper() for t in tickers]
        for group, tickers in groups_raw.items()
    }

    return Config(
        tickers=raw.get("tickers", []),
        start_date=raw.get("start_date"),
        active_universe=raw.get("active_universe"),
        universes=raw.get("universes", {}) or {},
        market_regime_universe=raw.get("market_regime_universe", {}) or {},
        years_back=raw.get("years_back", 10),
        interval=raw.get("interval", "1d"),
        state_lookback=raw["state_lookback"],
        bull_threshold_pct=float(raw["bull_threshold_pct"]),
        bear_threshold_pct=float(raw["bear_threshold_pct"]),
        training_window=raw["training_window"],
        signal_threshold_pct=float(raw["signal_threshold_pct"]),
        forward_return_days=raw["forward_return_days"],
        ticker_groups=ticker_groups,
        paths=paths,
    )
