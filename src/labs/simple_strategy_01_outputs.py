"""Output organization helpers for Simple Strategy #1."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import shutil
from typing import Any


ROOT_FILE_SUFFIXES = {".csv", ".html", ".txt", ".log"}
ROOT_KEEP_DIRS = {"latest", "runs", "archive"}
LATEST_ARTIFACTS = [
    "simple_strategy_01_report.html",
    "final_verdict_summary.csv",
    "final_verdict_summary.md",
    "key_metrics_summary.csv",
    "summary_by_symbol.csv",
    "strategy_vs_buy_hold.csv",
    "baseline_vs_optimized_summary.csv",
    "quasi_p_values.csv",
    "quasi_p_values_1000.csv",
    "permutation_1000_summary.csv",
    "time_split_validation.csv",
    "direction_mode_comparison.csv",
    "sizing_stress_summary.csv",
]


@dataclass
class OutputLayout:
    """Resolved output folders for one lab run."""

    base_dir: Path
    run_dir: Path
    latest_dir: Path
    archive_dir: Path
    moved_files: list[tuple[Path, Path]] = field(default_factory=list)
    moved_dirs: list[tuple[Path, Path]] = field(default_factory=list)
    latest_files: list[tuple[Path, Path]] = field(default_factory=list)


def prepare_simple_strategy_output(
    base_dir: str | Path,
    stage: str,
    output_config: dict[str, Any] | None = None,
) -> OutputLayout:
    """Create the standard output layout and return the run folder."""
    output_config = output_config or {}
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "archive"
    latest = base / "latest"
    runs = base / "runs"
    archive.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    moved_files: list[tuple[Path, Path]] = []
    moved_dirs: list[tuple[Path, Path]] = []
    if output_config.get("archive_old_root_outputs", True) or output_config.get(
        "clean_before_run",
        False,
    ):
        moved_files = archive_old_root_outputs(base)
    moved_dirs = archive_aborted_runs(base)
    latest_files = seed_latest_from_archive(base)
    write_output_readme(base)

    organize = output_config.get("organize_runs", True)
    run_dir = _unique_run_dir(runs, stage) if organize else base
    run_dir.mkdir(parents=True, exist_ok=False if organize else True)
    return OutputLayout(
        base_dir=base,
        run_dir=run_dir,
        latest_dir=latest,
        archive_dir=archive,
        moved_files=moved_files,
        moved_dirs=moved_dirs,
        latest_files=latest_files,
    )


def clean_simple_strategy_output(base_dir: str | Path) -> OutputLayout:
    """Archive root-level outputs without running research."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "latest").mkdir(parents=True, exist_ok=True)
    (base / "runs").mkdir(parents=True, exist_ok=True)
    (base / "archive").mkdir(parents=True, exist_ok=True)
    moved_files = archive_old_root_outputs(base)
    moved_dirs = archive_aborted_runs(base)
    latest_files = seed_latest_from_archive(base)
    write_output_readme(base)
    return OutputLayout(
        base_dir=base,
        run_dir=base,
        latest_dir=base / "latest",
        archive_dir=base / "archive",
        moved_files=moved_files,
        moved_dirs=moved_dirs,
        latest_files=latest_files,
    )


def archive_old_root_outputs(base_dir: str | Path) -> list[tuple[Path, Path]]:
    """Move root-level output files into archive/old_flat_outputs/<timestamp>."""
    base = Path(base_dir)
    files = [
        path
        for path in base.iterdir()
        if path.is_file()
        and path.suffix.lower() in ROOT_FILE_SUFFIXES
        and path.name != "README.txt"
    ]
    if not files:
        return []
    destination = _unique_archive_dir(base / "archive" / "old_flat_outputs")
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    for source in files:
        target = destination / source.name
        shutil.move(str(source), str(target))
        moved.append((source, target))
    return moved


def archive_aborted_runs(base_dir: str | Path) -> list[tuple[Path, Path]]:
    """Move preserved/aborted run folders into archive/aborted_runs."""
    base = Path(base_dir)
    destination_root = base / "archive" / "aborted_runs"
    destination_root.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    for source in base.iterdir():
        if not source.is_dir():
            continue
        if source.name in ROOT_KEEP_DIRS:
            continue
        if not _looks_like_aborted_run(source.name):
            continue
        target = _unique_named_dir(destination_root / source.name)
        shutil.move(str(source), str(target))
        moved.append((source, target))
    return moved


def copy_final_outputs_to_latest(
    run_paths: dict[str, Path],
    latest_dir: str | Path,
) -> dict[str, Path]:
    """Copy selected final artifacts from a successful run into latest/."""
    latest = Path(latest_dir)
    latest.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    by_name = {path.name: path for path in run_paths.values()}
    for artifact in LATEST_ARTIFACTS:
        source = by_name.get(artifact)
        if source is None or not source.exists():
            continue
        target = latest / artifact
        shutil.copy2(source, target)
        copied[f"latest_{source.stem}"] = target
    return copied


def seed_latest_from_archive(base_dir: str | Path) -> list[tuple[Path, Path]]:
    """Seed latest/ from the newest archived flat-output folder if possible."""
    base = Path(base_dir)
    latest = base / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    archive_root = base / "archive" / "old_flat_outputs"
    if not archive_root.exists():
        return []
    archive_dirs = sorted(
        [path for path in archive_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    if not archive_dirs:
        return []
    source_dir = archive_dirs[0]
    copied: list[tuple[Path, Path]] = []
    for artifact in LATEST_ARTIFACTS:
        source = source_dir / artifact
        if not source.exists():
            continue
        target = latest / artifact
        shutil.copy2(source, target)
        copied.append((source, target))
    key_metrics = latest / "key_metrics_summary.csv"
    if not key_metrics.exists():
        synthesized = _synthesize_key_metrics_summary(latest)
        if synthesized is not None:
            copied.append((synthesized, key_metrics))
    return copied


def write_output_readme(base_dir: str | Path) -> Path:
    """Write the root README that explains the output structure."""
    base = Path(base_dir)
    readme = base / "README.txt"
    text = """Simple Strategy 01 Output Layout

Start here:
- latest/ contains the most important final artifacts copied from the most recent successful run.
- runs/ contains one timestamped folder per run. Each run folder keeps its own CSVs, HTML report, logs, and checkpoints.
- archive/ contains older root-level outputs and aborted run artifacts that were moved out of the root folder.

Folders:
- latest/: quick-access copy of the current report, final verdict, key metrics, symbol summary, buy-and-hold comparison, baseline-vs-optimized summary, and quasi p-values.
- runs/<timestamp>_<stage_name>/: complete outputs for that specific run.
- archive/old_flat_outputs/<timestamp>/: old CSV/HTML/TXT/LOG files that used to live directly in this root folder.
- archive/aborted_runs/: preserved aborted-run folders, such as interrupted full-grid attempts.
- archive/partial_checkpoints/: reserved for manual checkpoint archiving if ever needed.

Root folder policy:
- Future runs should not write CSV/HTML/checkpoint files directly here.
- Old outputs are archived, not deleted.
- Look in latest/ first, then the newest runs/ folder if you need full details.
"""
    readme.write_text(text, encoding="utf-8")
    return readme


def _unique_run_dir(runs_dir: Path, stage: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stamp}_{_slug(stage)}"
    candidate = runs_dir / base_name
    counter = 2
    while candidate.exists():
        candidate = runs_dir / f"{base_name}_{counter}"
        counter += 1
    return candidate


def _synthesize_key_metrics_summary(latest_dir: Path) -> Path | None:
    """Create key_metrics_summary.csv for older archived runs that lack it."""
    summary_path = latest_dir / "summary_by_symbol.csv"
    compare_path = latest_dir / "strategy_vs_buy_hold.csv"
    if not summary_path.exists():
        return None
    import pandas as pd

    summary = pd.read_csv(summary_path)
    if summary.empty:
        return None
    columns = [
        "symbol",
        "total_return",
        "cagr",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "number_of_trades",
        "final_equity",
        "long_trade_count",
        "short_trade_count",
        "long_net_pnl",
        "short_net_pnl",
    ]
    out = summary[[col for col in columns if col in summary]].copy()
    if compare_path.exists():
        compare = pd.read_csv(compare_path)
        compare_cols = [
            col
            for col in ["symbol", "buy_hold_return", "excess_vs_buy_hold"]
            if col in compare
        ]
        if compare_cols and "symbol" in compare_cols:
            out = out.merge(compare[compare_cols], on="symbol", how="left")
    key_metrics = latest_dir / "key_metrics_summary.csv"
    out.to_csv(key_metrics, index=False)
    return key_metrics


def _unique_archive_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    counter = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{counter}"
        counter += 1
    return candidate


def _unique_named_dir(path: Path) -> Path:
    candidate = path
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{counter}")
        counter += 1
    return candidate


def _looks_like_aborted_run(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("preserved_") or "abort" in lowered or "aborted" in lowered


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "run"
