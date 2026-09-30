"""
src/config.py

Loads config.yaml once and exposes absolute, resolved paths so that every
other module can be run from any working directory (a notebook, the app,
a test runner, a cron job) without breaking on relative-path assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The project root is defined as two levels up from this file
# (src/config.py -> src/ -> project root), computed once at import time.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Absolute, ready-to-use filesystem paths derived from config.yaml."""

    raw_data: Path
    processed_dir: Path
    clean_monthly_csv: Path
    reconciliation_csv: Path
    annual_summary_csv: Path
    reports_dir: Path
    figures_dir: Path
    summary_report_md: Path
    models_dir: Path
    model_file: Path

    def ensure_output_dirs(self) -> None:
        """Create every output directory this project writes to, idempotently."""
        for d in (self.processed_dir, self.reports_dir, self.figures_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    """Typed view over config.yaml — avoids scattering raw dict lookups everywhere."""

    project_name: str
    kpi_id: str
    kpi_name: str
    paths: Paths
    complete_financial_years: list[str]
    train_start_date: str
    train_end_date: str
    holdout_periods: list[str]
    significance_level: float
    iqr_multiplier: float
    within_year_zscore_threshold: float
    structural_break_min_segment: int
    theme: dict[str, str]
    raw: dict[str, Any] = field(repr=False)  # the untouched parsed YAML, for anything not modeled above


def load_config(config_path: Path | str = CONFIG_PATH) -> AppConfig:
    """
    Parse config.yaml and resolve every path relative to the project root.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    KeyError
        If a required section is absent — fails loudly rather than defaulting
        silently, since a missing path or parameter should stop the pipeline,
        not produce a quietly-wrong result.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    p = raw["paths"]
    paths = Paths(
        raw_data=PROJECT_ROOT / p["raw_data"],
        processed_dir=PROJECT_ROOT / p["processed_dir"],
        clean_monthly_csv=PROJECT_ROOT / p["clean_monthly_csv"],
        reconciliation_csv=PROJECT_ROOT / p["reconciliation_csv"],
        annual_summary_csv=PROJECT_ROOT / p["annual_summary_csv"],
        reports_dir=PROJECT_ROOT / p["reports_dir"],
        figures_dir=PROJECT_ROOT / p["figures_dir"],
        summary_report_md=PROJECT_ROOT / p["summary_report_md"],
        models_dir=PROJECT_ROOT / p["models_dir"],
        model_file=PROJECT_ROOT / p["model_file"],
    )

    a = raw["analysis"]
    return AppConfig(
        project_name=raw["project"]["name"],
        kpi_id=raw["project"]["kpi_id"],
        kpi_name=raw["project"]["kpi_name"],
        paths=paths,
        complete_financial_years=a["complete_financial_years"],
        train_start_date=a["train_start_date"],
        train_end_date=a["train_end_date"],
        holdout_periods=a["holdout_periods"],
        significance_level=a["significance_level"],
        iqr_multiplier=a["iqr_multiplier"],
        within_year_zscore_threshold=a["within_year_zscore_threshold"],
        structural_break_min_segment=a["structural_break_min_segment"],
        theme=raw["theme"],
        raw=raw,
    )
