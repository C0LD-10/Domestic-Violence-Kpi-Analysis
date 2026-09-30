"""
src/data_loader.py

Ingests the raw DOMV9a KPI export and produces a cleaned, typed DataFrame.

Three cleaning problems are handled explicitly (see README "Data cleaning
decisions" for the rationale behind each):
  1. `value` mixes a numeric magnitude with a unit suffix ("278.00 Number").
  2. `value` sometimes literally equals "NC" (Not Collected) — a meaningful
     missing-data flag, kept distinct from an ordinary blank/NaN.
  3. `period` conflates financial year and reporting granularity into one
     string and is decomposed into two explicit columns.
"""

from __future__ import annotations

import pandas as pd

from src.config import AppConfig


def load_raw(config: AppConfig) -> pd.DataFrame:
    """Load the raw semicolon-delimited KPI export exactly as delivered."""
    return pd.read_csv(config.paths.raw_data, sep=";")


def classify_granularity(period: str) -> str:
    """
    Infer reporting granularity from a `period` string.

    Examples
    --------
    "2013/2014"       -> "Annual"
    "2013/2014_Q2"     -> "Quarterly"
    "2013/2014_Nov"    -> "Monthly"
    """
    if "_" not in period:
        return "Annual"
    suffix = period.split("_")[1]
    return "Quarterly" if suffix.startswith("Q") else "Monthly"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three cleaning transformations described in the module docstring.

    Returns a new DataFrame; `raw` is never mutated in place, so callers can
    always compare cleaned output against the original export if needed.
    """
    df = raw.copy()

    # 1) Explicit "Not Collected" flag, kept separate from numeric parsing.
    df["not_collected"] = df["value"].astype(str).str.strip().eq("NC")

    # 2) Extract the numeric magnitude, discarding the " Number" unit suffix.
    df["value_num"] = pd.to_numeric(
        df["value"].astype(str).str.extract(r"([\d\.]+)")[0], errors="coerce"
    )

    # 3) Parse dates and decompose the period key.
    df["StartDate"] = pd.to_datetime(df["StartDate"], format="%d/%m/%Y")
    df["EndDate"] = pd.to_datetime(df["EndDate"], format="%d/%m/%Y")
    df["financial_year"] = df["period"].str.split("_").str[0]
    df["granularity"] = df["period"].apply(classify_granularity)

    return df


def build_clean_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to monthly rows carrying a genuinely reported numeric value.

    Quarterly and annual rows are deliberately NOT included here — they are
    held back as an independent set for validation.validate_reconciliation().
    """
    monthly = df[df["granularity"] == "Monthly"].sort_values("StartDate").reset_index(drop=True)
    clean_monthly = monthly[monthly["value_num"].notna()].copy()
    clean_monthly = clean_monthly.sort_values("StartDate").reset_index(drop=True)
    clean_monthly["month_name"] = clean_monthly["StartDate"].dt.strftime("%b")
    return clean_monthly


def month_status(df_monthly_all: pd.DataFrame) -> pd.DataFrame:
    """
    Label every monthly row (including gaps) as Reported / Not Collected / Blank.

    Used by validation/reporting code that needs to visualize *where* gaps
    fall in time, not just how many rows are missing.
    """
    out = df_monthly_all.copy()
    out["status"] = out.apply(
        lambda r: "Reported" if pd.notna(r["value_num"])
        else ("Not Collected (flagged)" if r["not_collected"] else "Blank in extract"),
        axis=1,
    )
    return out


def load_and_clean(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point: raw -> cleaned full frame, cleaned monthly-only frame."""
    raw = load_raw(config)
    df = clean(raw)
    clean_monthly = build_clean_monthly(df)
    return df, clean_monthly
