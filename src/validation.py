"""
src/validation.py

Cross-granularity reconciliation: for every financial year with complete
(12/12 month) monthly coverage, does summing the monthly values reproduce
the independently-reported annual total? This is the project's core data-
trust check — if it fails, nothing downstream (trend, seasonality, forecast)
should be treated as reliable.
"""

from __future__ import annotations

import pandas as pd


def reconcile_monthly_vs_annual(df: pd.DataFrame, clean_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Compare monthly-summed totals against independently-reported annual totals.

    Parameters
    ----------
    df : the full cleaned frame (all granularities), from data_loader.clean()
    clean_monthly : the monthly-only, value-present frame, from data_loader.build_clean_monthly()

    Returns
    -------
    DataFrame indexed by financial_year with columns:
        monthly_sum, count, annual_reported, complete_year, discrepancy
    A `discrepancy` of 0 for a `complete_year=True` row is the evidence that
    the KPI export is internally consistent for that year.
    """
    annual_reported = (
        df[(df["granularity"] == "Annual") & df["value_num"].notna()]
        .set_index("financial_year")["value_num"]
    )
    monthly_sum_by_fy = clean_monthly.groupby("financial_year")["value_num"].agg(["sum", "count"])

    reconciliation = monthly_sum_by_fy.join(annual_reported.rename("annual_reported"), how="left")
    reconciliation = reconciliation.rename(columns={"sum": "monthly_sum"})
    reconciliation["complete_year"] = reconciliation["count"] == 12
    reconciliation["discrepancy"] = reconciliation["annual_reported"] - reconciliation["monthly_sum"]
    return reconciliation


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row completeness broken down by reporting granularity: valid / flagged
    "Not Collected" / blank-in-extract. See README "Two gap regimes" for why
    these are treated as distinct, non-interchangeable categories.
    """
    quality = pd.DataFrame({
        "n_rows": df.groupby("granularity").size(),
        "n_valid_numeric": df.groupby("granularity")["value_num"].apply(lambda s: s.notna().sum()),
        "n_flagged_NC": df.groupby("granularity")["not_collected"].sum(),
        "n_blank_missing": df.groupby("granularity").apply(
            lambda g: ((~g["not_collected"]) & g["value_num"].isna()).sum()
        ),
    })
    quality.loc["TOTAL"] = quality.sum()
    return quality


def assert_reconciliation_clean(reconciliation: pd.DataFrame) -> None:
    """
    Raise if any complete year shows a non-zero discrepancy.

    Intended to be called from the pipeline as a hard gate: if this fails,
    the pipeline should stop rather than silently proceed to fit models on
    data whose internal consistency has broken down.
    """
    complete = reconciliation[reconciliation["complete_year"]]
    bad = complete[complete["discrepancy"].abs() > 1e-6]
    if len(bad):
        raise ValueError(
            f"Reconciliation check FAILED for {len(bad)} year(s):\n{bad}\n"
            "Monthly totals no longer match reported annual totals — "
            "investigate before trusting downstream analysis."
        )
