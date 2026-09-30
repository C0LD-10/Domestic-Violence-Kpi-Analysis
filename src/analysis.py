"""
src/analysis.py

Statistical characterization of the cleaned monthly/annual series:
    - annual trend significance (Kendall's tau + OLS regression)
    - seasonality (one-way ANOVA across calendar months)
    - structural break (SSR-minimizing two-segment scan)
    - outlier screening (global Tukey IQR + within-year z-score)

Every function returns plain pandas/numpy objects (no plotting, no I/O) so
they can be unit-tested in isolation and reused identically by the pipeline,
the notebook, and the Streamlit app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Annual trend
# ---------------------------------------------------------------------------

def annual_trend_tests(annual_totals: pd.Series) -> pd.DataFrame:
    """
    Test whether an annual series (indexed by financial year, in chronological
    order) shows a significant trend, using two complementary tests:

    - Kendall's tau: nonparametric, tests monotonic rank agreement only.
      Robust to nonlinearity but has very low power at small n (the smallest
      possible p-value with n=4 points is 0.083).
    - OLS linear regression: quantifies an average rate of change per year,
      but a high R^2 with n=4 is a near-mechanical consequence of fitting a
      line to few points and should not be over-interpreted as proof of a
      linear process.

    Both are reported side by side deliberately, so neither is cherry-picked.
    """
    idx = np.arange(len(annual_totals))
    values = annual_totals.values

    tau, p_kendall = stats.kendalltau(idx, values)
    slope, intercept, r_value, p_lr, std_err = stats.linregress(idx, values)

    return pd.DataFrame({
        "test": ["Kendall's tau (monotonic trend)", "OLS linear regression (rate of change)"],
        "statistic": [tau, slope],
        "p_value": [p_kendall, p_lr],
        "r_squared": [np.nan, r_value ** 2],
    })


def yoy_change(annual_totals: pd.Series) -> pd.DataFrame:
    """Year-on-year absolute and percentage change for an annual total series."""
    out = annual_totals.to_frame("total_incidents")
    out["yoy_change"] = out["total_incidents"].diff()
    out["yoy_pct"] = out["total_incidents"].pct_change() * 100
    return out


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------

MONTH_ORDER = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]


def seasonality_anova(clean_monthly: pd.DataFrame, complete_years: list[str]) -> tuple[pd.DataFrame, float, float]:
    """
    Test for a calendar-month effect using only financial years with complete
    (12/12) monthly coverage, so every month is represented an equal number
    of times (a fair comparison; partial years would bias the test).

    Returns
    -------
    (per_month_summary, F_statistic, p_value)
    """
    seasonal_data = clean_monthly[clean_monthly["financial_year"].isin(complete_years)].copy()

    summary = (
        seasonal_data.groupby("month_name")["value_num"]
        .agg(mean="mean", std="std", n="count")
        .reindex(MONTH_ORDER)
    )

    groups = [seasonal_data.loc[seasonal_data["month_name"] == m, "value_num"].values for m in MONTH_ORDER]
    f_stat, p_value = stats.f_oneway(*groups)
    return summary, f_stat, p_value


# ---------------------------------------------------------------------------
# Structural break
# ---------------------------------------------------------------------------

def _two_segment_ssr(t: np.ndarray, y: np.ndarray, breakpoint: int, min_segment: int) -> float:
    """Sum of squared residuals for independent linear fits either side of `breakpoint`."""
    left, right = t < breakpoint, t >= breakpoint
    if left.sum() < min_segment or right.sum() < min_segment:
        return np.inf
    ssr = 0.0
    for mask in (left, right):
        slope, intercept, *_ = stats.linregress(t[mask], y[mask])
        pred = intercept + slope * t[mask]
        ssr += np.sum((y[mask] - pred) ** 2)
    return ssr


def structural_break_scan(t: np.ndarray, y: np.ndarray, min_segment: int = 6) -> dict:
    """
    Scan every candidate breakpoint and return the SSR-minimizing split,
    compared against a single whole-window linear trend.

    This is a deliberately simple, fully auditable two-segment method —
    appropriate for a ~48-point series — rather than a more opaque
    multi-parameter change-point model. It is DESCRIPTIVE (identifies where
    the best-fit break falls) and is not itself a formal significance test
    for whether a break exists (see README limitations).
    """
    slope0, intercept0, r0, p0, se0 = stats.linregress(t, y)
    ssr_single = float(np.sum((y - (intercept0 + slope0 * t)) ** 2))

    candidates = np.arange(min_segment, len(t) - min_segment)
    ssr_scan = np.array([_two_segment_ssr(t, y, bp, min_segment) for bp in candidates])
    best_idx = int(np.argmin(ssr_scan))
    best_bp = int(candidates[best_idx])
    best_ssr = float(ssr_scan[best_idx])

    left, right = t < best_bp, t >= best_bp
    s1, i1, *_ = stats.linregress(t[left], y[left])
    s2, i2, *_ = stats.linregress(t[right], y[right])

    return {
        "single_trend_slope": slope0,
        "single_trend_intercept": intercept0,
        "single_trend_ssr": ssr_single,
        "best_breakpoint_t": best_bp,
        "best_ssr": best_ssr,
        "ssr_improvement_pct": (ssr_single - best_ssr) / ssr_single * 100,
        "segment1_slope": s1, "segment1_intercept": i1,
        "segment2_slope": s2, "segment2_intercept": i2,
    }


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def detect_outliers(clean_monthly: pd.DataFrame, iqr_multiplier: float = 1.5,
                     z_threshold: float = 2.0) -> dict:
    """
    Two independent outlier screens:
      - Global Tukey IQR rule (iqr_multiplier x IQR beyond Q1/Q3)
      - Within-financial-year z-score (catches anomalies a global rule might
        miss if the whole series has shifted level across years)
    """
    vals = clean_monthly["value_num"]
    q1, q3 = vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

    global_outliers = clean_monthly[(vals < lower) | (vals > upper)]

    working = clean_monthly.copy()
    working["fy_z"] = working.groupby("financial_year")["value_num"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0
    )
    local_outliers = working[working["fy_z"].abs() > z_threshold]

    return {
        "iqr_bounds": (float(lower), float(upper)),
        "global_outliers": global_outliers,
        "local_outliers": local_outliers,
    }
