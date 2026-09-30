"""
app/streamlit_app.py

Interactive dashboard for the DOMV9a KPI analysis.

Run with:
    streamlit run app/streamlit_app.py

Reads only PROCESSED artifacts (data/processed/, models/) written by
src/pipeline.py — it does not repeat analysis logic itself, so the app and
the pipeline can never silently drift apart. If those artifacts are missing,
the app tells the user to run the pipeline first rather than failing obscurely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly from app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analysis, forecasting, theme
from src.config import load_config

st.set_page_config(page_title="DOMV9a — Domestic Violence KPI Dashboard", layout="wide")

config = load_config()
th = theme.configure(config)

# ---------------------------------------------------------------------------
# Load processed artifacts (fail with a clear instruction, not a stack trace)
# ---------------------------------------------------------------------------

missing = [p for p in (config.paths.clean_monthly_csv, config.paths.reconciliation_csv,
                        config.paths.annual_summary_csv, config.paths.model_file) if not p.exists()]
if missing:
    st.error(
        "Processed data or model artifacts are missing:\n\n"
        + "\n".join(f"- `{p}`" for p in missing)
        + "\n\nRun `python -m src.pipeline` from the project root first, then reload this app."
    )
    st.stop()

clean_monthly = pd.read_csv(config.paths.clean_monthly_csv, parse_dates=["StartDate", "EndDate"])
reconciliation = pd.read_csv(config.paths.reconciliation_csv, index_col=0)
annual_summary = pd.read_csv(config.paths.annual_summary_csv, index_col=0)
model = forecasting.TrendSeasonalModel.load(config.paths.model_file)

# ---------------------------------------------------------------------------
# Custom CSS — apply the shared theme colors to the Streamlit chrome itself
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {th['paper']}; }}
        h1, h2, h3 {{ color: {th['ink']}; }}
        [data-testid="stMetricValue"] {{ color: {th['ink']}; }}
        .caveat-box {{
            background: {th['paper']};
            border-left: 4px solid {th['ink']};
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.92rem;
            color: {th['text']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Domestic Violence Incidents (Medium Risk) — KPI Dashboard")
st.caption(f"KPI `{config.kpi_id}` · {config.kpi_name} · data validated via `src/pipeline.py`")

st.markdown(
    """
    <div class="caveat-box">
    <b>Scope note:</b> this dashboard shows <i>recorded</i> medium-risk domestic violence incidents
    for one local authority. It is not a measure of true prevalence, and trends may reflect changes
    in recording practice as much as changes in underlying harm. See the README for full limitations.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# KPI summary row
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
latest_complete_fy = config.complete_financial_years[-1]
latest_total = int(annual_summary.loc[latest_complete_fy, "total_incidents"])
first_total = int(annual_summary.loc[config.complete_financial_years[0], "total_incidents"])
latest_yoy = annual_summary.loc[latest_complete_fy, "yoy_pct"]

col1.metric(f"Total incidents, {latest_complete_fy}", f"{latest_total}")
col2.metric("Growth since first complete year", f"{(latest_total / first_total - 1) * 100:+.0f}%")
col3.metric("Latest year-on-year change", f"{latest_yoy:+.1f}%")
col4.metric("Clean monthly observations", f"{len(clean_monthly)}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs mirroring the notebook's analytical structure
# ---------------------------------------------------------------------------

tab_trend, tab_season, tab_quality, tab_forecast = st.tabs(
    ["Annual trend", "Seasonality", "Data quality & reconciliation", "Forecast validation"]
)

with tab_trend:
    st.subheader("Annual totals & year-on-year growth")
    fig, ax1 = theme.new_fig((9, 4.3))
    xi = np.arange(len(annual_summary))
    ax1.bar(xi, annual_summary["total_incidents"], color=th["ink"], width=0.5, zorder=3)
    theme.apply_theme(ax1, th, xlabel="Financial year", ylabel="Total incidents")
    ax1.set_xticks(xi); ax1.set_xticklabels(annual_summary.index, rotation=0)
    ax2 = ax1.twinx()
    ax2.plot(xi, annual_summary["yoy_pct"], color=th["rose"], marker="o", linewidth=2.2, markersize=7, zorder=4)
    ax2.set_ylabel("YoY % change", color=th["rose"])
    ax2.tick_params(axis="y", colors=th["rose"]); ax2.spines["top"].set_visible(False); ax2.grid(False)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Growth **decelerated** every year (+66.5% -> +34.1% -> +15.1%) even as the absolute "
        "level kept rising — a saturating pattern, not an escalating one."
    )
    st.dataframe(annual_summary.style.format({"total_incidents": "{:.0f}", "yoy_change": "{:+.0f}", "yoy_pct": "{:+.1f}%"}))

with tab_season:
    st.subheader("Month-of-year effect")
    seasonal_summary, f_stat, p_anova = analysis.seasonality_anova(
        clean_monthly, config.complete_financial_years
    )
    fig, ax = theme.new_fig((10, 4.2))
    x = np.arange(len(analysis.MONTH_ORDER))
    ax.bar(x, seasonal_summary["mean"], yerr=seasonal_summary["std"], capsize=4,
           color=th["teal"], ecolor=th["grey"], zorder=3, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(analysis.MONTH_ORDER)
    theme.apply_theme(ax, th, ylabel="Mean incidents (+/- 1 SD)")
    st.pyplot(fig, use_container_width=True)
    verdict = "significant" if p_anova < config.significance_level else "**not statistically significant**"
    st.caption(f"One-way ANOVA across calendar months: F={f_stat:.2f}, p={p_anova:.3f} — {verdict} "
               f"at alpha={config.significance_level}. The visual winter bump does not survive testing.")

with tab_quality:
    st.subheader("Cross-granularity reconciliation")
    st.write(
        "For every financial year with complete (12/12) monthly reporting, do monthly sums "
        "match the independently-reported annual total?"
    )
    st.dataframe(reconciliation)
    complete_ok = reconciliation.loc[reconciliation["complete_year"] == True, "discrepancy"]
    if (complete_ok.abs() < 1e-6).all():
        st.success("All complete years reconcile exactly (discrepancy = 0). The export is internally consistent.")
    else:
        st.warning("Reconciliation discrepancies detected — see table above.")

    st.subheader("Monthly reporting status")
    status_color = {"Reported": th["ink"], "Not Collected (flagged)": th["amber"], "Blank in extract": th["rose"]}
    st.caption("Two distinct gap regimes exist: a declared 'Not Collected' interruption "
               "(Aug 2015-Mar 2017), and an apparently incomplete later extract "
               "(FY2017/18 absent entirely; FY2018/19 present but blank). See README for detail.")

with tab_forecast:
    st.subheader("Out-of-sample forecast validation")
    contig = clean_monthly[
        (clean_monthly["StartDate"] >= config.train_start_date)
        & (clean_monthly["StartDate"] <= config.train_end_date)
    ].sort_values("StartDate").reset_index(drop=True)
    contig["t"] = np.arange(len(contig))
    holdout = clean_monthly[clean_monthly["period"].isin(config.holdout_periods)].copy()
    holdout["t"] = np.arange(contig["t"].max() + 1, contig["t"].max() + 1 + len(holdout))
    val_results = forecasting.validate_on_holdout(model, contig, holdout)
    val_summary = forecasting.summarize_validation(val_results)

    fig, ax = theme.new_fig((10, 4.4))
    ax.plot(contig["StartDate"], contig["value_num"], color=th["grey"], linewidth=1.4, marker="o",
            markersize=3, alpha=0.9, zorder=3, label="Training data")
    ax.plot(holdout["StartDate"], val_results["actual"], color=th["ink"], linewidth=2.2, marker="o",
            markersize=7, zorder=5, label="Actual (holdout)")
    ax.plot(holdout["StartDate"], val_results["trend_seasonal_forecast"], color=th["teal"], linewidth=2.2,
            marker="s", markersize=7, linestyle="--", zorder=5, label="Trend+seasonal forecast")
    ax.plot(holdout["StartDate"], val_results["naive_baseline"], color=th["amber"], linewidth=1.8,
            linestyle=":", zorder=4, label="Naive baseline")
    theme.apply_theme(ax, th, xlabel="Month", ylabel="Incidents")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    st.pyplot(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model MAE", f"{val_summary['mae_model']:.2f}")
    c2.metric("Naive baseline MAE", f"{val_summary['mae_naive']:.2f}")
    c3.metric("Improvement over baseline", f"{val_summary['improvement_over_naive_pct']:.1f}%")
    st.caption(
        "This model is deliberately **not** extrapolated past this validated window — the "
        "reporting regime itself changes shortly after (see Data quality tab)."
    )

st.divider()
st.caption("Generated from data/processed/ and models/ — run `python -m src.pipeline` to refresh after updating data/raw/domviol.csv.")
