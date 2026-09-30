"""
src/pipeline.py

Orchestrates the full, reproducible run:
    load raw -> clean -> validate (hard gate) -> analyse -> forecast
    -> save processed CSVs -> save figures -> save model -> write summary report

Entry point:
    python -m src.pipeline

Every step is a thin call into src/{data_loader,validation,analysis,forecasting,theme}.py
— this file contains orchestration and I/O only, no analysis logic of its own,
so the same tested functions back the pipeline, the notebook, and the app.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis, data_loader, forecasting, theme, validation
from src.config import AppConfig, load_config


def run(config: AppConfig) -> None:
    config.paths.ensure_output_dirs()
    th = theme.configure(config)

    print(f"[1/7] Loading & cleaning raw data from {config.paths.raw_data} ...")
    df, clean_monthly = data_loader.load_and_clean(config)
    print(f"       {len(clean_monthly)} clean monthly observations retained "
          f"(of {len(df[df['granularity'] == 'Monthly'])} monthly rows in the raw export).")

    print("[2/7] Validating cross-granularity reconciliation ...")
    reconciliation = validation.reconcile_monthly_vs_annual(df, clean_monthly)
    validation.assert_reconciliation_clean(reconciliation)  # hard gate — raises on failure
    quality = validation.data_quality_summary(df)
    print("       Reconciliation OK: monthly sums match reported annual totals for all complete years.")

    print("[3/7] Running trend, seasonality, structural-break, and outlier analysis ...")
    annual_reported = df[(df["granularity"] == "Annual") & df["value_num"].notna()].set_index("financial_year")["value_num"]
    annual_complete = annual_reported.loc[config.complete_financial_years].to_frame("total_incidents")
    trend_tests = analysis.annual_trend_tests(annual_complete["total_incidents"])
    yoy = analysis.yoy_change(annual_complete["total_incidents"])

    seasonal_summary, f_stat, p_anova = analysis.seasonality_anova(
        clean_monthly, config.complete_financial_years
    )

    contig = clean_monthly[
        (clean_monthly["StartDate"] >= config.train_start_date)
        & (clean_monthly["StartDate"] <= config.train_end_date)
    ].sort_values("StartDate").reset_index(drop=True)
    contig["t"] = np.arange(len(contig))
    break_result = analysis.structural_break_scan(
        contig["t"].values, contig["value_num"].values,
        min_segment=config.structural_break_min_segment,
    )

    outliers = analysis.detect_outliers(
        clean_monthly, iqr_multiplier=config.iqr_multiplier,
        z_threshold=config.within_year_zscore_threshold,
    )

    print("[4/7] Fitting & validating the forecast model out-of-sample ...")
    model = forecasting.fit_trend_seasonal_model(contig)
    holdout = clean_monthly[clean_monthly["period"].isin(config.holdout_periods)].copy()
    holdout["t"] = np.arange(contig["t"].max() + 1, contig["t"].max() + 1 + len(holdout))
    val_results = forecasting.validate_on_holdout(model, contig, holdout)
    val_summary = forecasting.summarize_validation(val_results)
    print(f"       Model MAE={val_summary['mae_model']:.2f} vs naive MAE={val_summary['mae_naive']:.2f} "
          f"({val_summary['improvement_over_naive_pct']:.1f}% improvement)")

    print("[5/7] Writing processed datasets to data/processed/ ...")
    clean_monthly.to_csv(config.paths.clean_monthly_csv, index=False)
    reconciliation.to_csv(config.paths.reconciliation_csv)
    annual_complete.join(yoy[["yoy_change", "yoy_pct"]]).to_csv(config.paths.annual_summary_csv)

    print("[6/7] Rendering figures to reports/figures/ ...")
    _save_figures(config, th, df, clean_monthly, quality, annual_complete, yoy,
                   seasonal_summary, p_anova, contig, break_result, outliers,
                   model, holdout, val_results)

    print("[7/7] Saving model & writing summary report ...")
    model.save(config.paths.model_file)
    _write_summary_report(config, quality, reconciliation, trend_tests, yoy,
                           f_stat, p_anova, break_result, outliers, val_summary)

    print(f"\nDone. Outputs written under: {config.paths.reports_dir}, "
          f"{config.paths.processed_dir}, {config.paths.models_dir}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figures(config, th, df, clean_monthly, quality, annual_complete, yoy,
                   seasonal_summary, p_anova, contig, break_result, outliers,
                   model, holdout, val_results) -> None:
    figs_dir = config.paths.figures_dir

    # 1. Data quality by granularity
    fig, ax = theme.new_fig((8, 4.2))
    gran = quality.drop("TOTAL")
    x = np.arange(len(gran))
    valid, nc, blank = gran["n_valid_numeric"], gran["n_flagged_NC"], gran["n_blank_missing"]
    ax.bar(x, valid, 0.6, color=th["ink"], label="Valid numeric", zorder=3)
    ax.bar(x, nc, 0.6, bottom=valid, color=th["amber"], label='Flagged "Not Collected"', zorder=3)
    ax.bar(x, blank, 0.6, bottom=valid + nc, color=th["rose"], label="Blank / missing", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(gran.index)
    theme.apply_theme(ax, th, title="Row completeness by reporting granularity", ylabel="Number of rows")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figs_dir / "01_data_quality.png"); plt.close(fig)

    # 2. Annual trend + YoY%
    fig, ax1 = theme.new_fig((9, 4.6))
    xi = np.arange(len(annual_complete))
    ax1.bar(xi, annual_complete["total_incidents"], color=th["ink"], width=0.5, zorder=3, label="Annual total")
    theme.apply_theme(ax1, th, title="Annual recorded medium-risk DV incidents",
                       xlabel="Financial year", ylabel="Total incidents")
    ax1.set_xticks(xi); ax1.set_xticklabels(annual_complete.index)
    ax2 = ax1.twinx()
    ax2.plot(xi, yoy["yoy_pct"], color=th["rose"], marker="o", linewidth=2.2, markersize=7, zorder=4, label="YoY % change")
    ax2.set_ylabel("Year-on-year % change", color=th["rose"])
    ax2.tick_params(axis="y", colors=th["rose"]); ax2.spines["top"].set_visible(False); ax2.grid(False)
    for i, v in enumerate(yoy["yoy_pct"]):
        if not np.isnan(v):
            ax2.annotate(f"{v:+.1f}%", (i, v), textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color=th["rose"], fontweight="bold")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figs_dir / "02_annual_trend.png"); plt.close(fig)

    # 3. Seasonality
    fig, ax = theme.new_fig((10, 4.4))
    x = np.arange(len(analysis.MONTH_ORDER))
    ax.bar(x, seasonal_summary["mean"], yerr=seasonal_summary["std"], capsize=4,
           color=th["teal"], ecolor=th["grey"], zorder=3, alpha=0.9)
    overall_mean = clean_monthly[clean_monthly["financial_year"].isin(config.complete_financial_years)]["value_num"].mean()
    ax.axhline(overall_mean, color=th["ink"], linestyle="--", linewidth=1.5, label=f"Overall mean ({overall_mean:.1f})", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(analysis.MONTH_ORDER)
    theme.apply_theme(ax, th, title=f"Mean monthly incidents by calendar month (ANOVA p={p_anova:.2f}, not significant)",
                       ylabel="Mean incidents (+/- 1 SD)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figs_dir / "03_seasonality.png"); plt.close(fig)

    # 4. Structural break
    fig, ax = theme.new_fig((10, 4.6))
    t, y = contig["t"].values, contig["value_num"].values
    ax.plot(contig["StartDate"], y, color=th["grey"], marker="o", markersize=4, linewidth=1.2, alpha=0.9, zorder=3, label="Observed")
    ax.plot(contig["StartDate"], break_result["single_trend_intercept"] + break_result["single_trend_slope"] * t,
            color=th["amber"], linewidth=2, linestyle="--", zorder=4, label="Single linear trend")
    bp = break_result["best_breakpoint_t"]
    left, right = t < bp, t >= bp
    ax.plot(contig.loc[left, "StartDate"], break_result["segment1_intercept"] + break_result["segment1_slope"] * t[left],
            color=th["ink"], linewidth=2.4, zorder=5, label="Segment 1 trend")
    ax.plot(contig.loc[right, "StartDate"], break_result["segment2_intercept"] + break_result["segment2_slope"] * t[right],
            color=th["rose"], linewidth=2.4, zorder=5, label="Segment 2 trend")
    ax.axvline(contig.loc[contig["t"] == bp, "StartDate"].values[0], color=th["rose"], linestyle=":", linewidth=1.5, zorder=2)
    theme.apply_theme(ax, th, title="Best-fit structural break (SSR-minimizing breakpoint)", xlabel="Month", ylabel="Incidents")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(figs_dir / "04_structural_break.png"); plt.close(fig)

    # 5. Outliers
    fig, ax = theme.new_fig((11, 4.4))
    ax.plot(clean_monthly["StartDate"], clean_monthly["value_num"], color=th["ink"], linewidth=1.6,
            marker="o", markersize=4, zorder=3, label="Monthly incidents")
    lo, hi = outliers["iqr_bounds"]
    ax.axhspan(lo, hi, color=th["teal"], alpha=0.12, zorder=1, label="Tukey IQR normal range")
    loc_out = outliers["local_outliers"]
    if len(loc_out):
        ax.scatter(loc_out["StartDate"], loc_out["value_num"], color=th["rose"], s=90, zorder=5,
                   edgecolor="white", linewidth=1.2, label="Within-year |z| > threshold")
    theme.apply_theme(ax, th, title="Full monthly series with outlier screening", xlabel="Month", ylabel="Incidents")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(figs_dir / "05_outliers.png"); plt.close(fig)

    # 6. Forecast validation
    fig, ax = theme.new_fig((10, 4.6))
    ax.plot(contig["StartDate"], contig["value_num"], color=th["grey"], linewidth=1.4, marker="o",
            markersize=3.5, alpha=0.9, zorder=3, label="Training data")
    ax.plot(holdout["StartDate"], val_results["actual"], color=th["ink"], linewidth=2.2, marker="o",
            markersize=7, zorder=5, label="Actual (holdout)")
    ax.plot(holdout["StartDate"], val_results["trend_seasonal_forecast"], color=th["teal"], linewidth=2.2,
            marker="s", markersize=7, linestyle="--", zorder=5, label="Trend+seasonal forecast")
    ax.plot(holdout["StartDate"], val_results["naive_baseline"], color=th["amber"], linewidth=1.8,
            linestyle=":", zorder=4, label="Naive baseline")
    theme.apply_theme(ax, th, title="Out-of-sample forecast validation", xlabel="Month", ylabel="Incidents")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(figs_dir / "06_forecast_validation.png"); plt.close(fig)

    print(f"       6 figures written to {figs_dir}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary_report(config, quality, reconciliation, trend_tests, yoy,
                           f_stat, p_anova, break_result, outliers, val_summary) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {config.kpi_name} ({config.kpi_id}) — Pipeline Summary Report",
        "",
        f"*Generated automatically by `src/pipeline.py` on {generated}.*",
        "",
        "## Data quality", "",
        quality.to_markdown(), "",
        "## Cross-granularity reconciliation",
        "",
        "Monthly-summed totals vs. independently-reported annual totals "
        "(a discrepancy of 0 for a complete year is the evidence the export is internally consistent):",
        "",
        reconciliation.round(2).to_markdown(), "",
        "## Annual trend", "",
        trend_tests.round(4).to_markdown(index=False), "",
        yoy.round(2).to_markdown(), "",
        "## Seasonality",
        "",
        f"One-way ANOVA across calendar months: F = {f_stat:.3f}, p = {p_anova:.3f} "
        f"({'significant' if p_anova < config.significance_level else 'NOT significant'} "
        f"at alpha={config.significance_level}).",
        "",
        "## Structural break (descriptive)",
        "",
        f"- Single-trend SSR: {break_result['single_trend_ssr']:,.0f}",
        f"- Best two-segment SSR: {break_result['best_ssr']:,.0f} "
        f"(breakpoint at t={break_result['best_breakpoint_t']})",
        f"- SSR improvement: {break_result['ssr_improvement_pct']:.1f}%",
        f"- Segment 1 slope: {break_result['segment1_slope']:+.2f} incidents/month",
        f"- Segment 2 slope: {break_result['segment2_slope']:+.2f} incidents/month",
        "",
        "## Outliers",
        "",
        f"- Global Tukey IQR bounds: [{outliers['iqr_bounds'][0]:.1f}, {outliers['iqr_bounds'][1]:.1f}] "
        f"-> {len(outliers['global_outliers'])} outlier(s)",
        f"- Within-year |z|>threshold check -> {len(outliers['local_outliers'])} outlier(s)",
        "",
        "## Forecast validation (out-of-sample)",
        "",
        f"- Trend+seasonal model MAE: {val_summary['mae_model']:.2f}  "
        f"(MAPE: {val_summary['mape_model_pct']:.1f}%)",
        f"- Naive baseline MAE: {val_summary['mae_naive']:.2f}",
        f"- Improvement over naive baseline: {val_summary['improvement_over_naive_pct']:.1f}%",
        "",
        "## Scope note",
        "",
        "This KPI measures **recorded** medium-risk domestic violence incidents, not the "
        "true prevalence of domestic abuse. See `README.md` -> Limitations for the full "
        "caveats before this report is used to support operational decisions.",
        "",
    ]
    config.paths.summary_report_md.write_text("\n".join(lines))


if __name__ == "__main__":
    try:
        cfg = load_config()
        run(cfg)
    except Exception as exc:  # surface a clean, non-traceback-only failure message
        print(f"\nPipeline FAILED: {exc}", file=sys.stderr)
        raise
