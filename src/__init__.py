"""
src/ — analysis pipeline for the DOMV9a domestic violence KPI.

Modules:
    config       Loads config.yaml and resolves project-root-relative paths.
    theme        Shared matplotlib color theme, used by the pipeline and the app.
    data_loader  Ingests the raw KPI export and produces a cleaned, typed frame.
    validation   Cross-granularity reconciliation (monthly <-> quarterly <-> annual).
    analysis     Trend tests, seasonality ANOVA, structural break, outlier screen.
    forecasting  Trend + seasonal-index model, with honest out-of-sample validation.
    pipeline     Orchestrates the full run: load -> clean -> validate -> analyse ->
                 forecast -> figures -> summary report. Entry point: `python -m src.pipeline`.
"""

__version__ = "1.0.0"
