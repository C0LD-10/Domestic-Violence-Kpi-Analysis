"""
src/forecasting.py

A deliberately simple, fully auditable trend + additive-seasonal-index model,
appropriate given ~48 training observations and no statistically significant
seasonality (src/analysis.seasonality_anova). Complex models (e.g. ARIMA with
several parameters) would risk over-fitting a series this short.

The model is always validated out-of-sample (never judged on training fit
alone) — see `validate_on_holdout()`. It is intentionally NOT extrapolated
past the validated holdout window; the pipeline stops the forecast there
because the data-quality audit (src/validation.py) shows the reporting
regime itself changes shortly after (see README limitations).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis import MONTH_ORDER


@dataclass
class TrendSeasonalModel:
    """A fitted trend + seasonal-index model, JSON-serializable for models/."""

    slope: float
    intercept: float
    seasonal_index: dict  # {month_name: additive offset}
    train_start_t: int
    train_end_t: int
    r_squared: float

    def predict(self, t_values: np.ndarray, month_names: list[str]) -> np.ndarray:
        """Predict incident counts for given time indices and their calendar months."""
        trend = self.intercept + self.slope * np.asarray(t_values)
        seasonal = np.array([self.seasonal_index.get(m, 0.0) for m in month_names])
        return trend + seasonal

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrendSeasonalModel":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def fit_trend_seasonal_model(train: pd.DataFrame) -> TrendSeasonalModel:
    """
    Fit trend + seasonal index on a contiguous training frame.

    Parameters
    ----------
    train : DataFrame with columns 't' (integer time index), 'value_num',
            and 'month_name', sorted chronologically with no gaps.
    """
    t, y = train["t"].values, train["value_num"].values
    slope, intercept, r_value, p_value, se = stats.linregress(t, y)

    trend_pred = intercept + slope * t
    resid = y - trend_pred
    seasonal_index = (
        pd.Series(resid, index=train["month_name"]).groupby(level=0).mean().reindex(MONTH_ORDER).to_dict()
    )
    # Replace any NaN (a month absent from training, shouldn't happen with a
    # full 12-month cycle, but guarded defensively) with 0 = no adjustment.
    seasonal_index = {k: (0.0 if pd.isna(v) else float(v)) for k, v in seasonal_index.items()}

    return TrendSeasonalModel(
        slope=float(slope),
        intercept=float(intercept),
        seasonal_index=seasonal_index,
        train_start_t=int(t.min()),
        train_end_t=int(t.max()),
        r_squared=float(r_value ** 2),
    )


def naive_baseline(train: pd.DataFrame, horizon: int, window: int = 4) -> np.ndarray:
    """Persistence baseline: repeat the mean of the last `window` training months."""
    return np.repeat(train["value_num"].tail(window).mean(), horizon)


def validate_on_holdout(model: TrendSeasonalModel, train: pd.DataFrame,
                         holdout: pd.DataFrame) -> pd.DataFrame:
    """
    Score the fitted model against genuinely held-out actuals, alongside the
    naive baseline it must beat to be worth using.

    Parameters
    ----------
    holdout : DataFrame with 't', 'month_name', 'value_num' (actual), sorted
              chronologically, immediately following the training window.
    """
    forecast = model.predict(holdout["t"].values, holdout["month_name"].tolist())
    baseline = naive_baseline(train, horizon=len(holdout))

    results = pd.DataFrame({
        "period": holdout["period"].values if "period" in holdout else holdout.index,
        "actual": holdout["value_num"].values,
        "trend_seasonal_forecast": np.round(forecast, 1),
        "naive_baseline": np.round(baseline, 1),
    })
    results["model_abs_error"] = (results["actual"] - results["trend_seasonal_forecast"]).abs()
    results["naive_abs_error"] = (results["actual"] - results["naive_baseline"]).abs()
    return results


def summarize_validation(results: pd.DataFrame) -> dict:
    """Aggregate error metrics from validate_on_holdout()'s per-period results."""
    mae_model = results["model_abs_error"].mean()
    mae_naive = results["naive_abs_error"].mean()
    mape_model = (results["model_abs_error"] / results["actual"]).mean() * 100
    return {
        "mae_model": float(mae_model),
        "mae_naive": float(mae_naive),
        "mape_model_pct": float(mape_model),
        "improvement_over_naive_pct": float((1 - mae_model / mae_naive) * 100),
    }
