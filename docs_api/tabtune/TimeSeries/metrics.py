"""Forecast accuracy metrics.

Point accuracy is reported as MAE, RMSE, MSE, sMAPE and MASE; probabilistic
accuracy as the mean pinball loss, the weighted quantile loss (WQL) and the
empirical coverage of the widest central interval. MASE and WQL follow the
fev-bench and GIFT-Eval definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ..evaluation.metrics import forecasting_metrics

__all__ = [
    "season_length",
    "seasonal_scale",
    "score_forecast",
    "LOWER_IS_BETTER",
]

LOWER_IS_BETTER: Mapping[str, bool] = {
    "mae": True,
    "rmse": True,
    "mse": True,
    "smape": True,
    "mase": True,
    "mean_pinball_loss": True,
    "wql": True,
}

_SEASONS = {"S": 3600, "T": 1440, "H": 24, "B": 5, "M": 12, "Q": 4}
_NATURAL_SEASONS = {**_SEASONS, "S": 60, "T": 60, "D": 7, "W": 52}
_BASE_CODES = {
    "s": "S", "min": "T", "t": "T", "h": "H", "b": "B", "d": "D", "w": "W",
    "m": "M", "ms": "M", "me": "M", "bm": "M", "bms": "M", "bme": "M",
    "q": "Q", "qs": "Q", "qe": "Q", "bq": "Q", "bqs": "Q", "bqe": "Q",
}  # fmt: skip


def season_length(freq: str, *, natural: bool = False) -> int:
    """Seasonal period of a pandas frequency.

    By default the GluonTS and fev-bench convention used for MASE: sub-daily
    frequencies use the daily cycle (24 for hourly, 1440 for minutely),
    monthly 12, quarterly 4, business-daily 5, and daily, weekly and yearly
    data 1. ``natural=True`` uses the shortest calendar cycle instead (a week
    for daily data, a year for weekly, an hour for minutely), which suits
    seasonal baselines. A multiple such as ``"15min"`` divides the period
    when it divides evenly, and gives 1 otherwise.
    """
    try:
        offset = to_offset(freq)
    except ValueError:
        return 1
    code = _BASE_CODES.get(offset.name.split("-")[0].lower())
    period = (_NATURAL_SEASONS if natural else _SEASONS).get(code, 1)
    n = max(int(getattr(offset, "n", 1) or 1), 1)
    return period // n if period % n == 0 else 1


def seasonal_scale(history: np.ndarray, season: int) -> float:
    """Mean absolute seasonal difference of ``history``: the MASE denominator.

    Falls back to lag 1 when the history is shorter than two seasons, and
    returns ``NaN`` when no difference can be computed or all are zero.
    """
    values = np.asarray(history, dtype=float)
    for lag in (season, 1):
        if lag >= 1 and len(values) > lag:
            diffs = np.abs(values[lag:] - values[:-lag])
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size and diffs.mean() > 0:
                return float(diffs.mean())
    return float("nan")


def score_forecast(
    rows: pd.DataFrame,
    levels: Sequence[float],
    scales: Mapping[Any, float] | None = None,
    *,
    series_key: Sequence[str] = ("target",),
) -> dict[str, float]:
    """Metrics for aligned forecasts.

    Args:
        rows: One row per forecast point with ``__actual__``, ``point``, one
            column per quantile level named ``str(level)`` and the columns in
            ``series_key``.
        levels: The quantile levels present in ``rows``.
        scales: MASE denominator per series, keyed by the tuple of
            ``series_key`` values (or the bare value for a one-column key).
        series_key: Columns identifying a series.

    Returns:
        ``mae``, ``rmse``, ``mse``, ``smape`` and, when ``scales`` are given,
        ``mase``; with quantiles also ``mean_pinball_loss``, ``wql`` and
        ``coverage_<p>`` for the widest central interval.
    """
    y = rows["__actual__"].to_numpy(dtype=float)
    point = rows["point"].to_numpy(dtype=float)
    quantile_cols = [str(q) for q in levels]
    quantiles = rows[quantile_cols].to_numpy(dtype=float) if levels else None
    results = forecasting_metrics(y, point, quantiles, list(levels))

    denom = np.abs(y) + np.abs(point)
    ratio = np.divide(2 * np.abs(y - point), denom, out=np.zeros_like(y), where=denom > 0)
    results["smape"] = float(np.mean(ratio))

    if scales:
        keys = list(series_key)
        per_series = []
        for key, group in rows.groupby(keys, sort=False):
            key = key[0] if isinstance(key, tuple) and len(key) == 1 else key
            scale = scales.get(key)
            if scale is None or not np.isfinite(scale) or scale <= 0:
                continue
            err = np.abs(group["__actual__"].to_numpy(float) - group["point"].to_numpy(float))
            per_series.append(err.mean() / scale)
        if per_series:
            results["mase"] = float(np.mean(per_series))

    if quantiles is not None:
        q = np.asarray(levels, dtype=float)
        err = y[:, None] - quantiles
        loss = np.maximum(q * err, (q - 1) * err)
        total = np.abs(y).sum()
        if total > 0:
            results["wql"] = float(2 * loss.sum() / (total * len(q)))
        interval = _widest_central_interval(levels)
        if interval is not None:
            lo, hi, width = interval
            inside = (y >= quantiles[:, lo]) & (y <= quantiles[:, hi])
            results[f"coverage_{width}"] = float(inside.mean())
    return results


def _widest_central_interval(levels: Sequence[float]) -> tuple[int, int, int] | None:
    """Column indices and nominal width (percent) of the widest symmetric pair of levels."""
    for i, low in enumerate(levels):
        if low >= 0.5:
            break
        for j in range(len(levels) - 1, i, -1):
            if abs(levels[j] - (1 - low)) < 1e-9:
                return i, j, int(round((levels[j] - low) * 100))
    return None
