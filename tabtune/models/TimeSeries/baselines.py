"""Statistical baselines behind the adapter interface.

These need no weights or network. Prediction intervals are Gaussian with the benchmark-method standard errors of
Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.,
section 5.5), estimated from in-sample one-step residuals:

=====================  ================================================
Method                 h-step standard deviation
=====================  ================================================
Naive                  ``sigma * sqrt(h)``
Seasonal naive         ``sigma * sqrt(k + 1)``, ``k = floor((h - 1) / m)``
Mean                   ``sigma * sqrt(1 + 1 / T)``
Drift                  ``sigma * sqrt(h * (1 + h / (T - 1)))``
=====================  ================================================

Missing values in the history are skipped when estimating levels and
residuals, so the baselines accept ``NaN`` natively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import AdapterOutput, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...TimeSeries.schema import TimeSeriesPanel

__all__ = [
    "NaiveAdapter",
    "SeasonalNaiveAdapter",
    "MeanAdapter",
    "DriftAdapter",
    "WindowAverageAdapter",
]


def _normal_ppf(levels: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.ppf(levels)


class _BaselineAdapter(TSFMAdapter):
    """Shared machinery: NaN-aware histories and Gaussian quantiles."""

    point_forecast = "mean"
    known_params: tuple[str, ...] = ("season_length",)

    def __init__(self, spec: Any, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        from ...registry.errors import ConfigError
        from ._params import as_int, read_params

        # These models hold no weights: dtype and batch_size are validated but unused.
        self.model_params = read_params(
            self.model_params,
            name=spec.name,
            known=(*self.known_params, "batch_size", "dtype"),
        )
        self._resolve_dtype()
        if "batch_size" in self.model_params:
            as_int(self.model_params, "batch_size", 1, name=spec.name, minimum=1)
        for key in self.known_params:
            value = self.model_params.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not float(value).is_integer() or int(value) < 1:
                raise ConfigError(
                    f"{spec.name} model_params[{key!r}] must be a positive integer, got {value!r}."
                )

    def load(self) -> None:
        self._model = type(self).__name__

    def _season(self, panel: TimeSeriesPanel) -> int:
        from ...TimeSeries.metrics import season_length

        value = self.model_params.get("season_length")
        return int(value) if value else season_length(panel.freq, natural=True)

    def _point_and_sigma(
        self, values: np.ndarray, horizon: int, season: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        horizon = config.prediction_length
        levels = np.asarray(config.quantile_levels, dtype=float)
        season = self._season(panel)
        points = np.empty((len(panel), horizon))
        sigmas = np.empty((len(panel), horizon))
        for i, raw in enumerate(panel.values):
            values = np.asarray(raw, dtype=float)
            point, sigma = self._point_and_sigma(values, horizon, season)
            points[i] = point
            sigmas[i] = np.nan_to_num(sigma, nan=0.0)
        quantiles = None
        if len(levels):
            z = _normal_ppf(levels)
            quantiles = points[:, :, None] + sigmas[:, :, None] * z[None, None, :]
        return AdapterOutput(point=points, quantiles=quantiles)


def _last_observed(values: np.ndarray) -> float:
    observed = values[~np.isnan(values)]
    return float(observed[-1])


def _residual_sigma(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    residuals = values[lag:] - values[:-lag]
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 2:
        return 0.0
    return float(np.sqrt(np.mean(residuals**2)))


class NaiveAdapter(_BaselineAdapter):
    """Repeat the last observed value."""

    def _point_and_sigma(self, values, horizon, season):
        steps = np.arange(1, horizon + 1)
        return np.full(horizon, _last_observed(values)), _residual_sigma(values, 1) * np.sqrt(steps)


class SeasonalNaiveAdapter(_BaselineAdapter):
    """Repeat the last observed seasonal cycle (``model_params['season_length']`` overrides)."""

    def _point_and_sigma(self, values, horizon, season):
        if season <= 1 or len(values) < season:
            return NaiveAdapter._point_and_sigma(self, values, horizon, 1)
        cycle = values[-season:].copy()
        if np.isnan(cycle).any():
            for j in np.where(np.isnan(cycle))[0]:
                position = len(values) - season + j
                earlier = values[position % season :: season][:-1]
                earlier = earlier[~np.isnan(earlier)]
                cycle[j] = earlier[-1] if len(earlier) else _last_observed(values)
        point = np.resize(cycle, horizon)
        k = (np.arange(1, horizon + 1) - 1) // season
        return point, _residual_sigma(values, season) * np.sqrt(k + 1)


class MeanAdapter(_BaselineAdapter):
    """Forecast the historical mean."""

    def _point_and_sigma(self, values, horizon, season):
        observed = values[~np.isnan(values)]
        mean = float(observed.mean())
        sigma = float(observed.std(ddof=1)) if len(observed) > 1 else 0.0
        return np.full(horizon, mean), np.full(horizon, sigma * np.sqrt(1 + 1 / len(observed)))


class DriftAdapter(_BaselineAdapter):
    """Extrapolate the line from the first to the last observation (random walk with drift)."""

    def _point_and_sigma(self, values, horizon, season):
        observed = values[~np.isnan(values)]
        n = len(observed)
        steps = np.arange(1, horizon + 1)
        if n < 2:
            return np.full(horizon, observed[-1]), np.zeros(horizon)
        slope = (observed[-1] - observed[0]) / (n - 1)
        point = observed[-1] + slope * steps
        residuals = np.diff(observed) - slope
        sigma = float(np.sqrt(np.sum(residuals**2) / max(n - 2, 1)))
        return point, sigma * np.sqrt(steps * (1 + steps / (n - 1)))


class WindowAverageAdapter(_BaselineAdapter):
    """Forecast the mean of the last ``window`` observations (``model_params['window']``, default one season)."""

    known_params = ("season_length", "window")

    def _point_and_sigma(self, values, horizon, season):
        window = int(self.model_params.get("window") or max(season, 1))
        recent = values[-window:]
        recent = recent[~np.isnan(recent)]
        if len(recent) == 0:
            recent = values[~np.isnan(values)][-1:]
        sigma = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
        return np.full(horizon, float(recent.mean())), np.full(
            horizon, sigma * np.sqrt(1 + 1 / len(recent))
        )
