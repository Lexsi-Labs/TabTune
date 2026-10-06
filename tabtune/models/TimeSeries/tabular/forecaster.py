"""Any tabular regression model as a time series forecaster.

:class:`TabularForecasterAdapter` serves the generated ``TabularTS-<model>``
registry entries (one per TabTune tabular model with a regression head) and
the ``TabularTS-GBM`` baseline. Its checkpoint is the tabular model name
(``"TabICLv2"``) or ``"sklearn-gbm"``.

Two designs are available (:mod:`.features`), chosen with ``model_params["features"]``:

``"time"`` (default for foundation models)
    TabPFN-TS's recipe with any regressor. Each series is fitted on its own
    context rows, featurised by time only, and queried at the horizon rows.
``"lags"`` (default for the GBM baseline)
    One pooled direct multi-horizon regression over all series. Rows hold
    lags, rolling statistics, the seasonal lag, the horizon, calendar
    features of the target time and covariates (past-only ones included).

Targets are standardised per series with context statistics (``scaling``) and
forecasts are mapped back. Backends with native quantiles (the TabPFN family
and TabICLv2 through ``TabularPipeline``, and the GBM baseline's quantile-loss
models) return them; the other models are point forecasters, so use
:meth:`TimeSeriesPipeline.calibrate` for intervals.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ....registry.errors import ConfigError
from .._params import as_int, read_params
from ..base import AdapterOutput, TSFMAdapter
from .backends import PipelineBackend, TabularRegressorBackend, gbm_backend
from .features import AUTO_SEASONAL_DEFAULTS, lag_design, time_design
from .tabpfn_ts import handle_missing

if TYPE_CHECKING:
    from ....config.schemas import ForecastConfig
    from ....registry.TimeSeries import TimeSeriesModelSpec
    from ....TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["GBM_CHECKPOINT", "TabularForecasterAdapter"]

GBM_CHECKPOINT = "sklearn-gbm"
_FEATURES = ("time", "lags")
_KNOWN_KEYS = (
    "features",
    "max_context",
    "max_top_k",
    "lags",
    "season_length",
    "windows",
    "seasonal_lags",
    "max_train_rows",
    "scaling",
    "model_params",
    "processor_params",
    "backend",
    "max_iter",
    "anchor_levels",
    "point_forecast",
)


def _standardise(y: np.ndarray, method: str) -> tuple[np.ndarray, float, float]:
    if method == "none" or y.size == 0:
        return y, 0.0, 1.0
    loc = float(np.mean(y))
    scale = float(np.std(y))
    if not np.isfinite(scale) or scale <= 1e-8 * max(1.0, abs(loc)):
        scale = 1.0
    return (y - loc) / scale, loc, scale


class TabularForecasterAdapter(TSFMAdapter):
    """Tabular regressor as forecaster.

    ``model_params``:
        features: ``"time"`` or ``"lags"`` (default from the spec: ``"time"``
            for foundation models, ``"lags"`` for the GBM baseline).
        max_context: Rows per series in the time design (default 4,096; the
            engine has already cut to the spec's limit).
        max_top_k: Seasonal periods in the time design (default 12).
        lags, season, windows, seasonal_lags: Lag design settings (defaults:
            two seasons of lags capped at 64, the frequency's natural season,
            windows (7, 28), 3 same-phase seasonal lags).
        max_train_rows: Row budget of the pooled lag design (default 10,000).
        scaling: ``"standard"`` (default) or ``"none"``.
        point: ``"median"`` (default) makes the point forecast the 0.5
            quantile when the backend has quantiles; ``"model"`` keeps the
            backend's own point prediction.
        model_params, processor_params: Forwarded to ``TabularPipeline``.
        backend: A :class:`~.backends.TabularRegressorBackend` instance to use
            instead (for custom regressors and tests).
        max_iter, anchor_levels: GBM baseline only: boosting iterations
            (default 100) and the quantile levels actually fitted (default
            0.1/0.5/0.9, others probit-interpolated; ``None`` fits every level).
    """

    def __init__(
        self,
        spec: TimeSeriesModelSpec,
        *,
        checkpoint: str,
        device: str,
        model_params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            spec, checkpoint=checkpoint, device=device, model_params=model_params, seed=seed
        )
        params = self.model_params = read_params(
            self.model_params, name=spec.name, known=_KNOWN_KEYS
        )
        default = "lags" if checkpoint == GBM_CHECKPOINT else "time"
        self.features = params.get("features", default)
        if self.features not in _FEATURES:
            raise ConfigError(
                f"{spec.name} features must be one of {_FEATURES}, got {self.features!r}."
            )
        self.max_context = as_int(params, "max_context", 4096, name=spec.name, minimum=1)
        self.max_top_k = as_int(
            params, "max_top_k", AUTO_SEASONAL_DEFAULTS["max_top_k"], name=spec.name, minimum=1
        )
        self.max_train_rows = as_int(
            params, "max_train_rows", 10_000, name=spec.name, minimum=1
        )
        self.scaling = params.get("scaling", "standard")
        if self.scaling not in ("standard", "none"):
            raise ConfigError(
                f"{spec.name} scaling must be 'standard' or 'none', got {self.scaling!r}."
            )
        self.point = params.get("point_forecast", "median")
        if self.point not in ("median", "model"):
            raise ConfigError(
                f"{spec.name} point_forecast must be 'median' or 'model', got {self.point!r}."
            )
        for key in ("lags", "windows", "seasonal_lags", "season_length", "max_iter"):
            if params.get(key) is not None and not isinstance(params[key], (list, tuple)):
                as_int(params, key, 1, name=spec.name, minimum=1)
        for name, value in (
            ("max_context", self.max_context),
            ("max_top_k", self.max_top_k),
            ("max_train_rows", self.max_train_rows),
        ):
            if value < 1:
                raise ConfigError(f"{name} must be >= 1, got {value}.")
        backend = params.get("backend")
        if backend is not None and not isinstance(backend, TabularRegressorBackend):
            raise ConfigError("backend must be a TabularRegressorBackend instance.")
        self._custom_backend = backend

    @property
    def respects_seed(self) -> bool:
        """True for the GBM baseline only; TabPFN's randomness is set via ``estimator_kwargs``."""
        return self.checkpoint == GBM_CHECKPOINT

    @property
    def native_quantiles(self) -> bool:
        return bool(self._model is not None and self._model.native_quantiles)

    def load(self) -> None:
        if self._custom_backend is not None:
            self._model = self._custom_backend
        elif self.checkpoint == GBM_CHECKPOINT:
            self._model = gbm_backend(
                seed=self.seed if self.seed is not None else 0,
                max_iter=int(self.model_params.get("max_iter", 100)),
                anchor_levels=self.model_params.get("anchor_levels", (0.1, 0.5, 0.9)),
            )
        else:
            self._model = PipelineBackend(
                self.checkpoint,
                model_params=self.model_params.get("model_params"),
                processor_params=self.model_params.get("processor_params"),
                device=self.device,
            )
        self._median_point = self.point == "median" and self._model.native_quantiles
        self.point_forecast = (
            "median" if self._median_point else getattr(self._model, "point_statistic", "mean")
        )

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        horizon = int(config.prediction_length)
        levels = [float(q) for q in config.quantile_levels] if self._model.native_quantiles else []
        query = sorted({*levels, 0.5}) if self._median_point else levels
        if self.features == "time":
            point, quantiles = self._forecast_time(panel, horizon, query)
        else:
            point, quantiles = self._forecast_lags(panel, horizon, query)
        if self._median_point:
            point = quantiles[..., query.index(0.5)]
            quantiles = quantiles[..., [query.index(q) for q in levels]] if levels else None
        return AdapterOutput(point=point, quantiles=quantiles)

    def _forecast_time(
        self, panel: TimeSeriesPanel, horizon: int, levels: list[float]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        point = np.empty((len(panel), horizon))
        quantiles = np.empty((len(panel), horizon, len(levels))) if levels else None
        for i, values in enumerate(panel.values):
            stamps = pd.date_range(end=panel.last_timestamps[i], periods=len(values), freq=panel.freq)
            y, stamps, kept = handle_missing(np.asarray(values, dtype=float), stamps)
            y, stamps, kept = y[-self.max_context :], stamps[-self.max_context :], kept[-self.max_context :]
            future = pd.date_range(start=panel.last_timestamps[i], periods=horizon + 1, freq=panel.freq)[1:]
            past_cov, future_cov = {}, {}
            block, fut = panel.past_covariates[i], panel.future_covariates[i]
            if fut:
                past_cov = {k: np.asarray(block[k])[kept] for k in block if k in fut}
                future_cov = {k: np.asarray(fut[k])[:horizon] for k in past_cov}
            X_train, y_train, X_test = time_design(
                y, stamps, future, past_covariates=past_cov, future_covariates=future_cov,
                max_top_k=self.max_top_k,
            )
            z, loc, scale = _standardise(y_train, self.scaling)
            p, q = self._model.clone().fit(X_train, z).predict(X_test, levels)
            point[i] = p * scale + loc
            if quantiles is not None:
                quantiles[i] = q * scale + loc
        return point, quantiles

    def _forecast_lags(
        self, panel: TimeSeriesPanel, horizon: int, levels: list[float]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        from ....TimeSeries.metrics import season_length

        season = self.model_params.get("season_length")
        season = int(season) if season is not None else season_length(panel.freq, natural=True)
        lags = int(self.model_params.get("lags", min(64, max(8, 2 * season))))
        windows = tuple(int(w) for w in self.model_params.get("windows", (7, 28)))
        design = lag_design(
            panel.values,
            panel.last_timestamps,
            panel.freq,
            horizon,
            lags=lags,
            season=season,
            windows=windows,
            seasonal_lags=int(self.model_params.get("seasonal_lags", 3)),
            max_train_rows=self.max_train_rows,
            scaling=self.scaling,
            past_covariates=panel.past_covariates,
            future_covariates=panel.future_covariates,
        )
        if len(design.y_train) == 0:
            raise ConfigError(
                f"{self.spec.name}: no training rows (every series is shorter than the horizon "
                f"{horizon} plus one observation); use features='time' or a shorter horizon."
            )
        backend = self._model.clone().fit(design.X_train, design.y_train)
        p, q = backend.predict(design.X_test, levels)
        n = len(panel)
        point = p.reshape(n, horizon) * design.scale[:, None] + design.loc[:, None]
        quantiles = None
        if q is not None:
            quantiles = (
                q.reshape(n, horizon, len(levels)) * design.scale[:, None, None]
                + design.loc[:, None, None]
            )
        return point, quantiles
