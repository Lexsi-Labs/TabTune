"""TabPFN-TS: zero-shot forecasting as tabular regression (Hoo et al., 2025).

A re-implementation of ``TabPFNTSPipeline.predict`` from
PriorLabs/tabpfn-time-series v1.3.0 on TabTune's vendored TabPFN regressors.
Per series: rows with a missing target are dropped (a series with at most one
observation is zero-filled instead), the last ``max_context_length`` rows are
kept, only covariates known over the horizon are used, time features are built
by :func:`~.features.time_design`, and a fresh TabPFN regressor fitted on the
context rows is queried at the horizon rows with ``output_type="main"``. The
point is the ``"median"`` output; the quantiles are TabPFN's own.

Checkpoints: ``tabpfn-ts-3.5`` (default; TabPFN-3.5, context 32,768, 12
periods), ``tabpfn-ts-3`` (``tabpfn-v3-regressor-v3_20260506_timeseries.ckpt``,
context 32,768, 12 periods) and ``tabpfn-ts-2`` (``tabpfn-v2-regressor-2noar4o2.ckpt``,
context 4,096, 5 periods). A constant target returns that constant, non-numeric
known covariates become integer codes, and the cloud client is not offered.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ....registry.errors import ConfigError
from .._params import as_bool, as_int, read_params
from ..base import AdapterOutput, TSFMAdapter
from .backends import TabPFNBackend
from .features import time_design

if TYPE_CHECKING:
    from ....config.schemas import ForecastConfig
    from ....registry.TimeSeries import TimeSeriesModelSpec
    from ....TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["TABPFN_TS_VARIANTS", "TabPFNTSAdapter", "handle_missing"]

TABPFN_TS_VARIANTS: dict[str, dict[str, Any]] = {
    "tabpfn-ts-3.5": {
        "version": "v3.5",
        "model_path": None,
        "max_context_length": 32768,
        "max_top_k": 12,
        "release": "tabpfn-time-series 1.3.0 (2026-09-16)",
    },
    "tabpfn-ts-3": {
        "version": "v3",
        "model_path": "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
        "max_context_length": 32768,
        "max_top_k": 12,
        "release": "tabpfn-time-series 1.1.0 (2026-05-12)",
    },
    "tabpfn-ts-2": {
        "version": "v2",
        "model_path": "tabpfn-v2-regressor-2noar4o2.ckpt",
        "max_context_length": 4096,
        "max_top_k": 5,
        "release": "tabpfn-time-series 1.0.x (paper configuration)",
    },
}

_POINT_STATISTICS = ("median", "mean", "mode")

_KNOWN_KEYS = (
    "max_context",
    "max_top_k",
    "point_forecast",
    "model_path",
    "use_covariates",
    "estimator_kwargs",
)


def handle_missing(
    values: np.ndarray, timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Upstream's missing-value rule; returns ``(values, timestamps, kept row indices)``.

    At most one observed value: fill the gaps with 0 and keep every row.
    Otherwise: drop the rows with a missing target.
    """
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    if observed.sum() <= 1:
        return np.where(observed, values, 0.0), pd.DatetimeIndex(timestamps), np.arange(len(values))
    keep = np.flatnonzero(observed)
    return values[keep], pd.DatetimeIndex(timestamps)[keep], keep


class TabPFNTSAdapter(TSFMAdapter):
    """TabPFN-TS forecaster (per-series TabPFN regression on time features).

    The checkpoint selects the upstream configuration (see :data:`TABPFN_TS_VARIANTS`).
    ``model_params`` override single settings:

    max_context_length: Rows of history kept per series.
    max_top_k: Seasonal periods encoded (``AutoSeasonalFeature.max_top_k``).
    output_selection: Point forecast: ``"median"`` (default), ``"mean"`` or ``"mode"``.
    model_path: TabPFN checkpoint file, instead of the variant's.
    use_covariates: Use known covariates (default ``True``).
    estimator_kwargs: Extra ``TabPFNRegressor`` arguments (e.g. ``n_estimators``).
    """

    point_forecast = "median"

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
        if checkpoint not in TABPFN_TS_VARIANTS:
            raise ConfigError(
                f"TabPFN-TS checkpoint must be one of {list(TABPFN_TS_VARIANTS)}, got {checkpoint!r}."
            )
        params = self.model_params = read_params(
            self.model_params, name=spec.name, known=_KNOWN_KEYS
        )
        variant = TABPFN_TS_VARIANTS[checkpoint]
        self.version = variant["version"]
        self.model_path = params.get("model_path", variant["model_path"])
        self.max_context_length = as_int(
            params, "max_context", variant["max_context_length"], name=spec.name, minimum=1
        )
        self.max_top_k = as_int(
            params, "max_top_k", variant["max_top_k"], name=spec.name, minimum=1
        )
        self.output_selection = params.get("point_forecast", "median")
        if self.output_selection not in _POINT_STATISTICS:
            raise ConfigError(
                f"{spec.name} point_forecast must be one of {_POINT_STATISTICS}, "
                f"got {self.output_selection!r}"
            )
        self.point_forecast = self.output_selection
        self.use_covariates = as_bool(params, "use_covariates", True, name=spec.name)
        self.estimator_kwargs = dict(params.get("estimator_kwargs") or {})
        self._template = self._backend()

    def _backend(self) -> TabPFNBackend:
        return TabPFNBackend(
            self.version,
            model_path=self.model_path,
            output_selection=self.output_selection,
            device=self.device,
            estimator_kwargs=self.estimator_kwargs,
            factory=self.model_params.get("_factory"),
        )

    def load(self) -> None:
        self._model = self._backend()

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        horizon = int(config.prediction_length)
        levels = [float(q) for q in config.quantile_levels]
        point = np.empty((len(panel), horizon))
        quantiles = np.empty((len(panel), horizon, len(levels))) if levels else None
        for i, values in enumerate(panel.values):
            stamps = pd.date_range(end=panel.last_timestamps[i], periods=len(values), freq=panel.freq)
            y, kept_stamps, kept = handle_missing(np.asarray(values, dtype=float), stamps)
            y, kept_stamps, kept = (
                y[-self.max_context_length :],
                kept_stamps[-self.max_context_length :],
                kept[-self.max_context_length :],
            )
            future_stamps = pd.date_range(
                start=panel.last_timestamps[i], periods=horizon + 1, freq=panel.freq
            )[1:]
            past, future = self._covariates(panel, i, kept, horizon)
            X_train, y_train, X_test = time_design(
                y,
                kept_stamps,
                future_stamps,
                past_covariates=past,
                future_covariates=future,
                max_top_k=self.max_top_k,
            )
            backend = self._model.clone().fit(X_train, y_train)
            p, q = backend.predict(X_test, levels)
            point[i] = p
            if quantiles is not None:
                quantiles[i] = q
        return AdapterOutput(point=point, quantiles=quantiles)

    def _covariates(
        self, panel: TimeSeriesPanel, row: int, kept: np.ndarray, horizon: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Known covariates of one series, aligned with the kept context rows."""
        if not self.use_covariates:
            return {}, {}
        past_block = panel.past_covariates[row]
        future_block = panel.future_covariates[row]
        past = {
            name: np.asarray(past_block[name])[kept]
            for name in past_block
            if name in future_block
        }
        future = {name: np.asarray(future_block[name])[:horizon] for name in past}
        return past, future
