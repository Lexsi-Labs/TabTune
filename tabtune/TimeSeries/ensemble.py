"""Ensembles of time series pipelines, the counterpart of ``TabularEnsemble``.

Members are ``TimeSeriesPipeline`` configurations, so foundation models,
tabular forecasters and statistical baselines mix freely. Quantiles are
combined level by level (Vincentization): the ensemble's q-quantile is a
weighted average of the members' q-quantiles.

Weights are learned on validation windows at the end of the history. A
zero-shot member is scored on rolling-origin forecasts over them; a
fine-tuned member is first trained without them and refitted on the whole
history afterwards.

Strategies:

=====================  =====================================================
``mean``, ``median``   uniform weights, or the per-level median (no fitting)
``weighted_averaging`` weights proportional to ``1 / validation loss``
``greedy_selection``   Caruana et al. (2004) ensemble selection with
                       replacement, as in AutoGluon-TimeSeries (default)
``stacking``           non-negative least squares on the point forecasts
``best``               all weight on the best member
=====================  =====================================================

The validation loss is the weighted quantile loss when every member
forecasts quantiles, else the absolute error scaled by ``sum |y|``. When some
members are point-only, the quantiles are combined over the
probabilistic members with renormalised weights and shifted so their median
sits on the ensemble point.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..logger import log_table, logged_operation
from .forecast import ForecastResult
from .schema import TimeSeriesSchema
from .tasks import check_forecasting_only

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesEnsemble", "STRATEGIES"]

STRATEGIES = ("greedy_selection", "weighted_averaging", "stacking", "best", "mean", "median")


class TimeSeriesEnsemble:
    """Fit several time series pipelines and combine their forecasts.

    Args:
        models: Member configurations: model names, or dicts of
            ``TimeSeriesPipeline`` arguments with ``model_name`` and an
            optional ``label``.
        ensemble_strategy: One of :data:`STRATEGIES`.
        forecast_params: Shared by every member; ``prediction_length`` is
            required.
        validation_windows: Windows at the end of the history used to learn
            the weights.
        greedy_ensemble_size: Rounds of greedy selection.
        verbose: Log each member's validation loss.

    Attributes:
        pipelines_: Fitted members, by label.
        weights_: Ensemble weight of each member, by label.
        validation_losses_: Validation loss of each member, by label.
        loss_name_: ``"wql"`` or ``"wape"``.

    Example:
        >>> ens = TimeSeriesEnsemble(["SeasonalNaive", "ChronosBolt", "TiRex"],
        ...                          forecast_params={"prediction_length": 24})   # doctest: +SKIP
        >>> ens.fit(history, schema).predict().to_pandas()                           # doctest: +SKIP
    """

    def __init__(
        self,
        models: Sequence[str | dict[str, Any]],
        ensemble_strategy: str = "greedy_selection",
        *,
        forecast_params: dict,
        validation_windows: int = 1,
        greedy_ensemble_size: int = 100,
        verbose: bool = True,
    ) -> None:
        if ensemble_strategy not in STRATEGIES:
            raise ValueError(f"ensemble_strategy must be one of {STRATEGIES}, got {ensemble_strategy!r}")
        if not models:
            raise ValueError("models must name at least one member")
        if "prediction_length" not in (forecast_params or {}):
            raise ValueError("forecast_params must include prediction_length")
        if validation_windows < 1:
            raise ValueError("validation_windows must be >= 1")
        self.models = [m if isinstance(m, dict) else {"model_name": m} for m in models]
        for config in self.models:
            check_forecasting_only(config, context="TimeSeriesEnsemble")
        self.ensemble_strategy = ensemble_strategy
        self.forecast_params = dict(forecast_params)
        self.validation_windows = validation_windows
        self.greedy_ensemble_size = greedy_ensemble_size
        self.verbose = verbose
        labels = [self._label(m) for m in self.models]
        duplicates = sorted({x for x in labels if labels.count(x) > 1})
        if duplicates:
            raise ValueError(f"Member labels must be unique; repeated: {duplicates}. Pass 'label'.")
        self.pipelines_: dict[str, Any] = {}
        self.weights_: dict[str, float] = {}
        self.validation_losses_: dict[str, float] = {}
        self.fit_times_: dict[str, float] = {}
        self.loss_name_: str | None = None
        self.schema_: TimeSeriesSchema | None = None

    @staticmethod
    def _label(config: dict[str, Any]) -> str:
        if config.get("label"):
            return str(config["label"])
        strategy = config.get("tuning_strategy", "inference")
        return config["model_name"] if strategy == "inference" else f"{config['model_name']} / {strategy}"

    @logged_operation("ensemble_fit")
    def fit(
        self,
        df: pd.DataFrame,
        schema: TimeSeriesSchema,
        *,
        future_df: pd.DataFrame | None = None,
    ) -> TimeSeriesEnsemble:
        """Fit every member and learn the ensemble weights.

        Args:
            df: Long-format history.
            schema: Column roles.
            future_df: Known covariates over the horizon after ``df``.

        Raises:
            RuntimeError: If no member can be fitted and validated.
        """
        from .data import split_horizon
        from .pipeline import TimeSeriesPipeline

        horizon = self.forecast_params["prediction_length"]
        held_out = horizon * self.validation_windows
        train, _ = split_horizon(df, schema, held_out)
        self.schema_ = schema
        self.pipelines_, self.fit_times_ = {}, {}
        rows: dict[str, pd.DataFrame] = {}
        for config in self.models:
            label = self._label(config)
            kwargs = {k: v for k, v in config.items() if k not in ("model_name", "label")}
            if self.verbose:
                logger.info("[TimeSeriesEnsemble] Fitting and validating member %s", label)
            try:
                started = time.perf_counter()
                pipe = TimeSeriesPipeline(
                    config["model_name"], forecast_params=self.forecast_params, **kwargs
                )
                trains = pipe.tuning_strategy != "inference"
                pipe.fit(train if trains else df, schema, future_df=None if trains else future_df)
                panel = schema.to_panel(df, native_missing=pipe._accepts_missing())
                rows[label] = pipe._backtest_rows(panel, self.validation_windows, None)
                if trains:
                    pipe.fit(df, schema, future_df=future_df)
                self.fit_times_[label] = time.perf_counter() - started
                self.pipelines_[label] = pipe
            except Exception as exc:
                logger.error("[TimeSeriesEnsemble] %s failed: %s: %s", label, type(exc).__name__, exc)
        if not self.pipelines_:
            raise RuntimeError("No ensemble member could be fitted and validated.")
        self._learn_weights(rows)
        return self

    def _learn_weights(self, rows: dict[str, pd.DataFrame]) -> None:
        labels = list(rows)
        levels = [str(q) for q in self.forecast_params.get("quantile_levels", [0.1, 0.5, 0.9])]
        probabilistic = all(rows[k].attrs["quantiles"] for k in labels) and bool(levels)
        self.loss_name_ = "wql" if probabilistic else "wape"
        y = rows[labels[0]]["__actual__"].to_numpy(dtype=float)
        points = np.stack([rows[k]["point"].to_numpy(dtype=float) for k in labels])
        quantiles = (
            np.stack([rows[k][levels].to_numpy(dtype=float) for k in labels]) if probabilistic else None
        )
        q = np.asarray([float(x) for x in levels])

        def loss(weights: np.ndarray) -> float:
            if quantiles is not None:
                combined = np.tensordot(weights, quantiles, axes=1)
                err = y[:, None] - combined
                return float(2 * np.maximum(q * err, (q - 1) * err).sum() / (np.abs(y).sum() * len(q) + 1e-12))
            return float(np.abs(y - weights @ points).sum() / (np.abs(y).sum() + 1e-12))

        eye = np.eye(len(labels))
        member_losses = np.array([loss(eye[i]) for i in range(len(labels))])
        self.validation_losses_ = dict(zip(labels, member_losses.tolist(), strict=True))
        strategy = self.ensemble_strategy
        if strategy in ("mean", "median"):
            weights = np.full(len(labels), 1.0 / len(labels))
        elif strategy == "best":
            weights = eye[int(np.argmin(member_losses))]
        elif strategy == "weighted_averaging":
            inverse = 1.0 / np.maximum(member_losses, 1e-12)
            weights = inverse / inverse.sum()
        elif strategy == "stacking":
            from scipy.optimize import nnls

            coef, _ = nnls(points.T, y)
            weights = coef / coef.sum() if coef.sum() > 0 else eye[int(np.argmin(member_losses))]
        else:
            counts = np.zeros(len(labels))
            for _ in range(self.greedy_ensemble_size):
                trials = [loss((counts + eye[i]) / (counts.sum() + 1)) for i in range(len(labels))]
                counts[int(np.argmin(trials))] += 1
            weights = counts / counts.sum()
        self.weights_ = dict(zip(labels, weights.tolist(), strict=True))
        if self.verbose:
            log_table(logger, f"Ensemble weights / {strategy}",
                      ["Model", self.loss_name_.upper(), "Weight"],
                      [(label, self.validation_losses_[label], self.weights_[label]) for label in labels])

    @logged_operation("ensemble_predict")
    def predict(
        self, df: pd.DataFrame | None = None, *, future_df: pd.DataFrame | None = None
    ) -> ForecastResult:
        """Combine the members' forecasts from the fitted history or from ``df``."""
        if not self.pipelines_:
            raise RuntimeError("You must call fit() on the ensemble before predict().")
        active = {k: w for k, w in self.weights_.items() if w > 0 or self.ensemble_strategy == "median"}
        forecasts = {k: self.pipelines_[k].predict(df, future_df=future_df) for k in active}
        first = next(iter(forecasts.values()))
        labels = list(forecasts)
        weights = np.array([active[k] for k in labels])
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(labels), 1 / len(labels))
        points = np.stack([forecasts[k].point for k in labels])
        with_q = [k for k in labels if forecasts[k].quantiles is not None]
        if self.ensemble_strategy == "median":
            point = np.median(points, axis=0)
        else:
            point = np.tensordot(weights, points, axes=1)
        quantiles, levels = None, ()
        if with_q:
            levels = forecasts[with_q[0]].quantile_levels
            stack = np.stack([forecasts[k].quantiles for k in with_q])
            if self.ensemble_strategy == "median":
                quantiles = np.median(stack, axis=0)
            else:
                w = np.array([active[k] for k in with_q])
                w = w / w.sum() if w.sum() > 0 else np.full(len(with_q), 1 / len(with_q))
                quantiles = np.tensordot(w, stack, axes=1)
            if len(with_q) < len(labels):
                median = _median_level(quantiles, levels)
                if median is not None:
                    quantiles = quantiles + (point - median)[..., None]
            quantiles = np.sort(quantiles, axis=-1)
        return ForecastResult(
            item_ids=first.item_ids,
            timestamps=first.timestamps,
            point=point,
            quantiles=quantiles,
            quantile_levels=tuple(levels),
            targets=first.targets,
            item_id_column=first.item_id_column,
            timestamp_column=first.timestamp_column,
            metadata={
                "model": "TimeSeriesEnsemble",
                "checkpoint": None,
                "device": None,
                "tuning_strategy": None,
                "training_occurred": any(
                    bool(getattr(pipe, "training_occurred_", False))
                    for pipe in self.pipelines_.values()
                ),
                "strategy": self.ensemble_strategy,
                "ensemble_strategy": self.ensemble_strategy,
                "members": {k: round(float(active[k]), 6) for k in labels},
            },
        )

    def evaluate(
        self,
        df_actual: pd.DataFrame,
        *,
        forecast: ForecastResult | None = None,
        output_format: str = "rich",
    ) -> dict[str, Any]:
        """Score the ensemble forecast as :meth:`TimeSeriesPipeline.evaluate` does."""
        if not self.pipelines_:
            raise RuntimeError("You must call fit() on the ensemble before evaluate().")
        forecast = forecast if forecast is not None else self.predict()
        reference = next(iter(self.pipelines_.values()))
        return reference.evaluate(df_actual, forecast=forecast, output_format=output_format)

    def get_leaderboard(self) -> pd.DataFrame:
        """Each member's validation loss, weight and fit time, best first."""
        frame = pd.DataFrame(
            {
                "member": list(self.validation_losses_),
                f"validation_{self.loss_name_}": list(self.validation_losses_.values()),
                "weight": [self.weights_.get(k, 0.0) for k in self.validation_losses_],
                "fit_s": [self.fit_times_.get(k, np.nan) for k in self.validation_losses_],
            }
        )
        return frame.sort_values(f"validation_{self.loss_name_}").reset_index(drop=True)

    def to_json(self) -> str:
        """Members, weights and validation losses as JSON."""
        return json.dumps(
            {
                "strategy": self.ensemble_strategy,
                "loss": self.loss_name_,
                "weights": self.weights_,
                "validation_losses": self.validation_losses_,
            },
            indent=2,
        )

    def __repr__(self) -> str:
        return (
            f"TimeSeriesEnsemble(strategy={self.ensemble_strategy!r}, members={len(self.models)}, "
            f"fitted={len(self.pipelines_)})"
        )


def _median_level(quantiles: np.ndarray, levels: Sequence[float]) -> np.ndarray | None:
    """The 0.5 quantile, interpolated from the neighbouring levels when absent."""
    grid = np.asarray(levels, dtype=float)
    if grid.size == 0 or grid[0] > 0.5 or grid[-1] < 0.5:
        return None
    j = int(np.searchsorted(grid, 0.5))
    if abs(grid[j] - 0.5) < 1e-12:
        return quantiles[..., j]
    w = (0.5 - grid[j - 1]) / (grid[j] - grid[j - 1])
    return (1 - w) * quantiles[..., j - 1] + w * quantiles[..., j]
