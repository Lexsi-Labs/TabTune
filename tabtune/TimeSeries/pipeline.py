"""TimeSeriesPipeline: unified API for time series foundation models."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import pickle
from dataclasses import replace
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .._internal.deprecation import warn_once
from .._internal.device import resolve_device
from ..caching import make_cache
from ..config.schemas import ForecastConfig, TuningConfig
from ..logger import log_banner, log_event, log_metrics, log_table, logged_operation, stage, track
from ..models.TimeSeries.base import (
    AdapterOutput,
    TrainingSpec,
    TSFMAdapter,
    missing_extra_error,
)
from ..registry import check_license
from ..registry.errors import ConfigError, ModelNotFoundError, UnsupportedStrategyError
from ..registry.TimeSeries import (
    TimeSeriesModelSpec,
    check_forecast_envelope,
    check_schema_support,
    get_time_series_model_spec,
    validate_time_series_request,
)
from .backtest import rolling_windows
from .calibration import ConformalCalibrator
from .forecast import ForecastResult, future_timestamps
from .metrics import LOWER_IS_BETTER, score_forecast, season_length, seasonal_scale
from .schema import TimeSeriesPanel, TimeSeriesSchema
from .tasks import (
    AnomalyResult,
    EmbeddingResult,
    ImputationResult,
    check_forecasting_only,
    fill_gaps,
    rolled_timestamps,
    score_anomalies,
)

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesPipeline"]

_SUPPORTED_TASKS = ("forecasting", "anomaly_detection", "imputation", "embedding")
_SUPPORTED_STRATEGIES = ("inference", "finetune", "peft")

_INFERENCE_TUNING_KEYS = ("device", "seed")
_TRAINING_TUNING_KEYS = (
    "device",
    "seed",
    "epochs",
    "steps_per_epoch",
    "learning_rate",
    "batch_size",
    "early_stopping",
    "early_stopping_patience",
    "validation_split",
    "gradient_clip_norm",
    "peft_config",
)
_DEFAULT_STEPS_PER_EPOCH = 50

_TASK_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "forecasting": (),
    "embedding": (),
    "anomaly_detection": (
        "method",
        "stride",
        "min_context",
        "coverage",
        "alpha",
        "threshold",
        "contamination",
        "reference_fraction",
    ),
    "imputation": ("method", "min_context"),
}

_PIPELINE_MODEL_KEYS = ("checkpoint", "device")

_ENVELOPE_MODES = ("error", "warn", "ignore")
_LICENSE_MODES = ("research", "commercial", "ignore")
_OUTPUT_FORMATS = ("rich", "json")

_METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mse": "MSE",
    "smape": "sMAPE",
    "mase": "MASE",
    "mean_pinball_loss": "Mean Pinball Loss",
    "wql": "WQL",
}


def _metric_label(key: str) -> str:
    """Display name of a forecast metric, e.g. ``coverage_80`` -> ``Coverage (80%)``."""
    if key.startswith("coverage_"):
        return f"Coverage ({key[len('coverage_'):]}%)"
    return _METRIC_LABELS.get(key, key)


def _log_metrics(metrics: dict[str, Any]) -> None:
    log_metrics(logger, {k: v for k, v in metrics.items() if not isinstance(v, dict)},
                title="Forecast evaluation", labels={k: _metric_label(k) for k in metrics})


class TimeSeriesPipeline:
    """Fit, forecast with and evaluate a time series foundation model."""

    def __init__(
        self,
        model_name: str,
        task_type: str = "forecasting",
        tuning_strategy: str = "inference",
        tuning_params: dict | None = None,
        model_params: dict | None = None,
        forecast_params: dict | None = None,
        *,
        task_params: dict | None = None,
        cache: Any = None,
        envelope_mode: str = "warn",
        license_mode: str = "research",
        validate: bool = True,
    ) -> None:
        """Build a pipeline around a time series foundation model.

        Args:
            model_name: Model name or alias. See
                :func:`tabtune.registry.list_time_series_models`; register your
                own with :func:`tabtune.registry.register_time_series_model`.
            task_type: ``"forecasting"`` (default), ``"anomaly_detection"``,
                ``"imputation"`` or ``"embedding"``. :meth:`predict` returns a
                :class:`ForecastResult`, :class:`AnomalyResult`,
                :class:`ImputationResult` or :class:`EmbeddingResult`.
            tuning_strategy: ``"inference"`` (zero-shot), ``"finetune"`` (all
                weights) or ``"peft"`` (LoRA adapters), for models whose spec
                declares them.
            tuning_params: For zero-shot inference only ``device`` and
                ``seed`` apply. Fine-tuning also reads ``epochs`` and
                ``steps_per_epoch`` (one step is one batch of sampled
                windows; default 50 per epoch), ``learning_rate``,
                ``batch_size``, ``early_stopping`` with
                ``early_stopping_patience`` (in epochs), ``validation_split``
                (any value above 0 holds out the last ``prediction_length``
                steps of every series), ``gradient_clip_norm`` (default 1.0)
                and, for ``"peft"``, ``peft_config`` (``r``, ``lora_alpha``,
                ``lora_dropout``, ``target_modules``).
            model_params: ``checkpoint`` selects the weights: a validated
                checkpoint id (defaults to the spec's ``default_checkpoint``)
                or a local checkpoint directory. ``device`` selects the device
                when ``tuning_params`` does not. Other keys are forwarded to
                the model's adapter.
            forecast_params: Validated against
                :class:`~tabtune.config.ForecastConfig`. ``prediction_length``
                is required for forecasting; for anomaly detection it is the
                number of steps forecast from each origin (default 1).
            task_params: Task options. Anomaly detection: ``method``
                (``"forecast_error"``, ``"interval"`` or ``"likelihood"``),
                ``stride``, ``min_context``, ``coverage``, ``alpha``,
                ``threshold`` (``"conformal"``, ``"contamination"`` or a
                number), ``contamination`` and ``reference_fraction``.
                Imputation: ``method`` (``"bidirectional"`` or ``"forecast"``)
                and ``min_context``.
            cache: Result cache: ``'memory'``, ``'disk'``, ``None`` or a
                :class:`~tabtune.caching.PredictionCache`.
            envelope_mode: How to treat requests beyond the model's documented
                limits: ``'error'``, ``'warn'`` (default) or ``'ignore'``.
            license_mode: ``'research'`` (default), ``'commercial'`` to fail
                fast on weights that forbid commercial use, or ``'ignore'``.
            validate: Check the weight license and restrict ``checkpoint`` to
                the validated list (local directories are always allowed).
                The model, task and strategy are checked regardless, because
                the pipeline cannot run anything else.

        Raises:
            ModelNotFoundError: Unknown model name.
            UnsupportedTaskError: The model does not implement ``task_type``.
            UnsupportedStrategyError: The model does not support ``tuning_strategy``.
            LicenseError: ``license_mode='commercial'`` and the weights forbid it.
            ConfigError: Invalid ``forecast_params``, ``tuning_params``,
                ``task_params``, checkpoint, ``envelope_mode`` or ``license_mode``.
        """
        _check_choice("envelope_mode", envelope_mode, _ENVELOPE_MODES)
        _check_choice("license_mode", license_mode, _LICENSE_MODES)

        self.task_type = task_type
        self.tuning_strategy = tuning_strategy
        self.tuning_params = dict(tuning_params or {})
        self.model_params = dict(model_params or {})
        self.task_params = dict(task_params or {})
        self.envelope_mode = envelope_mode
        self.license_mode = license_mode
        self.validate = validate
        self.cache = make_cache(cache)

        self.spec: TimeSeriesModelSpec = _resolve_spec(model_name)

        validate_time_series_request(self.spec.name, task_type, tuning_strategy)
        if tuning_strategy not in _SUPPORTED_STRATEGIES:
            raise UnsupportedStrategyError(
                self.spec.name, tuning_strategy, task_type, _SUPPORTED_STRATEGIES
            )
        self.model_name = self.spec.name

        params = dict(forecast_params or {})
        if task_type != "forecasting":
            params.setdefault("prediction_length", 1)
        self.forecast_config = ForecastConfig.from_dict(params, context="forecast_params")
        self.tuning_config = TuningConfig.from_dict(self.tuning_params, context="tuning_params")
        allowed = _INFERENCE_TUNING_KEYS if tuning_strategy == "inference" else _TRAINING_TUNING_KEYS
        ignored = sorted(k for k in self.tuning_params if k not in allowed)
        if ignored:
            reason = (
                "no training happens for zero-shot inference"
                if tuning_strategy == "inference"
                else "time series fine-tuning does not use them"
            )
            warn_once(
                f"tuning_params {ignored} are ignored: {reason}. Only {list(allowed)} apply.",
                UserWarning,
                key=f"ts-ignored-tuning-params:{tuning_strategy}:{','.join(ignored)}",
            )
        unknown = sorted(k for k in self.task_params if k not in _TASK_PARAM_KEYS[task_type])
        if unknown:
            raise ConfigError(
                f"task_params {unknown} are not options of task_type={task_type!r}; "
                f"known: {list(_TASK_PARAM_KEYS[task_type])}"
            )

        context = self.forecast_config.context_length
        if context is not None and self.spec.max_context is not None and context > self.spec.max_context:
            warn_once(
                f"context_length={context} exceeds {self.model_name}'s maximum context of "
                f"{self.spec.max_context}; the last {self.spec.max_context} observations "
                f"are used.",
                UserWarning,
                key=f"ts-context-length:{self.model_name}:{context}",
            )

        self.checkpoint: str = self.model_params.get("checkpoint") or self.spec.default_checkpoint
        self._check_checkpoint()
        if validate:
            check_license(replace(self.spec, license=self.spec.license_for(self.checkpoint)), license_mode)

        self.adapter_: TSFMAdapter | None = None
        self.schema_: TimeSeriesSchema | None = None
        self.history_: TimeSeriesPanel | None = None
        self.calibrator_: ConformalCalibrator | None = None
        self.training_report_: dict[str, Any] | None = None
        self.training_occurred_ = False
        self._fit_digest: str | None = None
        self._calibration_digest: str | None = None
        self._is_fitted = False

        log_banner(logger, domain="timeseries")
        logger.info(
            "[TimeSeriesPipeline] Initialized for model '%s' (checkpoint '%s'), "
            "task '%s', strategy '%s'",
            self.model_name,
            self.checkpoint,
            self.task_type,
            self.tuning_strategy,
        )
        logger.debug(
            "[TimeSeriesPipeline] %s Config: %s",
            self.model_name,
            {
                "forecast_params": self.forecast_config.to_dict(),
                "tuning_params": self.tuning_params,
                "model_params": self.model_params,
                "task_params": self.task_params,
            },
        )

    @logged_operation()
    def fit(
        self,
        df: pd.DataFrame,
        schema: TimeSeriesSchema,
        *,
        future_df: pd.DataFrame | None = None,
    ) -> TimeSeriesPipeline:
        """Validate the history, load the model and, for fine-tuning, train it.

        For zero-shot inference no training happens: ``fit`` validates and
        stores the history and loads the weights. With ``"finetune"`` or
        ``"peft"`` the model is then trained on windows sampled from the
        history; :attr:`training_report_` records the run.

        Args:
            df: Long-format history, one row per (item, timestamp).
            schema: Names the target, timestamp, item and covariate columns.
            future_df: Values of the schema's ``known_covariates`` over the
                horizon that follows this history. Only needed when the schema
                names any; :meth:`predict` can supply a different one.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``df`` does not satisfy ``schema``.
            ConfigError: If the schema uses a feature the model does not
                support, such as covariates or several targets.
            UnsupportedStrategyError: If the model's adapter cannot be trained.
            EnvelopeError: If the horizon exceeds the model's documented limit
                and ``envelope_mode='error'``.
            ImportError: If the model's adapter or its backend is not installed.
        """
        if not isinstance(schema, TimeSeriesSchema):
            raise TypeError(f"schema must be a TimeSeriesSchema, got {type(schema).__name__}")


        check_schema_support(self.spec, schema)
        horizon = self.forecast_config.prediction_length
        panel = schema.to_panel(
            df,
            native_missing=self._accepts_missing(),
            future_df=future_df,
            horizon=horizon,
        )
        log_event(logger, "data_validated", "History validated",
                  series=len(panel.item_groups()), target_series=len(panel), rows=len(df), frequency=panel.freq,
                  context_length=self._context_length(), horizon=horizon,
                  targets=len(schema.target_names))
        check_schema_support(self.spec, schema, panel=panel)
        if self.task_type != "forecasting" and panel.covariate_names:
            warn_once(
                f"covariate column(s) {list(panel.covariate_names)} are not used by "
                f"task_type={self.task_type!r}: it is built on forecasts of the target "
                f"history alone. They are ignored. Use task_type='forecasting' to "
                f"condition on them.",
                UserWarning,
                key=f"ts-covariates-unused:{self.task_type}:{','.join(panel.covariate_names)}",
            )
        if self.task_type == "forecasting":
            check_forecast_envelope(self.spec, prediction_length=horizon, mode=self.envelope_mode)

        adapter_cls = self._resolve_adapter_class()
        if self.tuning_strategy != "inference" and not _trains(adapter_cls):
            raise UnsupportedStrategyError(
                self.model_name, self.tuning_strategy, self.task_type, ("inference",)
            )
        adapter_params = {
            k: v for k, v in self.model_params.items() if k not in _PIPELINE_MODEL_KEYS
        }
        self.adapter_ = adapter_cls(
            self.spec,
            checkpoint=self.checkpoint,
            device=resolve_device(self._requested_device()),
            model_params=adapter_params,
            seed=self.tuning_config.seed,
        )
        logger.info(
            "[TimeSeriesPipeline] Loading %s on %s", self.model_name, self.adapter_.device
        )
        with stage(logger, "load", device=str(self.adapter_.device)):
            self.adapter_.load()
        if self.tuning_config.seed is not None and not self.adapter_.respects_seed:
            warn_once(
                f"{self.model_name} does not use tuning_params['seed']: its forecasts "
                f"either are deterministic or draw on randomness it does not expose. "
                f"The seed still seeds fine-tuning where the model supports it.",
                UserWarning,
                key=f"ts-seed-ignored:{self.model_name}",
            )

        self.schema_ = schema
        self.history_ = panel
        self.calibrator_ = None
        self.training_report_ = None
        self.training_occurred_ = False
        if self.tuning_strategy == "inference":
            logger.info(
                "[TimeSeriesPipeline] Fitting %s in inference mode (zero-shot) on %d series",
                self.model_name,
                len(panel),
            )
        else:
            logger.info(
                "[TimeSeriesPipeline] Fine-tuning %s (%s) on %d series (-> %s)",
                self.model_name,
                self.tuning_strategy,
                len(panel),
                type(self.adapter_).__name__,
            )
            self.training_report_ = self.adapter_.finetune(
                _interpolated(panel) if not self.spec.native_missing else panel,
                self._training_spec(),
            )
            self.training_occurred_ = True
            self._log_training_report()
        self._fit_digest = _digest(panel.fingerprint(), _tensor_digest(self.adapter_.tuned_state))
        self._calibration_digest = None
        self._is_fitted = True
        logger.info(
            "[TimeSeriesPipeline] Fit process complete: %d series, frequency '%s', %s",
            len(panel),
            panel.freq,
            "trained" if self.training_occurred_ else "no training (zero-shot)",
        )
        return self

    @logged_operation()
    def predict(
        self, df: pd.DataFrame | None = None, *, future_df: pd.DataFrame | None = None
    ) -> ForecastResult | AnomalyResult | ImputationResult | EmbeddingResult:
        """Run the pipeline's task on the fitted history or on ``df``.

        Forecasting returns ``prediction_length`` steps past the end of each
        series; anomaly detection scores every observation after a warm-up;
        imputation returns the history with its gaps filled; embedding returns
        one vector per series.

        Args:
            df: History in the fitted schema. ``None`` uses the history given
                to :meth:`fit`. Passing a frame runs the task on new origins
                or new items without refitting.
            future_df: Values of the schema's ``known_covariates`` over the
                horizon (forecasting only). ``None`` reuses the ones given to
                :meth:`fit`, which only match when the forecast origins have
                not moved.

        Returns:
            An immutable result, safe to share when cached.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If a series has no observed value within the model's
                context window.
            ConfigError: If the schema names known covariates and no values for
                the horizon are available.
        """
        self._check_is_fitted("predict")
        panel = self._panel_for(df, future_df)
        if self.task_type == "forecasting":
            self._check_future_covariates(panel)
            compute, method = (lambda: self._forecast_uncached(panel)), "forecast"
        elif self.task_type == "anomaly_detection":
            compute, method = (lambda: self._detect_anomalies(panel)), "anomalies"
        elif self.task_type == "imputation":
            compute, method = (lambda: self._impute(panel)), "impute"
        else:
            compute, method = (lambda: self._embed(panel)), "embed"
        return self.cache.get_or_compute(
            self._cache_scope(), panel, method, compute, data_fingerprint=panel.fingerprint()
        )

    @logged_operation()
    def calibrate(
        self,
        df: pd.DataFrame | None = None,
        *,
        method: str | None = None,
        windows: int = 5,
        step: int | None = None,
        output_format: str = "rich",
    ) -> TimeSeriesPipeline:
        """Calibrate the forecast quantiles with split conformal prediction.

        Forecasts are made from ``windows`` rolling origins at the end of the
        history (or of ``df``), compared with what followed, and the
        per-horizon-step errors set how much each quantile moves. Later
        :meth:`predict` calls return the calibrated quantiles; the point
        forecast is unchanged. Fitting again removes the calibration.

        Use a history the model was not fine-tuned on, or the calibration
        will be optimistic.

        Args:
            df: History to calibrate on, in the fitted schema. ``None`` uses
                the fitted history.
            method: ``"cqr"`` (default for models with quantiles),
                ``"absolute"`` (default for point-only models) or ``"signed"``.
                See :mod:`tabtune.TimeSeries.calibration`.
            windows: Rolling origins per series.
            step: Steps between origins (default ``prediction_length``).
            output_format: ``'rich'`` logs a calibration report, ``'json'``
                prints its summary as JSON.

        Returns:
            ``self``.

        Raises:
            ConfigError: If the task is not forecasting or no quantile levels
                were requested.
            ValueError: If there are too few calibration rows for the
                requested levels.
        """
        self._check_is_fitted("calibrate")
        cfg = self.forecast_config
        if self.task_type != "forecasting":
            raise ConfigError("calibrate() applies to task_type='forecasting' only.")
        if not cfg.quantile_levels:
            raise ConfigError("calibrate() needs forecast_params['quantile_levels'].")
        _check_choice("output_format", output_format, _OUTPUT_FORMATS, error=ValueError)
        panel = self.history_ if df is None else self.schema_.to_panel(
            df, native_missing=self._accepts_missing()
        )
        ys, points, quantiles = [], [], []
        for window in track(self._windows(panel, windows, step), logger=logger,
                            description="Backtest windows scored"):
            output = self._run_adapter(window.context, cfg)
            ys.append(window.actual)
            points.append(output.point)
            quantiles.append(output.quantiles)
        has_quantiles = all(q is not None for q in quantiles)
        method = method or ("cqr" if has_quantiles and len(cfg.quantile_levels) >= 2 else "absolute")
        y, point = np.concatenate(ys), np.concatenate(points)
        q = np.concatenate(quantiles) if has_quantiles else None
        calibrator = ConformalCalibrator(method).fit(y, point, q, cfg.quantile_levels)
        check = calibrator.calibrate(point, q, cfg.quantile_levels)
        bad = [lv for j, lv in enumerate(cfg.quantile_levels) if not np.isfinite(check[..., j]).all()]
        if bad:
            raise ValueError(
                f"{calibrator.n_calibration_} calibration rows are too few for quantile level(s) "
                f"{bad}: a level q needs at least 1 / min(q, 1 - q) / 2 rows. Use more windows or "
                f"series, or less extreme levels."
            )
        self.calibrator_ = calibrator
        self._calibration_digest = _digest(pickle.dumps(calibrator, protocol=4))
        summary = {
            "method": method,
            "n_calibration_rows": int(calibrator.n_calibration_),
            "windows": windows,
            "quantile_levels": list(cfg.quantile_levels),
        }
        if output_format == "rich":
            log_metrics(logger, summary, title="Conformal calibration")
            logger.info("Later predictions use calibrated quantiles; point forecasts are unchanged.")
            logger.info("Calibration on fine-tuning data can give optimistic intervals.")
        else:
            print(json.dumps(summary, indent=4))
        return self

    @logged_operation()
    def backtest(
        self,
        df: pd.DataFrame | None = None,
        *,
        windows: int = 3,
        step: int | None = None,
        output_format: str | None = "rich",
    ) -> pd.DataFrame:
        """Score rolling-origin forecasts over the end of the history.

        The model is not refitted per window: zero-shot models do not need
        it, and a fine-tuned model is scored on windows it may have seen.

        Args:
            df: History to backtest on, in the fitted schema. ``None`` uses the
                fitted history.
            windows: Rolling origins per series; window 0 ends at the last
                observation.
            step: Steps between origins (default ``prediction_length``).
            output_format: ``'rich'`` (default) logs a backtest report,
                ``'json'`` prints the table as JSON and ``None`` is silent.

        Returns:
            One row per window with its cutoff, the number of series and the
            metrics of :meth:`evaluate`.
        """
        self._check_is_fitted("backtest")
        if self.task_type != "forecasting":
            raise ConfigError("backtest() applies to task_type='forecasting' only.")
        if output_format is not None:
            _check_choice("output_format", output_format, _OUTPUT_FORMATS, error=ValueError)
        panel = self.history_ if df is None else self.schema_.to_panel(
            df, native_missing=self._accepts_missing()
        )
        rows = self._backtest_rows(panel, windows, step)
        levels = list(self.forecast_config.quantile_levels) if rows.attrs["quantiles"] else []
        table = []
        for index, group in rows.groupby("window", sort=True):
            metrics = score_forecast(group, levels, rows.attrs["scales"], series_key=["__row__"])
            table.append(
                {
                    "window": int(index),
                    "cutoff": group["__cutoff__"].max(),
                    "n_series": int(group["__row__"].nunique()),
                    **metrics,
                }
            )
        frame = pd.DataFrame(table)
        if output_format == "json":
            print(frame.to_json(orient="records", date_format="iso", indent=4))
        elif output_format == "rich":
            log_table(logger, "Backtest windows", frame.columns, frame.itertuples(index=False, name=None))
            log_metrics(logger, frame.drop(columns=["window", "cutoff", "n_series"]).mean().to_dict(),
                        title="Mean across windows")
        return frame

    @logged_operation()
    def tune_context_length(
        self,
        candidates: list[int] | None = None,
        *,
        windows: int = 2,
        metric: str | None = None,
    ) -> pd.DataFrame:
        """Choose ``context_length`` by backtest and keep the best.

        Args:
            candidates: Context lengths to try. Defaults to powers of two from
                64 up to the model's maximum and the longest history.
            windows: Rolling origins per series for each candidate.
            metric: Metric to minimise; ``"wql"`` when quantiles are
                forecast, ``"mase"`` otherwise.

        Returns:
            One row per candidate with its metrics, best first.
        """
        self._check_is_fitted("tune_context_length")
        if self.task_type != "forecasting":
            raise ConfigError("tune_context_length() applies to task_type='forecasting' only.")
        longest = self.history_.max_length
        limit = min(self.spec.max_context or longest, longest)
        if candidates is None:
            candidates = [2**k for k in range(6, 16) if 2**k < limit] + [limit]
        original = self.forecast_config
        results = []
        try:
            for length in sorted(set(candidates)):
                self.forecast_config = original.merged(context_length=int(length))
                rows = self._backtest_rows(self.history_, windows, None)
                levels = list(original.quantile_levels) if rows.attrs["quantiles"] else []
                metrics = score_forecast(rows, levels, rows.attrs["scales"], series_key=["__row__"])
                results.append({"context_length": int(length), **metrics})
        finally:
            self.forecast_config = original
        table = pd.DataFrame(results)
        metric = metric or ("wql" if "wql" in table else "mase" if "mase" in table else "mae")
        if metric not in table:
            raise ValueError(f"metric {metric!r} is not available; choose from {list(table.columns[1:])}")
        table = table.sort_values(metric, ascending=LOWER_IS_BETTER.get(metric, True), kind="mergesort")
        best = int(table.iloc[0]["context_length"])
        self.forecast_config = original.merged(context_length=best)
        log_table(logger, f"Context length tuning / ranked by {_metric_label(metric)}",
                  table.columns, table.itertuples(index=False, name=None))
        log_event(logger, "context_selected", "Context length selected", context_length=best, metric=metric)
        return table.reset_index(drop=True)

    @logged_operation()
    def evaluate(
        self,
        df_actual: pd.DataFrame,
        *,
        forecast: ForecastResult | None = None,
        history: pd.DataFrame | None = None,
        output_format: str = "rich",
    ) -> dict[str, Any]:
        """Score forecasts against observed values.

        Compares the forecast with the rows of ``df_actual`` whose (item,
        timestamp) keys fall in the forecast window. By default the forecast is
        produced here from the fitted history; pass ``forecast`` to score one
        you already have. For sampling models without a seed, a new forecast
        differs from an earlier one.

        Args:
            df_actual: Observed values over the horizon, in the fitted schema.
            forecast: A forecast to score, e.g. the result of :meth:`predict`.
            history: History to forecast from instead of the fitted one.
                Cannot be combined with ``forecast``.
            output_format: ``'rich'`` logs the metrics, ``'json'`` prints them
                as JSON, as in ``TabularPipeline.evaluate``.

        Returns:
            ``mae``, ``rmse``, ``mse``, ``smape`` and ``mase`` over every
            forecast point; with quantiles also ``mean_pinball_loss``,
            ``wql`` and the coverage of the widest central interval. MASE is
            scaled by each series' seasonal naive error on its history. A
            multivariate forecast also reports ``per_target``, the same
            metrics for each target on its own scale, since averaging across
            targets that are measured in different units says little.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If both ``forecast`` and ``history`` are given, if
                timestamps differ in timezone, or if no row of ``df_actual``
                matches a forecast timestamp.
        """
        self._check_is_fitted("evaluate")
        if self.task_type != "forecasting":
            raise ConfigError("evaluate() applies to task_type='forecasting' only.")
        _check_choice("output_format", output_format, _OUTPUT_FORMATS, error=ValueError)
        if forecast is not None and history is not None:
            raise ValueError("Pass either forecast or history, not both.")
        schema = self.schema_
        history_panel = self.history_
        if forecast is None:
            if history is not None:
                history_panel = schema.to_panel(history, native_missing=self._accepts_missing())
            forecast = self.predict(history)
        predicted = forecast.to_pandas()

        keys = [c for c in (schema.item_id, schema.timestamp) if c is not None]
        targets = list(schema.target_names)
        missing = [c for c in (*keys, *targets) if c not in df_actual.columns]
        if missing:
            raise ValueError(f"df_actual is missing column(s) {missing}")

        actual = df_actual.loc[:, [*keys, *targets]].melt(
            id_vars=keys, value_vars=targets, var_name="target", value_name="__actual__"
        )
        actual[schema.timestamp] = pd.to_datetime(actual[schema.timestamp])
        _check_same_timezone(predicted[schema.timestamp], actual[schema.timestamp])

        merged = predicted.merge(actual, on=[*keys, "target"], how="inner")
        if merged.empty:
            raise ValueError(
                "No rows of df_actual fall inside the forecast window; pass the "
                "observations for the timestamps that follow the fitted history."
            )
        if len(merged) < len(predicted):
            warn_once(
                f"df_actual covers {len(merged)} of {len(predicted)} forecast points; "
                f"metrics use the overlap only.",
                UserWarning,
                key="ts-evaluate-partial-horizon",
            )

        series_key = [*([schema.item_id] if schema.item_id else []), "target"]
        scales = self._scales(history_panel, keyed_by_item=schema.item_id is not None)
        levels = list(forecast.quantile_levels)
        results: dict[str, Any] = score_forecast(merged, levels, scales, series_key=series_key)
        if len(targets) > 1:
            results["per_target"] = {
                name: score_forecast(rows, levels, scales, series_key=series_key)
                for name, rows in merged.groupby("target", sort=False)
            }
        if output_format == "json":
            print(json.dumps(results, indent=4))
        else:
            _log_metrics(results)
            for name, metrics in results.get("per_target", {}).items():
                log_metrics(logger, metrics, title=f"Target: {name}")
        return results

    @classmethod
    def select(
        cls,
        df: pd.DataFrame,
        schema: TimeSeriesSchema,
        forecast_params: dict,
        *,
        candidates: list[str] | None = None,
        windows: int = 1,
        rank_by: str | None = None,
        license_mode: str = "research",
        include_unverified_licenses: bool = False,
        future_df: pd.DataFrame | None = None,
        **pipeline_kwargs: Any,
    ) -> TimeSeriesPipeline:
        """Backtest candidate models on ``df`` and return the best one, fitted.

        Candidates default to every registered zero-shot forecaster that can
        read the schema and whose license suits ``license_mode``. The ranking
        is kept on the returned pipeline as ``selection_``.

        Args:
            df: History, in long format.
            schema: Column roles.
            forecast_params: As for the constructor.
            candidates: Model names to compare.
            windows: Backtest windows per series.
            rank_by: Metric to rank by; ``"wql"`` with quantiles, else ``"mase"``.
            license_mode: ``"commercial"`` keeps only models cleared for
                commercial use.
            include_unverified_licenses: With ``license_mode="commercial"``, also
                consider models whose weight license TabTune has not verified.
                Without it they are dropped from the candidate list.
            future_df: Known covariates over the horizon after ``df``.
            **pipeline_kwargs: Forwarded to each pipeline.
        """
        from .leaderboard import TimeSeriesLeaderboard

        check_forecasting_only(pipeline_kwargs, context="TimeSeriesPipeline.select")
        board = TimeSeriesLeaderboard(df, schema, forecast_params=forecast_params, windows=windows)
        if candidates is None:
            board.add_all(
                commercial_ok=True if license_mode == "commercial" else None,
                include_unverified_licenses=include_unverified_licenses,
                license_mode=license_mode,
                **pipeline_kwargs,
            )
        else:
            board.add_models(candidates, license_mode=license_mode, **pipeline_kwargs)
        ranking = board.run(rank_by=rank_by, display=False)
        best = board.best(rank_by=rank_by)
        if best is None:
            raise RuntimeError(f"No candidate could be backtested:\n{ranking.to_string()}")
        kwargs = {**pipeline_kwargs, **best.pipeline_kwargs, "license_mode": license_mode}
        pipe = cls(best.model_name, forecast_params=forecast_params, **kwargs)
        pipe.fit(df, schema, future_df=future_df)
        pipe.selection_ = ranking
        return pipe

    def save(self, file_path: str) -> None:
        """Save the fitted pipeline.

        Pretrained weights are not stored. They reload from ``checkpoint`` on
        first use after :meth:`load`, which needs network access or a
        populated Hugging Face cache; for offline use, fit with a local
        checkpoint directory. Fine-tuned weights (the LoRA adapters for
        ``"peft"``) are stored and re-applied.
        """
        self._check_is_fitted("save")
        logger.info("[TimeSeriesPipeline] Saving pipeline to %s", file_path)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> TimeSeriesPipeline:
        """Load a pipeline written by :meth:`save`."""
        logger.info("[TimeSeriesPipeline] Loading pipeline from %s", file_path)
        pipeline = joblib.load(file_path)
        if not isinstance(pipeline, cls):
            raise TypeError(f"{file_path} does not contain a {cls.__name__}")
        return pipeline

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor arguments, so the pipeline can be rebuilt or cloned."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "tuning_strategy": self.tuning_strategy,
            "tuning_params": dict(self.tuning_params),
            "model_params": dict(self.model_params),
            "forecast_params": self.forecast_config.to_dict(),
            "task_params": dict(self.task_params),
            "envelope_mode": self.envelope_mode,
            "license_mode": self.license_mode,
            "validate": self.validate,
        }

    def clear_cache(self) -> int:
        """Drop this pipeline's cached results. Returns the number removed."""
        return self.cache.invalidate(self._cache_scope())

    def _panel_for(
        self, df: pd.DataFrame | None, future_df: pd.DataFrame | None
    ) -> TimeSeriesPanel:
        horizon = self.forecast_config.prediction_length
        if df is None:
            panel = self.history_
            if future_df is not None:
                panel = self.schema_.attach_future(panel, future_df, horizon=horizon)
            return panel
        return self.schema_.to_panel(
            df,
            native_missing=self._accepts_missing(),
            future_df=future_df,
            horizon=horizon,
        )

    def _accepts_missing(self) -> bool:
        """Anomaly detection and imputation read histories with gaps whatever the model."""
        return self.spec.native_missing or self.task_type in ("anomaly_detection", "imputation")

    def _check_future_covariates(self, panel: TimeSeriesPanel) -> None:
        """Refuse to forecast with known covariates whose future values are absent."""
        known = self.schema_.known_covariates
        if not known:
            return
        if any(not row for row in panel.future_covariates):
            raise ConfigError(
                f"The schema names known_covariates {list(known)}, so their values over "
                f"the next {self.forecast_config.prediction_length} step(s) are needed. "
                f"Pass them as predict(future_df=...) (or fit(..., future_df=...))."
            )

    def _cache_scope(self) -> str:
        """Identify this fitted model *and* request for cache lookups.

        The cache key covers only the history and the method name, so
        everything else that changes a result has to be in the scope: the
        settings, the fitted state (history and tuned weights) and the
        calibration. It is built from content only, so a disk cache is shared
        across processes and by reloaded pipelines.
        """
        from .. import __version__

        settings = json.dumps(
            {
                "forecast": self.forecast_config.model_dump(),
                "model_params": self.model_params,
                "tuning_params": self.tuning_params,
                "task_params": self.task_params,
                "targets": list(self.schema_.target_names) if self.schema_ else [],
                "fit": getattr(self, "_fit_digest", None),
                "calibration": getattr(self, "_calibration_digest", None),
                "version": __version__,
            },
            sort_keys=True,
            default=repr,
        )
        return (
            f"{self.model_name}|{self.task_type}|{self.tuning_strategy}|{self.checkpoint}|"
            f"{_digest(settings)}"
        )

    def _log_training_report(self) -> None:
        """Log the fine-tuning summary held in :attr:`training_report_`."""
        report = self.training_report_ or {}
        log_metrics(logger, {k: v for k, v in report.items() if k != "history" and v is not None},
                    title="Fine-tuning summary")

    def _check_is_fitted(self, action: str) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"You must call fit() on the pipeline before {action}().")

    def _check_checkpoint(self) -> None:
        """Restrict the checkpoint to validated ones, allowing local directories."""
        if self.checkpoint in self.spec.checkpoints or not self.spec.checkpoints:
            return
        if os.path.isdir(self.checkpoint):
            logger.info(
                "[TimeSeriesPipeline] Using local checkpoint directory %s, which is not "
                "one of the validated %s checkpoints.",
                self.checkpoint,
                self.model_name,
            )
            return
        if self.validate:
            raise ConfigError(
                f"Checkpoint {self.checkpoint!r} is neither a validated checkpoint for "
                f"{self.model_name} nor a local directory. Validated: "
                f"{', '.join(self.spec.checkpoints)}. Pass validate=False to use it anyway."
            )

    def _requested_device(self) -> str | None:
        """Same precedence as ``TabularPipeline``: tuning_params, then model_params."""
        return self.tuning_params.get("device", self.model_params.get("device"))

    def _context_length(self) -> int | None:
        """The context the model sees: the requested length, capped at the model's."""
        requested = self.forecast_config.context_length
        limit = self.spec.max_context
        if requested is None:
            return limit
        return requested if limit is None else min(requested, limit)

    def _training_spec(self) -> TrainingSpec:
        cfg = self.tuning_config
        steps_per_epoch = cfg.steps_per_epoch or _DEFAULT_STEPS_PER_EPOCH
        peft = cfg.peft_config
        clip = cfg.gradient_clip_norm if "gradient_clip_norm" in self.tuning_params else 1.0
        return TrainingSpec(
            mode="lora" if self.tuning_strategy == "peft" else "full",
            prediction_length=self.forecast_config.prediction_length,
            context_length=(
                self._context_length() if self.forecast_config.context_length is not None else None
            ),
            steps=cfg.epochs * steps_per_epoch,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            gradient_clip_norm=clip,
            validation=cfg.early_stopping or cfg.validation_split > 0,
            patience=cfg.early_stopping_patience if cfg.early_stopping else None,
            validation_every=steps_per_epoch,
            lora_r=peft.r if peft else 8,
            lora_alpha=peft.lora_alpha if peft else 16,
            lora_dropout=peft.lora_dropout if peft else 0.05,
            lora_targets=tuple(peft.target_modules) if peft and peft.target_modules else None,
            seed=cfg.seed,
        )

    def _resolve_adapter_class(self) -> type[TSFMAdapter]:
        """Import the adapter named by the spec, with actionable errors."""
        target = self.spec.adapter
        if isinstance(target, str):
            module_path, _, attribute = target.partition(":")
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as exc:
                missing = exc.name or ""
                if missing and (module_path == missing or module_path.startswith(missing + ".")):
                    raise ImportError(
                        f"The adapter for {self.model_name} ({target}) is not available "
                        f"in this TabTune installation."
                    ) from exc
                raise missing_extra_error(self.spec, missing) from exc
            try:
                adapter_cls = getattr(module, attribute)
            except AttributeError as exc:
                raise ImportError(f"{module_path} has no adapter named {attribute!r}") from exc
        else:
            adapter_cls = target

        if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, TSFMAdapter)):
            raise TypeError(f"{adapter_cls!r} is not a TSFMAdapter subclass")
        return adapter_cls

    def _run_adapter(
        self,
        panel: TimeSeriesPanel,
        config: ForecastConfig,
        *,
        fill_missing: bool = False,
        verbose: bool = False,
    ) -> AdapterOutput:
        """Cut the panel to the model's context and forecast it."""
        if not self.adapter_.is_loaded:
            self.adapter_.load()
        context = self._context_length()
        model_panel = panel
        if context is not None and panel.max_length > context:
            logger.log(
                logging.INFO if verbose else logging.DEBUG,
                "[TimeSeriesPipeline] Using the last %d observations of each series "
                "(longest history: %d)",
                context,
                panel.max_length,
            )
            model_panel = panel.tail(context)
            model_panel.check_observed(context=context)
        if fill_missing and not self.spec.native_missing:
            model_panel = _interpolated(model_panel)
        output = self.adapter_.forecast(model_panel, config)
        self._check_adapter_output(output, panel, config)
        return output

    def _forecast_uncached(self, panel: TimeSeriesPanel) -> ForecastResult:
        cfg = self.forecast_config
        output = self._run_adapter(panel, cfg, verbose=True)
        return self._result(panel, output, cfg)

    def _provenance(self) -> dict[str, Any]:
        """Where a result came from: the keys every task's metadata carries.

        Kept in one place because ``fit`` trains for any task, so an anomaly,
        imputation or embedding result can come from a fine-tuned model and has
        to record that just as a forecast does.
        """
        return {
            "model": self.model_name,
            "checkpoint": self.checkpoint,
            "device": self.adapter_.device if self.adapter_ is not None else None,
            "tuning_strategy": self.tuning_strategy,
            "training_occurred": self.training_occurred_,
        }

    def _result(
        self, panel: TimeSeriesPanel, output: AdapterOutput, cfg: ForecastConfig
    ) -> ForecastResult:
        point, quantiles = _arrays(output)
        calibration = None
        if self.calibrator_ is not None and cfg.quantile_levels:
            quantiles = self.calibrator_.calibrate(point, quantiles, cfg.quantile_levels)
            calibration = self.calibrator_.summary()
        timestamps = np.stack(
            [
                future_timestamps(last, panel.freq, cfg.prediction_length).to_numpy()
                for last in panel.last_timestamps
            ]
        )
        schema = self.schema_
        return ForecastResult(
            item_ids=panel.item_ids,
            timestamps=timestamps,
            point=point,
            quantiles=quantiles,
            quantile_levels=tuple(cfg.quantile_levels) if quantiles is not None else (),
            targets=panel.target_names,
            item_id_column=schema.item_id,
            timestamp_column=schema.timestamp,
            metadata={
                **self._provenance(),
                "point_forecast": self.adapter_.point_forecast,
                "calibration": calibration,
            },
        )

    def _forecaster(self, panel: TimeSeriesPanel, horizon: int, levels: Any):
        """Forecast callback for the anomaly and imputation tasks."""
        config = ForecastConfig(
            prediction_length=horizon,
            quantile_levels=list(levels),
            context_length=self.forecast_config.context_length,
        )
        return _arrays(self._run_adapter(panel, config, fill_missing=True))

    def _detect_anomalies(self, panel: TimeSeriesPanel) -> AnomalyResult:
        params = dict(self.task_params)
        horizon = self.forecast_config.prediction_length
        season = season_length(panel.freq)
        params.setdefault("min_context", max(32, 2 * season) if season > 1 else 32)
        if not self.adapter_.is_loaded:
            self.adapter_.load()
        served = self.adapter_.quantile_range()
        if "coverage" not in params and served is not None:
            params["coverage"] = min(0.98, round(1 - 2 * max(served[0], 1 - served[1]), 10))
        context = self._context_length()
        per_row, info = score_anomalies(
            panel,
            self._forecaster,
            horizon=horizon,
            context=min(context, 512) if context else 512,
            **params,
        )
        frames = []
        for row, out in enumerate(per_row):
            frames.append(
                pd.DataFrame(
                    {
                        **self._key_columns(panel, row),
                        "value": panel.values[row],
                        "expected": out["expected"],
                        "lower": out["lower"],
                        "upper": out["upper"],
                        "score": out["score"],
                        "p_value": out["p_value"],
                        "is_anomaly": out["is_anomaly"],
                    }
                )
            )
        info.update(self._provenance())
        return AnomalyResult(pd.concat(frames, ignore_index=True), info["method"], info)

    def _impute(self, panel: TimeSeriesPanel) -> ImputationResult:
        filled, bands = fill_gaps(
            panel,
            self._forecaster,
            method=self.task_params.get("method", "bidirectional"),
            context=self._context_length() or 512,
            min_context=self.task_params.get("min_context", 8),
            native_missing=self.spec.native_missing,
        )
        schema = self.schema_
        single = len(schema.target_names) == 1
        frames = []
        for rows in panel.item_groups():
            first = rows[0]
            keys = self._key_columns(panel, first)
            keys.pop("target")
            data: dict[str, Any] = dict(keys)
            for row in rows:
                name = panel.target_names[row]
                data[name] = filled[row]
                flag = "imputed" if single else f"imputed_{name}"
                data[flag] = np.isnan(panel.values[row])
                if single:
                    data["lower"], data["upper"] = bands[row]
            frames.append(pd.DataFrame(data))
        return ImputationResult(
            pd.concat(frames, ignore_index=True),
            {
                **self._provenance(),
                "method": self.task_params.get("method", "bidirectional"),
                "min_context": self.task_params.get("min_context", 8),
            },
        )

    def _embed(self, panel: TimeSeriesPanel) -> EmbeddingResult:
        if not self.adapter_.is_loaded:
            self.adapter_.load()
        context = self._context_length()
        model_panel = panel.tail(context) if context and panel.max_length > context else panel
        vectors = np.asarray(self.adapter_.embed(model_panel), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(panel):
            raise ValueError(
                f"{type(self.adapter_).__name__}.embed() returned shape {vectors.shape}, "
                f"expected ({len(panel)}, dim)"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"{type(self.adapter_).__name__}.embed() returned non-finite values")
        return EmbeddingResult(
            item_ids=panel.item_ids,
            targets=panel.target_names,
            embeddings=vectors,
            item_id_column=self.schema_.item_id,
            metadata={**self._provenance(), "dim": int(vectors.shape[1])},
        )

    def _key_columns(self, panel: TimeSeriesPanel, row: int) -> dict[str, Any]:
        schema = self.schema_
        n = len(panel.values[row])
        keys: dict[str, Any] = {}
        if schema.item_id is not None:
            keys[schema.item_id] = pd.Index([panel.item_ids[row]] * n)
        keys[schema.timestamp] = rolled_timestamps(panel, row)
        keys["target"] = panel.target_names[row]
        return keys

    def _windows(self, panel: TimeSeriesPanel, windows: int, step: int | None):
        horizon = self.forecast_config.prediction_length
        return rolling_windows(
            panel,
            horizon=horizon,
            windows=windows,
            step=step,
            min_context=horizon,
            known_covariates=self.schema_.known_covariates,
        )

    def _backtest_rows(
        self, panel: TimeSeriesPanel, windows: int, step: int | None
    ) -> pd.DataFrame:
        """Long frame of backtest forecasts and actuals, one row per point."""
        cfg = self.forecast_config
        frames = []
        scales: dict[Any, float] = {}
        has_quantiles = True
        season = season_length(panel.freq)
        for window in track(self._windows(panel, windows, step), logger=logger,
                            description="Backtest windows scored"):
            result = self._result(window.context, self._run_adapter(window.context, cfg), cfg)
            has_quantiles &= result.quantiles is not None
            horizon = cfg.prediction_length
            for row in range(len(window.context)):
                key = (window.index, row)
                scales[key] = seasonal_scale(window.context.values[row], season)
                data = {
                    "window": window.index,
                    "__row__": [key] * horizon,
                    "__cutoff__": window.context.last_timestamps[row],
                    "__actual__": window.actual[row],
                    "point": result.point[row],
                }
                if result.quantiles is not None:
                    for j, level in enumerate(result.quantile_levels):
                        data[str(level)] = result.quantiles[row, :, j]
                frames.append(pd.DataFrame(data))
        rows = pd.concat(frames, ignore_index=True)
        rows = rows[np.isfinite(rows["__actual__"].to_numpy(dtype=float))]
        rows.attrs["scales"] = scales
        rows.attrs["quantiles"] = has_quantiles
        return rows

    def _scales(self, panel: TimeSeriesPanel, *, keyed_by_item: bool) -> dict[Any, float]:
        season = season_length(panel.freq)
        return {
            ((item, target) if keyed_by_item else target): seasonal_scale(values, season)
            for item, target, values in zip(
                panel.item_ids, panel.target_names, panel.values, strict=True
            )
        }

    def _check_adapter_output(
        self, output: AdapterOutput, panel: TimeSeriesPanel, config: ForecastConfig | None = None
    ) -> None:
        """Enforce the adapter contract in one place rather than in every adapter."""
        config = config or self.forecast_config
        name = type(self.adapter_).__name__
        if not isinstance(output, AdapterOutput):
            raise TypeError(f"{name}.forecast() must return AdapterOutput, got {type(output).__name__}")
        horizon = config.prediction_length
        expected = {"point": (len(panel), horizon)}
        if output.quantiles is not None:
            expected["quantiles"] = (len(panel), horizon, len(config.quantile_levels))
        for field, shape in expected.items():
            values = getattr(output, field)
            if np.shape(values) != shape:
                raise ValueError(
                    f"{name} returned {field} of shape {np.shape(values)}, expected {shape}"
                )
            finite = np.isfinite(np.asarray(values, dtype=float))
            if not finite.all():
                rows = np.where(~finite.reshape(len(panel), -1).all(axis=1))[0]
                raise ValueError(
                    f"{name} returned non-finite {field} values for item(s) "
                    f"{[panel.item_ids[i] for i in rows[:5]]}. Check those series for "
                    f"extreme values or scale."
                )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"TimeSeriesPipeline(model={self.model_name!r}, checkpoint={self.checkpoint!r}, "
            f"task={self.task_type!r}, strategy={self.tuning_strategy!r}, "
            f"prediction_length={self.forecast_config.prediction_length}, {status})"
        )


def _resolve_spec(model_name: str) -> TimeSeriesModelSpec:
    """Look up a model, pointing unknown names at the registration hook."""
    try:
        return get_time_series_model_spec(model_name)
    except ModelNotFoundError as exc:
        exc.args = (
            exc.args[0] + "\n  To use your own model, register it with "
            "tabtune.registry.register_time_series_model().",
        )
        raise


def _trains(adapter_cls: type[TSFMAdapter]) -> bool:
    """Whether an adapter overrides the fine-tuning hooks."""
    return (
        adapter_cls.finetune is not TSFMAdapter.finetune
        or adapter_cls._network is not TSFMAdapter._network
    )


def _arrays(output: AdapterOutput) -> tuple[np.ndarray, np.ndarray | None]:
    """The adapter's arrays as float, with quantiles sorted so they never cross."""
    point = np.asarray(output.point, dtype=float)
    if output.quantiles is None:
        return point, None
    return point, np.sort(np.asarray(output.quantiles, dtype=float), axis=-1)


def _digest(*parts: Any) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(part if isinstance(part, bytes) else repr(part).encode("utf-8"))
    return hasher.hexdigest()


def _tensor_digest(state: dict[str, Any] | None) -> str | None:
    """Content hash of a tuned state (tensors and metadata)."""
    if state is None:
        return None
    hasher = hashlib.blake2b(digest_size=16)
    for name in sorted(state):
        value = state[name]
        hasher.update(name.encode("utf-8"))
        if hasattr(value, "detach"):
            hasher.update(value.detach().cpu().double().contiguous().numpy().tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))
    return hasher.hexdigest()


def _interpolated(panel: TimeSeriesPanel) -> TimeSeriesPanel:
    """Fill ``NaN`` in every row by linear interpolation (nearest value at the edges)."""
    if not any(np.isnan(v).any() for v in panel.values):
        return panel
    values = tuple(
        pd.Series(v).interpolate(limit_direction="both").to_numpy(dtype=float) for v in panel.values
    )
    return replace(panel, values=values)


def _check_choice(
    name: str, value: str, choices: tuple[str, ...], *, error: type[Exception] = ConfigError
) -> None:
    if value not in choices:
        raise error(f"{name} must be one of {list(choices)}, got {value!r}")


def _check_same_timezone(predicted: pd.Series, actual: pd.Series) -> None:
    """Raise a clear error instead of pandas' merge error on mismatched timezones."""
    tz_predicted = getattr(predicted.dt, "tz", None)
    tz_actual = getattr(actual.dt, "tz", None)
    if str(tz_predicted) != str(tz_actual):
        raise ValueError(
            f"Forecast timestamps are in timezone {tz_predicted} but df_actual's are in "
            f"{tz_actual}. Localise or convert df_actual's timestamps to match the "
            f"fitted history."
        )
