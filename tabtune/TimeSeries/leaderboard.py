"""Compare time series models on one dataset, the way ``TabularLeaderboard`` compares tabular ones.

Two evaluation modes:

* **holdout**: pass ``df_actual``. Every configuration is fitted on ``df``
  and scored on the observations that follow it.
* **backtest** (default): every configuration is fitted on ``df`` without its
  last ``windows`` forecast windows and scored on rolling-origin forecasts
  over them.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logger import log_table, logged_operation, track
from .metrics import LOWER_IS_BETTER, score_forecast
from .schema import TimeSeriesSchema
from .tasks import check_forecasting_only

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesLeaderboard", "TimeSeriesLeaderboardEntry"]


@dataclass
class TimeSeriesLeaderboardEntry:
    """One configuration and how it performed.

    Attributes:
        model_name: Canonical model name.
        tuning_strategy: ``"inference"``, ``"finetune"`` or ``"peft"``.
        label: Display name.
        pipeline_kwargs: Constructor arguments that rebuild the configuration.
        metrics: Metric name to value; empty when the entry failed.
        fit_seconds: Wall-clock time of ``fit``.
        predict_seconds: Wall-clock time of the scored forecasts.
        license_name: Weight license, from the registry.
        commercial_use_ok: Tri-state commercial-use flag.
        error: Exception text when the entry failed.
        traceback: Full traceback of a failure.
    """

    model_name: str
    tuning_strategy: str
    label: str
    pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = float("nan")
    predict_seconds: float = float("nan")
    license_name: str = ""
    commercial_use_ok: bool | None = None
    error: str | None = None
    traceback: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "Model": self.label,
            "Strategy": self.tuning_strategy,
            "Status": "ok" if self.ok else "failed",
            **self.metrics,
            "fit_s": self.fit_seconds,
            "predict_s": self.predict_seconds,
            "License": self.license_name or "-",
            "Commercial": {True: "yes", False: "no"}.get(self.commercial_use_ok, "unverified"),
            "Error": self.error,
        }
        return row


class TimeSeriesLeaderboard:
    """Fit and score several time series configurations on one dataset.

    Args:
        df: Long-format history.
        schema: Column roles.
        forecast_params: Shared by every configuration; ``prediction_length``
            is required.
        df_actual: Observations after ``df`` (holdout mode). ``None`` backtests.
        future_df: Known covariates over the horizon after ``df`` (holdout mode).
        windows: Backtest windows per series.
        step: Steps between backtest origins (default ``prediction_length``).
        cache: Result cache shared by the pipelines.

    Example:
        >>> board = TimeSeriesLeaderboard(df, schema, forecast_params={"prediction_length": 24})  # doctest: +SKIP
        >>> board.add_models(["SeasonalNaive", "ChronosBolt", "TiRex"]).run()                   # doctest: +SKIP
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: TimeSeriesSchema,
        *,
        forecast_params: dict,
        df_actual: pd.DataFrame | None = None,
        future_df: pd.DataFrame | None = None,
        windows: int = 1,
        step: int | None = None,
        cache: str | None = None,
    ) -> None:
        if not isinstance(schema, TimeSeriesSchema):
            raise TypeError(f"schema must be a TimeSeriesSchema, got {type(schema).__name__}")
        if "prediction_length" not in (forecast_params or {}):
            raise ValueError("forecast_params must include prediction_length")
        if windows < 1:
            raise ValueError(f"windows must be >= 1, got {windows}")
        self.df = df
        self.schema = schema
        self.forecast_params = dict(forecast_params)
        self.df_actual = df_actual
        self.future_df = future_df
        self.windows = windows
        self.step = step
        self.cache = cache
        self.models_to_run: list[dict[str, Any]] = []
        self.entries: list[TimeSeriesLeaderboardEntry] = []
        self._rank_by: str | None = None

    @property
    def mode(self) -> str:
        """``"holdout"`` when ``df_actual`` was given, else ``"backtest"``."""
        return "holdout" if self.df_actual is not None else "backtest"

    def add_model(
        self,
        model_name: str,
        tuning_strategy: str = "inference",
        model_params: dict | None = None,
        tuning_params: dict | None = None,
        *,
        label: str | None = None,
        **pipeline_kwargs: Any,
    ) -> TimeSeriesLeaderboard:
        """Queue a configuration.

        Args:
            model_name: Model name or alias.
            tuning_strategy: ``"inference"``, ``"finetune"`` or ``"peft"``.
            model_params: Model parameters (``checkpoint``, ``device``, ...).
            tuning_params: Tuning parameters.
            label: Display name; defaults to the model name, followed by the
                strategy when it is not ``"inference"``.
            **pipeline_kwargs: Other ``TimeSeriesPipeline`` arguments.

        Returns:
            ``self``, so calls chain.
        """
        check_forecasting_only(pipeline_kwargs, context="TimeSeriesLeaderboard")
        kwargs = {
            "tuning_strategy": tuning_strategy,
            "model_params": dict(model_params or {}),
            "tuning_params": dict(tuning_params or {}),
            **pipeline_kwargs,
        }
        self.models_to_run.append(
            {
                "model_name": model_name,
                "label": label
                or (model_name if tuning_strategy == "inference" else f"{model_name} / {tuning_strategy}"),
                "kwargs": kwargs,
            }
        )
        return self

    def add_models(
        self, model_names: Iterable[str], tuning_strategy: str = "inference", **kwargs: Any
    ) -> TimeSeriesLeaderboard:
        """Queue several models sharing one configuration."""
        for name in model_names:
            self.add_model(name, tuning_strategy=tuning_strategy, **kwargs)
        return self

    def add_all(
        self,
        *,
        commercial_ok: bool | None = None,
        strategy: str = "inference",
        include_tabular: bool = False,
        include_unverified_licenses: bool = False,
        **kwargs: Any,
    ) -> TimeSeriesLeaderboard:
        """Queue every registered model that can read the schema.

        Args:
            commercial_ok: Keep only models cleared for commercial use.
            strategy: Tuning strategy applied to each.
            include_tabular: Also queue the ``TabularTS-*`` forecasters.
            include_unverified_licenses: With ``commercial_ok=True``, also queue
                models whose weight license TabTune has not verified.
            **kwargs: Forwarded to :meth:`add_model`.
        """
        from ..registry.TimeSeries import list_time_series_models

        schema = self.schema
        for spec in list_time_series_models(
            task="forecasting",
            strategy=strategy,
            commercial_ok=commercial_ok,
            include_unverified_licenses=include_unverified_licenses,
        ):
            if spec.name.startswith("TabularTS-") and not include_tabular:
                continue
            if schema.is_multivariate and not spec.supports_multivariate:
                continue
            if schema.covariate_names and not spec.supports_covariates:
                continue
            self.add_model(spec.name, tuning_strategy=strategy, **kwargs)
        return self

    @logged_operation("leaderboard")
    def run(
        self,
        rank_by: str | None = None,
        *,
        display: bool = True,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        """Fit and score every queued configuration.

        A failing configuration is recorded with ``Status='failed'``.

        Args:
            rank_by: Metric to sort by. Defaults to ``"wql"`` when every
                configuration forecasts quantiles, else ``"mase"``.
            display: Log the table.
            progress: Callback ``(index, total, label)``.

        Returns:
            The ranked table (see :meth:`to_frame`).
        """
        if not self.models_to_run:
            raise ValueError("No models queued; call add_model() first.")
        self.entries = []
        train = self._training_frame()
        total = len(self.models_to_run)
        logger.info(
            "[TimeSeriesLeaderboard] Starting leaderboard run: %d configuration(s), mode '%s'",
            total,
            self.mode,
        )
        for index, config in enumerate(track(self.models_to_run, logger=logger, description="Configurations processed")):
            if progress is not None:
                progress(index, total, config["label"])
            logger.info("[TimeSeriesLeaderboard] Running %d/%d: %s", index + 1, total, config["label"])
            entry = self._run_one(config, train)
            if entry.ok:
                logger.info(
                    "[TimeSeriesLeaderboard] %s complete (fit %.2fs, predict %.2fs)",
                    config["label"],
                    entry.fit_seconds,
                    entry.predict_seconds,
                )
            self.entries.append(entry)
        frame = self.to_frame(rank_by)
        if display:
            self.show()
        return frame

    def _training_frame(self) -> pd.DataFrame:
        """``df`` without the backtest windows, so no configuration trains on them."""
        if self.mode == "holdout":
            return self.df
        from .data import split_horizon

        horizon = self.forecast_params["prediction_length"]
        held_out = horizon + (self.windows - 1) * (self.step or horizon)
        return split_horizon(self.df, self.schema, held_out)[0]

    def _run_one(self, config: dict[str, Any], train: pd.DataFrame) -> TimeSeriesLeaderboardEntry:
        from .pipeline import TimeSeriesPipeline

        kwargs = dict(config["kwargs"])
        kwargs.setdefault("cache", self.cache)
        entry = TimeSeriesLeaderboardEntry(
            model_name=config["model_name"],
            tuning_strategy=kwargs["tuning_strategy"],
            label=config["label"],
            pipeline_kwargs={k: v for k, v in kwargs.items() if k != "cache"},
        )
        started = time.perf_counter()
        try:
            pipe = TimeSeriesPipeline(
                config["model_name"], forecast_params=self.forecast_params, **kwargs
            )
            entry.model_name = pipe.model_name
            entry.license_name = pipe.spec.license.name
            entry.commercial_use_ok = pipe.spec.license.commercial_use_ok
            future = self.future_df if self.mode == "holdout" else None
            pipe.fit(train, self.schema, future_df=future)
            entry.fit_seconds = time.perf_counter() - started
            predicted = time.perf_counter()
            if self.mode == "holdout":
                entry.metrics = pipe.evaluate(self.df_actual)
                entry.metrics.pop("per_target", None)
            else:
                full = self.schema.to_panel(self.df, native_missing=pipe._accepts_missing())
                rows = pipe._backtest_rows(full, self.windows, self.step)
                levels = list(pipe.forecast_config.quantile_levels) if rows.attrs["quantiles"] else []
                entry.metrics = score_forecast(rows, levels, rows.attrs["scales"], series_key=["__row__"])
            entry.predict_seconds = time.perf_counter() - predicted
        except Exception as exc:
            entry.error = f"{type(exc).__name__}: {exc}"
            entry.traceback = traceback.format_exc()
            if not np.isfinite(entry.fit_seconds):
                entry.fit_seconds = time.perf_counter() - started
            logger.error("[TimeSeriesLeaderboard] %s failed: %s", config["label"], entry.error)
        return entry

    def _rank_metric(self, rank_by: str | None) -> str:
        """``rank_by``, or the first of WQL, MASE and MAE that every successful entry reports."""
        reported = [set(e.metrics) for e in self.entries if e.ok]
        metrics = set().union(*reported) if reported else set()
        if rank_by is not None:
            if metrics and rank_by not in metrics:
                raise ValueError(f"rank_by={rank_by!r} is not a reported metric: {sorted(metrics)}")
            return rank_by
        for candidate in ("wql", "mase", "mae"):
            if reported and all(candidate in r for r in reported):
                return candidate
        return "mae"

    def to_frame(self, rank_by: str | None = None) -> pd.DataFrame:
        """The results, best first, with a ``Rank`` column (failed entries last)."""
        if not self.entries:
            return pd.DataFrame()
        metric = self._rank_metric(rank_by)
        self._rank_by = metric
        frame = pd.DataFrame([e.to_row() for e in self.entries])
        if metric in frame:
            frame = frame.sort_values(
                metric, ascending=LOWER_IS_BETTER.get(metric, True), na_position="last", kind="mergesort"
            )
        frame.insert(0, "Rank", range(1, len(frame) + 1))
        return frame.reset_index(drop=True)

    @property
    def results(self) -> pd.DataFrame:
        """The ranked table of the last run."""
        return self.to_frame(self._rank_by)

    def best(self, rank_by: str | None = None) -> TimeSeriesLeaderboardEntry | None:
        """The best successful entry, or ``None`` if every entry failed."""
        metric = self._rank_metric(rank_by)
        scored = [e for e in self.entries if e.ok and np.isfinite(e.metrics.get(metric, np.nan))]
        if not scored:
            return None
        sign = 1 if LOWER_IS_BETTER.get(metric, True) else -1
        return min(scored, key=lambda e: sign * e.metrics[metric])

    def show(self) -> None:
        """Render the ranked table through the configured logging sinks."""
        frame = self.results
        if frame.empty:
            logger.info("No results to display; call run() first.")
            return
        columns = [c for c in ("Model", "Strategy", "Status", self._rank_by, "fit_s", "predict_s") if c in frame]
        frame = frame[columns]
        log_table(logger, f"Leaderboard / ranked by {self._rank_by}",
                  frame.columns, frame.itertuples(index=False, name=None))

    def to_markdown(self, path: str | Path | None = None) -> str:
        """The ranked table as Markdown; written to ``path`` when given."""
        frame = self.results.drop(columns=["Error"], errors="ignore")
        text = frame.to_markdown(index=False, floatfmt=".4g")
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
            logger.info("[TimeSeriesLeaderboard] Wrote Markdown to %s", path)
        return text

    def to_csv(self, path: str | Path) -> Path:
        """Write the ranked table as CSV."""
        path = Path(path)
        self.results.to_csv(path, index=False)
        logger.info("[TimeSeriesLeaderboard] Wrote CSV to %s", path)
        return path

    def to_json(self, path: str | Path) -> Path:
        """Write the settings and entries as JSON."""
        path = Path(path)
        payload = {
            "mode": self.mode,
            "windows": self.windows,
            "forecast_params": self.forecast_params,
            "rank_by": self._rank_by,
            "entries": [
                {**e.to_row(), "pipeline_kwargs": e.pipeline_kwargs}
                for e in self.entries
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("[TimeSeriesLeaderboard] Wrote JSON to %s", path)
        return path

    def __repr__(self) -> str:
        return (
            f"TimeSeriesLeaderboard(mode={self.mode!r}, queued={len(self.models_to_run)}, "
            f"completed={len(self.entries)})"
        )
