"""Compare tabular foundation models on a shared dataset.

The leaderboard is the most demonstrable thing in TabTune: it is where the
library's breadth - thirteen models, four adaptation strategies - becomes a
single table a user can act on.

What changed in 0.2.0
---------------------
The previous implementation ranked on three metrics, wrote the string
``"Failed"`` into numeric columns when a model errored, built its DataFrame
twice, and could not be imported outside IPython. It now:

* reports the full metric bundle from :mod:`tabtune.evaluation.metrics`,
  including calibration (ECE, MCE, Brier), plus wall-clock timings and peak
  GPU memory;
* keeps failures as real ``NaN`` with the error preserved in its own column, so
  a broken entry cannot silently coerce a numeric column to ``object`` and
  corrupt the sort;
* records each model's weight license, so "best" and "deployable" can be told
  apart at a glance;
* exports to Markdown, CSV, JSON and a self-contained interactive HTML report
  with a quality-versus-latency Pareto view;
* works headlessly - IPython is imported lazily, only for display.

Example:
    >>> board = TabularLeaderboard(X_train, X_test, y_train, y_test)   # doctest: +SKIP
    >>> board.add_models(["TabICLv2", "OrionMSP", "Mitra"])            # doctest: +SKIP
    >>> board.run(rank_by="roc_auc_score")                             # doctest: +SKIP
    >>> board.to_html("leaderboard.html")                              # doctest: +SKIP
    >>> print(board.summary())                                         # doctest: +SKIP
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._internal.deprecation import warn_once
from ..evaluation.metrics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    compute_metrics,
    is_higher_better,
    primary_metric,
)
from ..TabularPipeline.pipeline import TabularPipeline

logger = logging.getLogger(__name__)

__all__ = ["TabularLeaderboard", "LeaderboardEntry"]

_TEXT_COLUMNS = frozenset(
    {"Model", "Strategy", "Mode", "Status", "License", "Commercial", "Error"}
)


@dataclass
class LeaderboardEntry:
    """One row of the leaderboard: a configuration and how it performed.

    Attributes:
        model_name: The model that was run.
        tuning_strategy: The adaptation strategy used.
        task_type: Task the entry was evaluated for.
        finetune_mode: Fine-tuning algorithm, when applicable.
        label: Optional display override.
        metrics: Metric name to value. Empty when the entry failed.
        fit_seconds: Wall-clock time spent in ``fit``.
        predict_seconds: Wall-clock time producing test predictions.
        peak_memory_mb: Peak GPU memory during the run, when measurable.
        license_name: Weight license of the model, from the registry.
        commercial_use_ok: Tri-state commercial-use flag.
        error: Exception text when the entry failed, else ``None``.
        traceback: Full traceback for a failed entry, for debugging.
    """

    model_name: str
    tuning_strategy: str
    task_type: str
    finetune_mode: str | None = None
    label: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = float("nan")
    predict_seconds: float = float("nan")
    peak_memory_mb: float = float("nan")
    license_name: str = ""
    commercial_use_ok: bool | None = None
    error: str | None = None
    traceback: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this entry completed without raising."""
        return self.error is None

    @property
    def display_name(self) -> str:
        """A human label combining model, strategy and mode."""
        if self.label:
            return self.label
        parts = [self.model_name, self.tuning_strategy]
        if self.finetune_mode and self.tuning_strategy != "inference":
            parts.append(self.finetune_mode)
        return " / ".join(parts)

    @property
    def commercial_badge(self) -> str:
        """Human-readable commercial-use status."""
        if self.commercial_use_ok is True:
            return "yes"
        if self.commercial_use_ok is False:
            return "no"
        return "unverified"

    def to_row(self) -> dict[str, Any]:
        """Return this entry as a flat dict, for DataFrame construction."""
        row: dict[str, Any] = {
            "Model": self.model_name,
            "Strategy": self.tuning_strategy,
            "Mode": self.finetune_mode or "-",
            "Status": "ok" if self.ok else "failed",
        }
        row.update(self.metrics)
        row["fit_s"] = self.fit_seconds
        row["predict_s"] = self.predict_seconds
        if self.peak_memory_mb == self.peak_memory_mb:  # not NaN
            row["peak_mem_mb"] = self.peak_memory_mb
        row["License"] = self.license_name or "-"
        row["Commercial"] = self.commercial_badge
        row["Error"] = self.error
        return row


class TabularLeaderboard:
    """Benchmark several TabTune configurations on one train/test split.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        task_type: ``'classification'`` or ``'regression'``.
        cache: Prediction cache shared by every entry. Defaults to ``'memory'``,
            which removes the duplicate forward passes each evaluation used to
            make.

    Raises:
        ValueError: On an invalid ``task_type`` or mismatched split lengths.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str = "classification",
        *,
        cache: str | None = "memory",
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got {task_type!r}"
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train has {len(X_train)} rows but y_train has {len(y_train)}"
            )
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test has {len(X_test)} rows but y_test has {len(y_test)}"
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type
        self.cache = cache

        self.models_to_run: list[dict[str, Any]] = []
        self.entries: list[LeaderboardEntry] = []
        self._ranked: pd.DataFrame | None = None
        self._rank_by: str | None = None

        logger.info(
            "[Leaderboard] Initialised for %s: %d train / %d test rows, %s features",
            task_type,
            len(X_train),
            len(X_test),
            X_train.shape[1] if hasattr(X_train, "shape") else "?",
        )

    # ------------------------------------------------------------ registration

    def add_model(
        self,
        model_name: str,
        tuning_strategy: str = "inference",
        model_params: dict | None = None,
        tuning_params: dict | None = None,
        finetune_mode: str | None = None,
        task_type: str | None = None,
        *,
        processor_params: dict | None = None,
        label: str | None = None,
        **pipeline_kwargs: Any,
    ) -> TabularLeaderboard:
        """Add a configuration to the leaderboard.

        Args:
            model_name: Model name or alias.
            tuning_strategy: ``'inference'``, ``'finetune'`` or ``'peft'``.
            model_params: Model constructor parameters.
            tuning_params: Adaptation parameters.
            finetune_mode: Fine-tuning algorithm.
            task_type: Override the leaderboard's task type for this entry.
            processor_params: Preprocessing parameters.
            label: Display label; defaults to ``"model / strategy"``.
            **pipeline_kwargs: Forwarded to :class:`TabularPipeline` (for
                example ``envelope_mode`` or ``license_mode``).

        Returns:
            ``self``, so calls chain.

        Raises:
            ValueError: On an invalid ``task_type``.
        """
        task_type = task_type or self.task_type
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got {task_type!r}"
            )

        self.models_to_run.append(
            {
                "model_name": model_name,
                "tuning_strategy": tuning_strategy,
                "model_params": model_params or {},
                "tuning_params": tuning_params or {},
                "processor_params": processor_params or {},
                "finetune_mode": finetune_mode,
                "task_type": task_type,
                "label": label,
                "pipeline_kwargs": pipeline_kwargs,
            }
        )
        logger.info(
            "[Leaderboard] Queued %s (strategy=%s%s)",
            model_name,
            tuning_strategy,
            f", mode={finetune_mode}" if finetune_mode else "",
        )
        return self

    def add_models(
        self,
        model_names: Iterable[str],
        tuning_strategy: str = "inference",
        **kwargs: Any,
    ) -> TabularLeaderboard:
        """Add several models sharing one configuration.

        Example:
            >>> board.add_models(["TabICLv2", "OrionMSP", "Mitra"])   # doctest: +SKIP
        """
        for name in model_names:
            self.add_model(name, tuning_strategy=tuning_strategy, **kwargs)
        return self

    def add_all(
        self,
        *,
        commercial_ok: bool | None = None,
        strategy: str = "inference",
        **kwargs: Any,
    ) -> TabularLeaderboard:
        """Add every registered model supporting this leaderboard's task.

        Args:
            commercial_ok: Restrict to models cleared for commercial use.
            strategy: Tuning strategy applied to each.
            **kwargs: Forwarded to :meth:`add_model`.

        Returns:
            ``self``.

        Example:
            >>> board.add_all(commercial_ok=True)      # doctest: +SKIP
        """
        from ..registry import list_models

        specs = list_models(
            task=self.task_type, strategy=strategy, commercial_ok=commercial_ok
        )
        for spec in specs:
            self.add_model(spec.name, tuning_strategy=strategy, **kwargs)
        logger.info("[Leaderboard] Queued %d registered models", len(specs))
        return self

    # -------------------------------------------------------------------- run

    def run(
        self,
        rank_by: str | None = None,
        *,
        display: bool = True,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        """Fit and evaluate every queued configuration.

        A failing entry is recorded with ``Status='failed'`` and ``NaN`` metrics
        rather than aborting, because a thirteen-model sweep should not be lost
        to one bad checkpoint.

        Args:
            rank_by: Metric to sort by. Defaults to ROC AUC for classification
                and R-squared for regression. Legacy display names such as
                ``'ROC AUC'`` resolve to their canonical keys.
            display: Render the table after running.
            progress: Optional callback ``(index, total, model_name)``.

        Returns:
            The ranked leaderboard, indexed by rank.
        """
        logger.info(
            "[Leaderboard] Starting run over %d configurations", len(self.models_to_run)
        )
        self.entries = []

        total = len(self.models_to_run)
        for index, config in enumerate(self.models_to_run, start=1):
            if progress is not None:
                progress(index, total, config["model_name"])
            logger.info(
                "[Leaderboard] [%d/%d] %s (%s)",
                index,
                total,
                config["model_name"],
                config["tuning_strategy"],
            )
            self.entries.append(self._run_one(config))

        self._rank_by = self._resolve_rank_metric(rank_by)
        self._ranked = self._build_frame(self._rank_by)

        logger.info(
            "[Leaderboard] Complete: %d succeeded, %d failed",
            sum(1 for e in self.entries if e.ok),
            sum(1 for e in self.entries if not e.ok),
        )
        if display:
            self.show()
        return self._ranked

    def _run_one(self, config: Mapping[str, Any]) -> LeaderboardEntry:
        """Fit and evaluate a single configuration, capturing any failure."""
        entry = LeaderboardEntry(
            model_name=config["model_name"],
            tuning_strategy=config["tuning_strategy"],
            finetune_mode=config.get("finetune_mode"),
            task_type=config["task_type"],
            label=config.get("label"),
        )

        try:
            from ..registry import get_model_spec

            spec = get_model_spec(config["model_name"])
            entry.license_name = spec.license.name
            entry.commercial_use_ok = spec.license.commercial_use_ok
        except Exception:
            pass  # unregistered models still run, they just carry no license info

        peak_tracked = _reset_peak_memory()
        fit_started = time.perf_counter()
        try:
            pipeline = TabularPipeline(
                model_name=config["model_name"],
                task_type=config["task_type"],
                tuning_strategy=config["tuning_strategy"],
                model_params=config["model_params"],
                tuning_params=config["tuning_params"],
                processor_params=config["processor_params"],
                finetune_mode=config.get("finetune_mode"),
                cache=self.cache,
                **config.get("pipeline_kwargs", {}),
            )
            pipeline.fit(self.X_train, self.y_train)
            entry.fit_seconds = time.perf_counter() - fit_started

            predict_started = time.perf_counter()
            y_pred = pipeline.predict(self.X_test)
            y_proba = None
            if config["task_type"] == "classification":
                try:
                    y_proba = pipeline.predict_proba(self.X_test)
                except Exception as exc:
                    logger.debug("[Leaderboard] predict_proba unavailable: %s", exc)
            entry.predict_seconds = time.perf_counter() - predict_started

            entry.metrics = compute_metrics(
                self.y_test, y_pred, y_proba, task_type=config["task_type"]
            )
            if peak_tracked:
                entry.peak_memory_mb = _peak_memory_mb()

        except Exception as exc:
            entry.error = f"{type(exc).__name__}: {exc}"
            entry.traceback = traceback.format_exc()
            if entry.fit_seconds != entry.fit_seconds:  # still NaN
                entry.fit_seconds = time.perf_counter() - fit_started
            logger.error("[Leaderboard] %s failed: %s", config["model_name"], entry.error)
            logger.debug("[Leaderboard] Traceback:\n%s", entry.traceback)

        return entry

    # ---------------------------------------------------------------- ranking

    def _resolve_rank_metric(self, rank_by: str | None) -> str:
        """Map a user-supplied ranking key to a canonical metric name."""
        if rank_by is None:
            return primary_metric(self.task_type)

        # Accept the display names the pre-0.2.0 implementation used.
        legacy = {
            "accuracy": "accuracy",
            "f1 score": "f1_score",
            "roc auc": "roc_auc_score",
            "r2 score": "r2_score",
            "mse": "mse",
            "rmse": "rmse",
            "mae": "mae",
        }
        key = str(rank_by).strip()
        canonical = legacy.get(key.lower(), key)

        available = set(CLASSIFICATION_METRICS) | set(REGRESSION_METRICS) | {"ece", "mce"}
        if canonical not in available:
            warn_once(
                f"rank_by={rank_by!r} is not a metric TabTune computes; falling back to "
                f"{primary_metric(self.task_type)!r}. Available: {sorted(available)}.",
                UserWarning,
                key=f"leaderboard-rank:{rank_by}",
            )
            return primary_metric(self.task_type)
        return canonical

    def _build_frame(self, rank_by: str) -> pd.DataFrame:
        """Assemble, coerce and sort the results table."""
        if not self.entries:
            return pd.DataFrame()

        frame = pd.DataFrame([entry.to_row() for entry in self.entries])

        # Failures carry NaN, never a "Failed" sentinel string: text in a
        # numeric column silently coerces the whole column to object dtype and
        # makes every downstream sort and aggregation unreliable.
        for column in frame.columns:
            if column in _TEXT_COLUMNS:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if rank_by in frame.columns:
            ascending = not is_higher_better(rank_by)
            # mergesort keeps insertion order stable among ties; na_position
            # keeps failed entries at the bottom whichever way we sort.
            frame = frame.sort_values(
                rank_by, ascending=ascending, na_position="last", kind="mergesort"
            ).reset_index(drop=True)

        frame.index = frame.index + 1
        frame.index.name = "Rank"

        # Keep the common case readable: no Error column when nothing failed.
        if "Error" in frame.columns and frame["Error"].isna().all():
            frame = frame.drop(columns=["Error"])
        return frame

    # ------------------------------------------------------------------ views

    @property
    def results(self) -> pd.DataFrame:
        """The ranked results, or an empty frame before :meth:`run`."""
        if self._ranked is None:
            return pd.DataFrame()
        return self._ranked

    def to_frame(self, rank_by: str | None = None) -> pd.DataFrame:
        """Return the ranked results, optionally re-ranked by another metric.

        Re-ranking does not re-run anything: every metric is already computed.
        """
        if rank_by is None:
            return self.results
        return self._build_frame(self._resolve_rank_metric(rank_by))

    def best(self, rank_by: str | None = None) -> LeaderboardEntry | None:
        """Return the best-performing successful entry, or ``None``."""
        metric = self._resolve_rank_metric(rank_by or self._rank_by)
        successful = [
            e
            for e in self.entries
            if e.ok and metric in e.metrics and e.metrics[metric] == e.metrics[metric]
        ]
        if not successful:
            return None
        sign = 1.0 if is_higher_better(metric) else -1.0
        return max(successful, key=lambda e: sign * e.metrics[metric])

    def pareto_front(self, metric: str | None = None) -> list[LeaderboardEntry]:
        """Return entries not dominated on both quality and prediction latency.

        An entry is dominated when another is at least as good *and* at least as
        fast. The survivors are the only configurations worth considering:
        everything else is strictly worse on both axes. For tabular foundation
        models this is the decision that actually matters, since the fastest and
        the most accurate model are rarely the same one.

        Args:
            metric: Quality metric. Defaults to the task's primary metric.

        Returns:
            Frontier entries, ordered by increasing prediction latency.
        """
        metric = self._resolve_rank_metric(metric or self._rank_by)
        candidates = [
            e
            for e in self.entries
            if e.ok
            and metric in e.metrics
            and e.metrics[metric] == e.metrics[metric]
            and e.predict_seconds == e.predict_seconds
        ]
        sign = 1.0 if is_higher_better(metric) else -1.0

        front: list[LeaderboardEntry] = []
        for entry in sorted(candidates, key=lambda e: e.predict_seconds):
            score = sign * entry.metrics[metric]
            if all(sign * other.metrics[metric] < score for other in front):
                front.append(entry)
        return front

    def show(self) -> None:
        """Render the leaderboard, using rich display when available.

        Falls back to plain text outside IPython. The previous implementation
        imported ``IPython.display`` at module scope, which made the whole
        module unimportable in a headless environment even though ``IPython``
        is only declared in the optional ``interactive`` extra.
        """
        frame = self.results
        if frame.empty:
            logger.info("[Leaderboard] No results to display; call run() first.")
            return

        try:
            from IPython.display import Markdown, display

            display(Markdown(f"### TabTune Leaderboard - ranked by `{self._rank_by}`"))
            display(frame)
            return
        except Exception:
            pass

        print(f"\nTabTune Leaderboard - ranked by {self._rank_by}")
        print(frame.to_string())

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render the leaderboard as a Markdown table.

        Args:
            path: Optional destination file.

        Returns:
            The Markdown text.
        """
        frame = self.results
        if frame.empty:
            return "_No leaderboard results; call run() first._"
        try:
            text = frame.to_markdown(floatfmt=".4f")
        except ImportError:  # tabulate is optional
            text = "```\n" + frame.to_string() + "\n```"
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
            logger.info("[Leaderboard] Wrote Markdown to %s", path)
        return text

    def to_csv(self, path: str | Path) -> Path:
        """Write the leaderboard to CSV and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(path)
        logger.info("[Leaderboard] Wrote CSV to %s", path)
        return path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        best = self.best()
        return {
            "task_type": self.task_type,
            "rank_by": self._rank_by,
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "n_features": (
                int(self.X_train.shape[1]) if hasattr(self.X_train, "shape") else None
            ),
            "best": best.display_name if best else None,
            "entries": [
                {
                    "model": e.model_name,
                    "strategy": e.tuning_strategy,
                    "finetune_mode": e.finetune_mode,
                    "status": "ok" if e.ok else "failed",
                    "metrics": e.metrics,
                    "fit_seconds": e.fit_seconds,
                    "predict_seconds": e.predict_seconds,
                    "peak_memory_mb": e.peak_memory_mb,
                    "license": e.license_name,
                    "commercial_use_ok": e.commercial_use_ok,
                    "error": e.error,
                }
                for e in self.entries
            ],
        }

    def to_json(self, path: str | Path) -> Path:
        """Write the run summary to JSON and return the path."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=_json_default)
            handle.write("\n")
        logger.info("[Leaderboard] Wrote JSON to %s", path)
        return path

    def to_html(
        self,
        path: str | Path | None = None,
        *,
        title: str = "TabTune Leaderboard",
        include_pareto: bool = True,
    ) -> str:
        """Render a self-contained interactive HTML report.

        The report needs no network access and no JavaScript dependencies, so it
        can be committed to a repository, attached to a pull request or served
        from a static bucket.

        Args:
            path: Optional destination file.
            title: Report heading.
            include_pareto: Include the quality-versus-latency scatter.

        Returns:
            The HTML source.
        """
        from .report import render_leaderboard_html

        html = render_leaderboard_html(self, title=title, include_pareto=include_pareto)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            logger.info("[Leaderboard] Wrote HTML report to %s", path)
        return html

    def summary(self) -> str:
        """Return a short human-readable summary of the run."""
        if not self.entries:
            return "Leaderboard has not been run yet."

        succeeded = sum(1 for e in self.entries if e.ok)
        lines = [
            f"TabTune leaderboard: {succeeded}/{len(self.entries)} configurations succeeded."
        ]

        best = self.best()
        if best is not None:
            metric = self._rank_by or primary_metric(self.task_type)
            lines.append(
                f"Best: {best.display_name} with {metric}={best.metrics[metric]:.4f} "
                f"({best.predict_seconds:.2f}s to predict, "
                f"license {best.license_name or 'unknown'})."
            )

        front = self.pareto_front()
        if len(front) > 1:
            lines.append(
                "Pareto front (quality vs latency): "
                + ", ".join(e.display_name for e in front)
            )

        deployable = [e for e in self.entries if e.ok and e.commercial_use_ok is True]
        if deployable and best is not None and best.commercial_use_ok is not True:
            metric = self._rank_by or primary_metric(self.task_type)
            sign = 1.0 if is_higher_better(metric) else -1.0
            top = max(
                (e for e in deployable if metric in e.metrics),
                key=lambda e: sign * e.metrics[metric],
                default=None,
            )
            if top is not None:
                lines.append(
                    f"Best commercially deployable: {top.display_name} "
                    f"({metric}={top.metrics[metric]:.4f}, {top.license_name})."
                )

        failures = [e for e in self.entries if not e.ok]
        if failures:
            lines.append(
                "Failed: " + ", ".join(f"{e.display_name} ({e.error})" for e in failures)
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.models_to_run)

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"TabularLeaderboard(task={self.task_type!r}, "
            f"queued={len(self.models_to_run)}, completed={len(self.entries)})"
        )


def _json_default(value: Any) -> Any:
    """Coerce numpy scalars so ``json.dump`` accepts them."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _reset_peak_memory() -> bool:
    """Reset CUDA peak-memory tracking. Returns whether tracking is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:  # pragma: no cover - CUDA absent
        pass
    return False


def _peak_memory_mb() -> float:
    """Return peak CUDA memory in MiB since the last reset, or ``nan``."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**2)
    except Exception:  # pragma: no cover
        pass
    return float("nan")
