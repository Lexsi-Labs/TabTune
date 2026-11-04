"""Distribution-shift evaluation: measure the IID-to-shifted performance gap.

An IID cross-validation score answers "how well does this model fit data drawn
like the training data?" A deployed model faces a different question: "how well
does it hold up on data from next quarter, from a different region, from a
cohort it has never seen?"

:class:`ShiftEvaluator` runs both and reports the delta. That delta - the
*shift gap* - is a far better predictor of production behaviour than the IID
score alone, and it is the number a model-risk reviewer asks for.

Example:
    >>> from tabtune.evaluation import ShiftEvaluator, TemporalSplit
    >>> evaluator = ShiftEvaluator(                             # doctest: +SKIP
    ...     splits={"temporal": TemporalSplit(n_splits=4, time_col="date")},
    ... )
    >>> report = evaluator.run(pipeline_factory, X, y)          # doctest: +SKIP
    >>> report.shift_gap                                        # doctest: +SKIP
    {'temporal': -0.043}
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import compute_metrics, is_higher_better, primary_metric
from .splits import resolve_split

logger = logging.getLogger(__name__)

__all__ = ["ShiftEvaluator", "ShiftReport", "FoldResult", "shift_gap"]


@dataclass
class FoldResult:
    """Metrics from one fold of one split scheme."""

    split_name: str
    fold: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    fit_seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this fold completed without error."""
        return self.error is None


@dataclass
class ShiftReport:
    """Aggregated results across split schemes, with the IID-to-shift deltas.

    Attributes:
        task_type: The task the evaluation was run for.
        baseline: Name of the split treated as the IID reference.
        folds: Every fold result, including failures.
        model_name: Optional label for the evaluated model.
    """

    task_type: str
    baseline: str
    folds: list[FoldResult] = field(default_factory=list)
    model_name: str | None = None

    # ------------------------------------------------------------ aggregates

    def split_names(self) -> list[str]:
        """Return the split schemes present, baseline first."""
        names = sorted({f.split_name for f in self.folds})
        if self.baseline in names:
            names.remove(self.baseline)
            names.insert(0, self.baseline)
        return names

    def mean_metrics(self, split_name: str) -> dict[str, float]:
        """Return per-metric means across successful folds of ``split_name``."""
        rows = [f.metrics for f in self.folds if f.split_name == split_name and f.ok]
        if not rows:
            return {}
        keys = sorted({k for row in rows for k in row})
        out: dict[str, float] = {}
        for key in keys:
            values = [row[key] for row in rows if key in row and row[key] == row[key]]
            out[key] = float(np.mean(values)) if values else float("nan")
        return out

    def std_metrics(self, split_name: str) -> dict[str, float]:
        """Return per-metric standard deviations across successful folds."""
        rows = [f.metrics for f in self.folds if f.split_name == split_name and f.ok]
        if len(rows) < 2:
            return {}
        keys = sorted({k for row in rows for k in row})
        out: dict[str, float] = {}
        for key in keys:
            values = [row[key] for row in rows if key in row and row[key] == row[key]]
            out[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return out

    @property
    def shift_gap(self) -> dict[str, float]:
        """Return the signed change in the primary metric versus the baseline.

        A negative value means the model performs *worse* under shift, which is
        the expected direction. The sign is normalised so that negative always
        means degradation, regardless of whether the metric is higher- or
        lower-is-better.
        """
        metric = primary_metric(self.task_type)
        return self.gap_for(metric)

    def gap_for(self, metric: str) -> dict[str, float]:
        """Return the shift gap for a specific metric.

        Args:
            metric: Metric key, e.g. ``"accuracy"`` or ``"rmse"``.

        Returns:
            Mapping from non-baseline split name to signed gap. Negative means
            degradation under shift.
        """
        baseline = self.mean_metrics(self.baseline).get(metric)
        out: dict[str, float] = {}
        if baseline is None or baseline != baseline:
            return out
        sign = 1.0 if is_higher_better(metric) else -1.0
        for name in self.split_names():
            if name == self.baseline:
                continue
            value = self.mean_metrics(name).get(metric)
            if value is None or value != value:
                continue
            out[name] = float(sign * (value - baseline))
        return out

    @property
    def failures(self) -> list[FoldResult]:
        """Return folds that raised."""
        return [f for f in self.folds if not f.ok]

    # ---------------------------------------------------------------- views

    def to_frame(self):
        """Return a tidy DataFrame with one row per fold."""
        import pandas as pd

        rows = []
        for fold in self.folds:
            row = {
                "model": self.model_name,
                "split": fold.split_name,
                "fold": fold.fold,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "fit_seconds": round(fold.fit_seconds, 3),
                "status": "ok" if fold.ok else "failed",
                "error": fold.error,
            }
            row.update(fold.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_frame(self):
        """Return one row per split scheme with means, stds and the shift gap."""
        import pandas as pd

        metric = primary_metric(self.task_type)
        gaps = self.gap_for(metric)
        rows = []
        for name in self.split_names():
            means = self.mean_metrics(name)
            stds = self.std_metrics(name)
            row: dict[str, Any] = {
                "split": name,
                "folds": sum(1 for f in self.folds if f.split_name == name and f.ok),
            }
            for key, value in means.items():
                row[key] = value
                if key in stds:
                    row[f"{key}_std"] = stds[key]
            row["shift_gap"] = 0.0 if name == self.baseline else gaps.get(name, float("nan"))
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, for model cards and CI artefacts."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "baseline": self.baseline,
            "primary_metric": primary_metric(self.task_type),
            "splits": {
                name: {
                    "mean": self.mean_metrics(name),
                    "std": self.std_metrics(name),
                }
                for name in self.split_names()
            },
            "shift_gap": self.shift_gap,
            "n_failures": len(self.failures),
        }

    def __str__(self) -> str:
        metric = primary_metric(self.task_type)
        lines = [
            f"ShiftReport({self.model_name or 'model'}, task={self.task_type}, "
            f"metric={metric})"
        ]
        for name in self.split_names():
            value = self.mean_metrics(name).get(metric, float("nan"))
            marker = " (baseline)" if name == self.baseline else ""
            gap = self.gap_for(metric).get(name)
            gap_text = f"  gap {gap:+.4f}" if gap is not None else ""
            lines.append(f"  {name:<20} {metric}={value:.4f}{marker}{gap_text}")
        if self.failures:
            lines.append(f"  {len(self.failures)} fold(s) failed")
        return "\n".join(lines)


def shift_gap(
    baseline_metrics: Mapping[str, float],
    shifted_metrics: Mapping[str, float],
    metric: str,
) -> float:
    """Return the signed degradation of ``metric`` from baseline to shifted.

    Negative means worse under shift, regardless of the metric's direction.

    Args:
        baseline_metrics: Metrics from the IID split.
        shifted_metrics: Metrics from the shifted split.
        metric: The metric key to compare.

    Returns:
        The signed gap, or ``nan`` when either value is missing.
    """
    base = baseline_metrics.get(metric)
    shifted = shifted_metrics.get(metric)
    if base is None or shifted is None or base != base or shifted != shifted:
        return float("nan")
    sign = 1.0 if is_higher_better(metric) else -1.0
    return float(sign * (shifted - base))


class ShiftEvaluator:
    """Evaluate a model under several split schemes and report the gaps.

    Args:
        splits: Mapping from a label to a splitter. Anything accepted by
            :func:`~tabtune.evaluation.splits.resolve_split` works, including
            plain names such as ``"temporal"``.
        baseline: Label of the split treated as the IID reference. Created
            automatically as stratified K-fold when not present in ``splits``.
        task_type: ``"classification"`` or ``"regression"``.
        n_splits: Folds for the auto-created baseline.
        random_state: Seed for the auto-created baseline.
        error_score: ``"raise"`` to propagate fold failures, or a float used as
            the metric value. Defaults to recording the error and continuing,
            because one broken fold should not lose a whole sweep.

    Example:
        >>> evaluator = ShiftEvaluator(splits={"temporal": "temporal"},
        ...                            task_type="classification")
        >>> evaluator.baseline
        'iid'
    """

    def __init__(
        self,
        splits: Mapping[str, Any] | Sequence[str] | None = None,
        *,
        baseline: str = "iid",
        task_type: str = "classification",
        n_splits: int = 5,
        random_state: int | None = 42,
        error_score: str | float = "record",
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got {task_type!r}"
            )
        if isinstance(error_score, str) and error_score not in ("raise", "record"):
            raise ValueError(
                f"error_score must be 'raise', 'record' or a float, got {error_score!r}"
            )

        self.task_type = task_type
        self.baseline = baseline
        self.n_splits = n_splits
        self.random_state = random_state
        self.error_score = error_score

        if splits is None:
            splits = {}
        elif not isinstance(splits, Mapping):
            splits = {str(name): name for name in splits}

        self.splits: dict[str, Any] = dict(splits)
        if self.baseline not in self.splits:
            self.splits[self.baseline] = resolve_split(
                None,
                task_type=task_type,
                n_splits=n_splits,
                random_state=random_state,
            )

    def run(
        self,
        pipeline_factory: Callable[[], Any],
        X: Any,
        y: Any,
        *,
        groups: Any = None,
        model_name: str | None = None,
        drop_split_columns: Sequence[str] | None = None,
    ) -> ShiftReport:
        """Fit and evaluate ``pipeline_factory()`` across every split scheme.

        A *factory* is required rather than a fitted pipeline: each fold needs
        an unfitted model, and reusing one instance would leak the previous
        fold's state.

        Args:
            pipeline_factory: Zero-argument callable returning an unfitted
                object with ``fit``/``predict`` (and ``predict_proba`` for
                classification).
            X: Feature matrix.
            y: Target vector.
            groups: Group labels, forwarded to splitters that need them.
            model_name: Label recorded in the report.
            drop_split_columns: Columns used only for splitting (a date or
                group key) that must not be seen by the model.

        Returns:
            A populated :class:`ShiftReport`.
        """
        import pandas as pd

        report = ShiftReport(
            task_type=self.task_type, baseline=self.baseline, model_name=model_name
        )

        for split_name, split_spec in self.splits.items():
            splitter = resolve_split(
                split_spec,
                task_type=self.task_type,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )
            logger.info(
                "[ShiftEvaluator] Running split %r (%s)",
                split_name,
                type(splitter).__name__,
            )

            try:
                folds = list(splitter.split(X, y, groups))
            except Exception as exc:
                logger.error("[ShiftEvaluator] Split %r could not be built: %s", split_name, exc)
                report.folds.append(
                    FoldResult(split_name, 0, 0, 0, {}, error=f"split failed: {exc}")
                )
                if self.error_score == "raise":
                    raise
                continue

            for fold_index, (train_idx, test_idx) in enumerate(folds):
                result = self._run_fold(
                    pipeline_factory,
                    X,
                    y,
                    train_idx,
                    test_idx,
                    split_name,
                    fold_index,
                    drop_split_columns,
                    pd,
                )
                report.folds.append(result)

        return report

    # --------------------------------------------------------------- internals

    def _run_fold(
        self,
        pipeline_factory: Callable[[], Any],
        X: Any,
        y: Any,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        split_name: str,
        fold_index: int,
        drop_split_columns: Sequence[str] | None,
        pd: Any,
    ) -> FoldResult:
        """Fit and score a single fold, converting failures into a record."""
        X_train, X_test = _take(X, train_idx, pd), _take(X, test_idx, pd)
        y_train, y_test = _take(y, train_idx, pd), _take(y, test_idx, pd)

        if drop_split_columns:
            keep = [c for c in getattr(X_train, "columns", []) if c not in set(drop_split_columns)]
            if keep:
                X_train, X_test = X_train[keep], X_test[keep]

        started = time.perf_counter()
        try:
            model = pipeline_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = None
            if self.task_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception as exc:  # probabilities are optional
                    logger.debug("[ShiftEvaluator] predict_proba unavailable: %s", exc)
            metrics = compute_metrics(
                y_test, y_pred, y_proba, task_type=self.task_type
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error(
                "[ShiftEvaluator] %s fold %d failed: %s", split_name, fold_index, exc
            )
            if self.error_score == "raise":
                raise
            metrics = {}
            if isinstance(self.error_score, (int, float)):
                metrics = {primary_metric(self.task_type): float(self.error_score)}
            return FoldResult(
                split_name,
                fold_index,
                len(train_idx),
                len(test_idx),
                metrics,
                fit_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )

        return FoldResult(
            split_name,
            fold_index,
            len(train_idx),
            len(test_idx),
            metrics,
            fit_seconds=time.perf_counter() - started,
        )


def _take(data: Any, index: np.ndarray, pd: Any) -> Any:
    """Positionally index a DataFrame, Series or array."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[index]
    return np.asarray(data)[index]
