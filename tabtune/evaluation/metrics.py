"""Shared metric computation.

Every metric TabTune reports is computed here. Before this module the same
formulas were reimplemented in ``TabularPipeline.evaluate``, the distillation
analysis package and the ensemble strategies, and they had already drifted (two
different expected-calibration-error implementations with different binning).

Centralising them means a metric added here appears in ``evaluate()``, the
leaderboard, the benchmark CSV and the shift report at once.

All functions are pure, take numpy-compatible inputs and never raise on
degenerate input: an undefined metric returns ``float('nan')`` rather than
blowing up a benchmark sweep halfway through.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "classification_metrics",
    "regression_metrics",
    "calibration_metrics",
    "expected_calibration_error",
    "compute_metrics",
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "HIGHER_IS_BETTER",
]

#: Metric keys reported for classification, in display order.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc_score",
    "mcc",
    "log_loss",
    "brier_score",
)

#: Metric keys reported for regression, in display order.
REGRESSION_METRICS: tuple[str, ...] = (
    "r2_score",
    "rmse",
    "mae",
    "mse",
    "median_absolute_error",
    "explained_variance",
    "max_error",
)

#: Direction of improvement per metric. Anything absent is assumed
#: higher-is-better, which is the common case for scores.
HIGHER_IS_BETTER: dict[str, bool] = {
    "accuracy": True,
    "balanced_accuracy": True,
    "f1_score": True,
    "precision": True,
    "recall": True,
    "roc_auc_score": True,
    "mcc": True,
    "r2_score": True,
    "explained_variance": True,
    "log_loss": False,
    "brier_score": False,
    "ece": False,
    "mce": False,
    "rmse": False,
    "mae": False,
    "mse": False,
    "median_absolute_error": False,
    "max_error": False,
}


def _safe(func, *args, default: float = float("nan"), **kwargs) -> float:
    """Call ``func`` returning ``default`` when the metric is undefined.

    Single-class test folds, constant predictions and all-NaN targets are
    normal in a benchmark sweep. They should produce a missing value, not abort
    the run.
    """
    try:
        value = func(*args, **kwargs)
    except Exception as exc:
        logger.debug("[Metrics] %s undefined: %s", getattr(func, "__name__", func), exc)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(getattr(values, "values", values))


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Compute the standard classification metric bundle.

    Args:
        y_true: Ground-truth labels, shape ``(n,)``.
        y_pred: Predicted labels, shape ``(n,)``.
        y_proba: Predicted probabilities, shape ``(n, n_classes)`` or ``(n,)``
            for binary. Optional; probability-dependent metrics are omitted
            when absent.
        labels: Explicit class ordering matching ``y_proba``'s columns.

    Returns:
        Mapping from metric name to value. Undefined metrics are ``nan``.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)

    out: dict[str, float] = {
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "f1_score": _safe(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "precision": _safe(precision_score, y_true, y_pred, average="weighted", zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, average="weighted", zero_division=0),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
    }

    if y_proba is None:
        return out

    proba = np.asarray(y_proba, dtype=float)
    classes = np.unique(y_true) if labels is None else np.asarray(labels)

    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    if proba.shape[1] == 2 and len(classes) == 2:
        out["roc_auc_score"] = _safe(roc_auc_score, y_true, proba[:, 1])
    else:
        out["roc_auc_score"] = _safe(
            roc_auc_score, y_true, proba, multi_class="ovr", average="weighted"
        )

    out["log_loss"] = _safe(log_loss, y_true, proba, labels=classes)
    out.update(calibration_metrics(y_true, proba, labels=classes))
    return out


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute the standard regression metric bundle.

    Args:
        y_true: Ground-truth targets, shape ``(n,)``.
        y_pred: Predicted targets, shape ``(n,)``.

    Returns:
        Mapping from metric name to value. Undefined metrics are ``nan``.
    """
    from sklearn.metrics import (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    y_true = _as_array(y_true).astype(float, copy=False)
    y_pred = _as_array(y_pred).astype(float, copy=False)

    mse = _safe(mean_squared_error, y_true, y_pred)
    return {
        "r2_score": _safe(r2_score, y_true, y_pred),
        "rmse": float(np.sqrt(mse)) if mse == mse else float("nan"),
        "mae": _safe(mean_absolute_error, y_true, y_pred),
        "mse": mse,
        "median_absolute_error": _safe(median_absolute_error, y_true, y_pred),
        "explained_variance": _safe(explained_variance_score, y_true, y_pred),
        "max_error": _safe(max_error, y_true, y_pred),
    }


def expected_calibration_error(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
    labels: Sequence[Any] | None = None,
    strategy: str = "uniform",
) -> tuple[float, float]:
    """Return the expected and maximum calibration error.

    Predictions are bucketed by their top-class confidence; within each bucket
    the gap between mean confidence and observed accuracy is measured. ECE is
    the sample-weighted mean of those gaps, MCE the largest.

    Args:
        y_true: Ground-truth labels, shape ``(n,)``.
        y_proba: Predicted probabilities, shape ``(n, n_classes)``.
        n_bins: Number of confidence buckets.
        labels: Class ordering matching ``y_proba``'s columns. Defaults to
            ``np.unique(y_true)``.
        strategy: ``"uniform"`` for equal-width bins, ``"quantile"`` for
            equal-count bins. Quantile binning is more stable when confidences
            cluster near 1.0, which they do for a well-fitted TFM.

    Returns:
        ``(ece, mce)``. Both are ``nan`` for empty input.
    """
    y_true = _as_array(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.size == 0 or y_true.size == 0:
        return float("nan"), float("nan")
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    classes = np.unique(y_true) if labels is None else np.asarray(labels)
    if proba.shape[1] != len(classes):
        # Mismatched shapes mean we cannot map columns to labels reliably.
        logger.debug(
            "[Metrics] Calibration skipped: proba has %d columns but %d classes",
            proba.shape[1],
            len(classes),
        )
        return float("nan"), float("nan")

    confidence = proba.max(axis=1)
    predicted = classes[proba.argmax(axis=1)]
    correct = (predicted == y_true).astype(float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    total = len(confidence)
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > low) & (confidence <= high)
        if low == edges[0]:
            mask |= confidence == low
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        ece += (count / total) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def calibration_metrics(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
    labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Compute calibration diagnostics: ECE, MCE and the multi-class Brier score.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities, shape ``(n, n_classes)``.
        n_bins: Confidence buckets for ECE/MCE.
        labels: Class ordering matching ``y_proba``'s columns.

    Returns:
        Mapping with ``ece``, ``mce`` and ``brier_score``.
    """
    y_true = _as_array(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    classes = np.unique(y_true) if labels is None else np.asarray(labels)
    ece, mce = expected_calibration_error(y_true, proba, n_bins=n_bins, labels=classes)

    brier = float("nan")
    if proba.shape[1] == len(classes) and len(y_true):
        # Multi-class Brier: mean squared error against the one-hot target.
        index = {label: i for i, label in enumerate(classes)}
        onehot = np.zeros_like(proba)
        for row, label in enumerate(y_true):
            column = index.get(label)
            if column is not None:
                onehot[row, column] = 1.0
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    return {"ece": ece, "mce": mce, "brier_score": brier}


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    *,
    task_type: str = "classification",
    labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Dispatch to the right metric bundle for ``task_type``.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        y_proba: Probabilities, classification only.
        task_type: ``"classification"`` or ``"regression"``.
        labels: Class ordering for probability columns.

    Returns:
        The metric bundle.

    Raises:
        ValueError: On an unknown ``task_type``.
    """
    if task_type == "classification":
        return classification_metrics(y_true, y_pred, y_proba, labels=labels)
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    raise ValueError(
        f"task_type must be 'classification' or 'regression', got {task_type!r}"
    )


def primary_metric(task_type: str) -> str:
    """Return the metric used for ranking by default."""
    return "roc_auc_score" if task_type == "classification" else "r2_score"


def is_higher_better(metric: str) -> bool:
    """Return whether larger values of ``metric`` are better."""
    return HIGHER_IS_BETTER.get(metric, True)


def format_metrics(metrics: Mapping[str, float], *, precision: int = 4) -> str:
    """Render a metric mapping as a compact single line, for logs."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value == value:
            parts.append(f"{key}={value:.{precision}f}")
    return " ".join(parts)
