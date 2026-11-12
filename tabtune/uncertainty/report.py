"""One-call uncertainty summary: calibration metrics plus conformal coverage.

``evaluate_calibration`` answers "are the probabilities honest?";
:class:`~tabtune.uncertainty.ConformalClassifier` answers "can I get sets with
guaranteed coverage?". :func:`uncertainty_report` runs both and adds the
size-stratified coverage score - the diagnostic that shows where the marginal
guarantee hides conditional failures.

.. versionadded:: 0.2.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .conformal import ConformalClassifier, ConformalRegressor, _label_indices

logger = logging.getLogger(__name__)

__all__ = ["size_stratified_coverage", "uncertainty_report"]

#: Strata smaller than this many test points are ignored by the SSCS: a
#: one-point stratum makes the worst-group value a coin flip, not a diagnostic.
_MIN_STRATUM = 10


def size_stratified_coverage(
    set_sizes: np.ndarray,
    covered: np.ndarray,
    *,
    min_stratum: int = _MIN_STRATUM,
) -> float:
    """Worst-group coverage when test points are stratified by set size.

    The definition implemented here: group test points by their **exact**
    prediction-set size, compute empirical coverage within each group with at
    least ``min_stratum`` points, and return the smallest group value. A
    marginally valid conformal predictor can still fail this badly - small sets
    on examples it wrongly finds easy, large sets elsewhere - and that failure
    is exactly what marginal coverage cannot see. Values close to the marginal
    coverage mean the sets adapt honestly; values far below it mean the
    guarantee is being carried by the easy strata.

    Args:
        set_sizes: Prediction-set size per test point, shape ``(n,)``.
        covered: Whether the true label was in the set, shape ``(n,)``.
        min_stratum: Minimum stratum size to include. Strata below it are
            skipped; if every stratum is below it, the marginal coverage is
            returned as the honest fallback.

    Returns:
        The worst qualifying stratum's coverage, or ``nan`` on empty input.

    .. versionadded:: 0.2.0
    """
    sizes = np.asarray(set_sizes).ravel()
    hits = np.asarray(covered, dtype=float).ravel()
    if sizes.size == 0:
        return float("nan")
    worst = float("inf")
    for size in np.unique(sizes):
        mask = sizes == size
        if int(mask.sum()) < min_stratum:
            continue
        worst = min(worst, float(hits[mask].mean()))
    if worst == float("inf"):
        logger.debug(
            "[Uncertainty] No set-size stratum reached %d points; "
            "falling back to marginal coverage.",
            min_stratum,
        )
        return float(hits.mean())
    return worst


def _is_classification(pipeline: Any) -> bool:
    """Decide the task from the pipeline, duck-typing where necessary."""
    task_type = getattr(pipeline, "task_type", None)
    if task_type is not None:
        return task_type == "classification"
    return hasattr(pipeline, "predict_proba")


def uncertainty_report(
    pipeline: Any,
    X_test: Any,
    y_test: Any,
    *,
    X_cal: Any = None,
    y_cal: Any = None,
    alpha: float = 0.1,
    n_bins: int = 15,
    method: str = "lac",
) -> dict[str, float]:
    """Summarise a fitted pipeline's uncertainty behaviour in one dict.

    For classification the calibration block (``ece``, ``mce``, ``brier``) is
    always returned, reusing :func:`tabtune.evaluation.metrics.calibration_metrics`
    so the numbers agree with ``evaluate()`` and the leaderboard. When a
    calibration split is also given, a :class:`ConformalClassifier` is fitted
    on it and evaluated on the test split, adding ``coverage``,
    ``avg_set_size``, ``sscs`` (see :func:`size_stratified_coverage`) and
    ``alpha``.

    The calibration block needs the label of each probability column. It is
    taken from the calibration split's conformal fit, else from the
    estimator's own ``classes_``, else from the classes in ``y_test``. If
    ``y_test`` lacks a class and neither of the others is available, the
    block is NaN and a warning is logged; pass ``X_cal``/``y_cal`` then.

    For regression there are no probabilities to score, so the report *requires*
    the calibration split and returns ``coverage``, ``avg_width`` and ``alpha``
    from a :class:`ConformalRegressor` with the ``'absolute'`` score.

    Args:
        pipeline: Fitted pipeline or scikit-learn estimator.
        X_test: Held-out test features the report is computed on.
        y_test: Test labels/targets.
        X_cal: Optional calibration features, disjoint from both the training
            and test data.
        y_cal: Optional calibration labels/targets.
        alpha: Target miscoverage for the conformal block.
        n_bins: Confidence buckets for ECE/MCE.
        method: Conformal score for classification, ``'lac'`` or ``'aps'``.

    Returns:
        Metric mapping as described above.

    Raises:
        ValueError: If only one of ``X_cal``/``y_cal`` is given, or for a
            regression pipeline without a calibration split.

    .. versionadded:: 0.2.0
    """
    if (X_cal is None) != (y_cal is None):
        raise ValueError(
            "Pass X_cal and y_cal together (or neither); got only one of them."
        )

    if not _is_classification(pipeline):
        if X_cal is None:
            raise ValueError(
                "uncertainty_report for regression needs a calibration split: "
                "pass X_cal and y_cal so a ConformalRegressor can be fitted."
            )
        conformal = ConformalRegressor(pipeline, method="absolute", alpha=alpha)
        conformal.calibrate(X_cal, y_cal)
        measured = conformal.coverage(X_test, y_test)
        return {
            "coverage": measured["coverage"],
            "avg_width": measured["avg_width"],
            "alpha": float(alpha),
        }

    from ..evaluation.metrics import calibration_metrics

    conformal_probe = ConformalClassifier(pipeline, method=method, alpha=alpha)
    proba = np.asarray(pipeline.predict_proba(X_test), dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    labels = None
    if X_cal is not None:
        conformal_probe.calibrate(X_cal, y_cal)
        labels = conformal_probe.classes_
    else:
        # A scikit-learn estimator's own ``classes_`` gives the column order in the
        # label space it was fitted on. TabularPipeline has no ``classes_`` of its
        # own (its model's can be the encoded space), so it is not used here.
        own = getattr(pipeline, "classes_", None)
        if own is not None and len(own) == proba.shape[1]:
            labels = np.asarray(own)
        else:
            n_test = len(np.unique(np.asarray(getattr(y_test, "values", y_test))))
            if n_test != proba.shape[1]:
                logger.warning(
                    "[uncertainty_report] y_test has %d classes but predict_proba has %d "
                    "columns, so the columns cannot be matched to labels: ece, mce and "
                    "brier are NaN. Pass X_cal and y_cal to resolve the class order.",
                    n_test, proba.shape[1],
                )

    calibration = calibration_metrics(y_test, proba, n_bins=n_bins, labels=labels)
    out: dict[str, float] = {
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "brier": calibration["brier_score"],
    }
    if X_cal is None:
        return out

    sets = conformal_probe.predict_set(X_test)
    y_index = _label_indices(y_test, conformal_probe.classes_)
    covered = sets[np.arange(len(y_index)), y_index]
    sizes = sets.sum(axis=1)
    out.update(
        {
            "coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(sizes)),
            "sscs": size_stratified_coverage(sizes, covered),
            "alpha": float(alpha),
        }
    )
    return out
