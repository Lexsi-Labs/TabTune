"""Evaluation: metrics, shift-aware splits and shift-gap reporting.

Two things live here.

**Shared metrics.** Every number TabTune reports - accuracy, ROC AUC, RMSE,
ECE, Brier - is computed by :mod:`tabtune.evaluation.metrics`, so the pipeline,
the leaderboard, the benchmark CSV and the shift report agree by construction
rather than by coincidence.

**Shift-aware evaluation.** An IID cross-validation score is the wrong question
for a model that will be deployed against future or out-of-cohort data.
:class:`TemporalSplit` and :class:`GroupedSplit` ask the right one, and
:class:`ShiftEvaluator` reports the *gap* between the two, which is the number
that predicts production behaviour.

Example:
    >>> from tabtune.evaluation import TemporalSplit, ShiftEvaluator
    >>> evaluator = ShiftEvaluator(splits={"temporal": TemporalSplit(4, time_col="date")})
    >>> report = evaluator.run(factory, X, y)      # doctest: +SKIP
    >>> print(report)                              # doctest: +SKIP
    ShiftReport(TabICLv2, task=classification, metric=roc_auc_score)
      iid                  roc_auc_score=0.9124 (baseline)
      temporal             roc_auc_score=0.8689  gap -0.0435
"""

from __future__ import annotations

from .metrics import (
    CLASSIFICATION_METRICS,
    HIGHER_IS_BETTER,
    REGRESSION_METRICS,
    calibration_metrics,
    classification_metrics,
    compute_metrics,
    expected_calibration_error,
    format_metrics,
    is_higher_better,
    primary_metric,
    regression_metrics,
)
from .shift import FoldResult, ShiftEvaluator, ShiftReport, shift_gap
from .splits import (
    SPLIT_REGISTRY,
    GroupedSplit,
    StratifiedGroupedSplit,
    TemporalSplit,
    resolve_split,
)

__all__ = [
    # metrics
    "compute_metrics",
    "classification_metrics",
    "regression_metrics",
    "calibration_metrics",
    "expected_calibration_error",
    "primary_metric",
    "is_higher_better",
    "format_metrics",
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "HIGHER_IS_BETTER",
    # splits
    "TemporalSplit",
    "GroupedSplit",
    "StratifiedGroupedSplit",
    "resolve_split",
    "SPLIT_REGISTRY",
    # shift
    "ShiftEvaluator",
    "ShiftReport",
    "FoldResult",
    "shift_gap",
]
