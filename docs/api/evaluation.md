# API: Evaluation

*New in 0.2.0.* Shared metrics, shift-aware splits and shift-gap reporting.
Narrative guide: [Shift-Aware Evaluation](../user-guide/shift-evaluation.md).

```python
from tabtune.evaluation import (
    # splits
    TemporalSplit, GroupedSplit, StratifiedGroupedSplit, resolve_split, SPLIT_REGISTRY,
    # shift
    ShiftEvaluator, ShiftReport, FoldResult, shift_gap,
    # metrics
    compute_metrics, classification_metrics, regression_metrics, calibration_metrics,
    expected_calibration_error, primary_metric, is_higher_better, format_metrics,
    CLASSIFICATION_METRICS, REGRESSION_METRICS, HIGHER_IS_BETTER,
)
```

---

## Splits

::: tabtune.evaluation.splits.TemporalSplit
    options:
      show_source: true

::: tabtune.evaluation.splits.GroupedSplit
    options:
      show_source: true

::: tabtune.evaluation.splits.StratifiedGroupedSplit
    options:
      show_source: true

::: tabtune.evaluation.splits.resolve_split
    options:
      show_source: true

---

## Shift evaluation

::: tabtune.evaluation.shift.ShiftEvaluator
    options:
      show_source: true

::: tabtune.evaluation.shift.ShiftReport
    options:
      show_source: true

::: tabtune.evaluation.shift.FoldResult
    options:
      show_source: true

::: tabtune.evaluation.shift.shift_gap
    options:
      show_source: true

---

## Metrics

::: tabtune.evaluation.metrics
    options:
      show_source: true
      members:
        - compute_metrics
        - classification_metrics
        - regression_metrics
        - calibration_metrics
        - expected_calibration_error
        - primary_metric
        - is_higher_better
        - format_metrics
