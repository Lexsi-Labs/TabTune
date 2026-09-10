# Shift-Aware Evaluation

*New in TabTune 0.2.0.*

An IID cross-validation score answers *"how well does this model fit data drawn like the
training data?"* A deployed model faces a different question: *"how well does it hold up on
data from next quarter, or from a cohort it has never seen?"*

`tabtune.evaluation` provides splitters that ask the right question and a `ShiftEvaluator`
that reports the **gap** between the two — which is the number that predicts production
behaviour.

```python
from tabtune.evaluation import (
    TemporalSplit, GroupedSplit, StratifiedGroupedSplit,
    ShiftEvaluator, ShiftReport, FoldResult, shift_gap, resolve_split,
)
```

---

## 1. Why the gap, not the score

> A model scoring **0.87 with a 0.004 gap** is a better production bet than one scoring
> **0.89 with a 0.06 gap**.

The model that wins on the IID split is not necessarily the one that holds up under drift.

---

## 2. Splitters

All three are scikit-learn-compatible (`split()` / `get_n_splits()`) and enforce their own
invariants.

### 2.1 `TemporalSplit`

Forward chaining — training data is always strictly older than test data. Never trains on
the future.

```python
TemporalSplit(
    n_splits=5,
    *,
    time_col=None,        # column in X holding the timestamp
    gap=0,                # rows to skip between train and test (embargo)
    max_train_size=None,  # rolling instead of expanding window
    test_size=None,
)
```

```python
temporal = TemporalSplit(n_splits=4, time_col="application_date", gap=24)
for train_idx, test_idx in temporal.split(X):
    assert dates[train_idx].max() < dates[test_idx].min()
```

Pass times explicitly instead of by column with `split(X, times=...)`.

### 2.2 `GroupedSplit`

Leave-groups-out — no group appears on both sides of a fold.

```python
GroupedSplit(n_splits=5, *, group_col=None, shuffle=False, random_state=None)
```

```python
grouped = GroupedSplit(n_splits=4, group_col="region")
for train_idx, test_idx in grouped.split(X):
    assert set(regions[train_idx]).isdisjoint(regions[test_idx])
```

### 2.3 `StratifiedGroupedSplit`

Grouped **and** class-balanced — leave-groups-out while keeping the class distribution
comparable across folds.

```python
StratifiedGroupedSplit(n_splits=5, *, group_col=None, random_state=None)
```

Groups can also be supplied as an array via `split(X, y, groups=site_ids)` instead of by
column name.

---

## 3. `ShiftEvaluator`

```python
ShiftEvaluator(
    splits=None,                # mapping name -> splitter, or list of names
    *,
    baseline="iid",             # the split the gap is measured against
    task_type="classification",
    n_splits=5,
    random_state=42,
    error_score="record",       # 'record' keeps going; or a float; or 'raise'
)
```

```python
report = evaluator.run(
    pipeline_factory,           # callable returning an UNFITTED estimator
    X, y,
    *,
    groups=None,
    model_name=None,
    drop_split_columns=None,    # columns to remove from the features
)
```

!!! warning "Pass a factory, not a fitted model"
    Each fold needs an unfitted estimator. Reusing one instance leaks the previous fold's
    state.

!!! warning "Drop the split-defining columns"
    Without `drop_split_columns`, the model can read the very column that defines the split
    and learn the cut point instead of the signal.

### 3.1 A complete run

```python
from tabtune import TabularPipeline
from tabtune.evaluation import ShiftEvaluator, TemporalSplit, GroupedSplit

evaluator = ShiftEvaluator(
    splits={
        "temporal": TemporalSplit(4, time_col="application_date", gap=24),
        "grouped":  GroupedSplit(4, group_col="region"),
    },
    task_type="classification",
    n_splits=4,
)

report = evaluator.run(
    lambda: TabularPipeline("TabICLv2", task_type="classification", cache="memory"),
    X, y,
    model_name="TabICLv2",
    drop_split_columns=["application_date", "region"],
)

print(report)
```

```
ShiftReport(TabICLv2, task=classification, metric=roc_auc_score)
  iid                  roc_auc_score=0.9124 (baseline)
  temporal             roc_auc_score=0.8689  gap -0.0435
  grouped              roc_auc_score=0.9002  gap -0.0122
```

A negative gap means worse under shift, whichever direction the metric runs.

---

## 4. Reading a `ShiftReport`

| Method | Returns |
|---|---|
| `split_names()` | Every split that ran, baseline first |
| `mean_metrics(name)` | Mean of each metric across that split's folds |
| `std_metrics(name)` | Standard deviation across folds |
| `shift_gap()` | Gap on the primary metric, per split |
| `gap_for(metric)` | Gap on any computed metric, per split |
| `failures()` | The `FoldResult`s that errored |
| `to_frame()` | Tidy per-fold DataFrame |
| `summary_frame()` | One row per split |
| `to_dict()` | JSON-serialisable payload for model cards and CI |

```python
summary = report.summary_frame()
print(summary[["split", "folds", "accuracy", "roc_auc_score", "ece", "shift_gap"]].round(4))

for metric in ("accuracy", "roc_auc_score", "ece"):
    print(metric, report.gap_for(metric))
```

!!! note "Check `failures()` before quoting any aggregate"
    With the default `error_score='record'` a fold can fail for ordinary reasons (a class
    missing from a temporal fold, say) and the report keeps going. The aggregate is over the
    folds that succeeded.

---

## 5. Comparing models under shift

```python
from tabtune.evaluation import shift_gap

for name, factory in [("TabICLv2", f1), ("OrionMSP", f2)]:
    result = ShiftEvaluator(
        splits={"temporal": TemporalSplit(4, time_col="application_date")},
        task_type="classification",
    ).run(factory, X, y, model_name=name, drop_split_columns=["application_date"])

    iid, shifted = result.mean_metrics("iid"), result.mean_metrics("temporal")
    print(name, shift_gap(iid, shifted, "roc_auc_score"))
```

---

## 6. Shared metrics

Every number TabTune reports — accuracy, ROC AUC, RMSE, ECE, Brier — is computed by
`tabtune.evaluation.metrics`, so the pipeline, the leaderboard, the benchmark CSV and the
shift report agree by construction rather than by coincidence.

```python
from tabtune.evaluation import (
    compute_metrics, classification_metrics, regression_metrics,
    calibration_metrics, expected_calibration_error,
    primary_metric, is_higher_better, format_metrics,
    CLASSIFICATION_METRICS, REGRESSION_METRICS, HIGHER_IS_BETTER,
)
```

---

## 7. Caching pays off here

Shift sweeps query the same test rows repeatedly. Enabling a disk cache on the factory's
pipeline avoids recomputing them:

```python
lambda: TabularPipeline("TabICLv2", cache="disk")
```

See [Prediction Caching](caching.md).

---

## See Also

- [Uncertainty Quantification](uncertainty.md) — conformal coverage also degrades under shift
- [Model Comparison](leaderboard.md)
- Runnable example: `examples/15_shift_aware_evaluation.py`
