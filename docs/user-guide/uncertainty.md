# Uncertainty Quantification: Conformal Prediction & Recalibration

*New in TabTune 0.2.0.*

Tabular foundation models win benchmarks on accuracy and lose them on uncertainty: their
probabilities come out of an in-context softmax that was never calibrated to *your* dataset.
TabTune could already **measure** that (`evaluate_calibration` reports ECE/MCE/Brier);
`tabtune.uncertainty` adds the two standard **fixes**.

Both wrappers consume only `predict_proba` / `predict`, so they work for every bundled
model, for distilled students, for ensembles, and for plain scikit-learn estimators. They
**compose around** the pipeline rather than mutating it, so `save()` / `load()` and
picklability are untouched.

```python
from tabtune.uncertainty import (
    ConformalClassifier,
    ConformalRegressor,
    Recalibrator,
    uncertainty_report,
    size_stratified_coverage,
)
```

---

## 1. The three-way split

Conformal prediction needs a calibration set the model has **never trained on**. A two-way
train/test split is not enough.

```python
from sklearn.model_selection import train_test_split

X_fit, X_rest, y_fit, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
```

!!! danger "In-context models: the training data *is* the support set"
    For an ICL model, passing the pipeline's own training frame as `X_cal` would void the
    guarantee silently. TabTune detects a re-used training frame by fingerprint and
    **raises** instead.

---

## 2. Prediction sets (classification)

Split conformal prediction is the engine. Score how *nonconforming* each calibration example
is, take the $\lceil (n+1)(1-\alpha) \rceil$-th smallest score as a threshold, and a test
point's prediction set is every label whose score clears that threshold.

```python
cp = ConformalClassifier(pipeline, method="lac", alpha=0.1).calibrate(X_cal, y_cal)

sets  = cp.predict_set(X_test)     # (n_test, n_classes) boolean matrix
sizes = cp.set_sizes(X_test)       # (n_test,) integer set sizes
cp.coverage(X_test, y_test)        # {'coverage': ..., 'avg_set_size': ...}

cp.q_hat_    # the calibrated threshold
cp.n_cal_    # calibration set size
cp.classes_  # class order matching the columns of `sets`
```

### 2.1 Constructor

```python
ConformalClassifier(
    pipeline,                 # anything with predict_proba
    method="lac",             # 'lac' | 'aps'
    alpha=0.1,                # miscoverage rate -> 90% target coverage
    *,
    randomized=False,         # randomized tie-breaking (APS)
    random_state=None,
)
```

| `method` | Score | Behaviour |
|---|---|---|
| `'lac'` | Least ambiguous set-valued classifier | Smallest average sets; can emit **empty** sets on confident rows |
| `'aps'` | Adaptive prediction sets | Sets grow on ambiguous rows, shrink on easy ones; never empty |

```python
aps = ConformalClassifier(pipeline, method="aps", alpha=0.1).calibrate(X_cal, y_cal)
sizes = aps.set_sizes(X_test)
# mean set size on confident rows (p_max > 0.9): 1.02
# mean set size on ambiguous rows (p_max < 0.6): 2.41
```

---

## 3. What the guarantee actually is

**Marginal** coverage under exchangeability:

$$P\big(y \in C(x)\big) \ge 1 - \alpha$$

on average over exchangeable draws of the calibration and test data. It is **not**:

- **conditional** coverage — no distribution-free method can promise 90% on every slice of
  the input space;
- **robust to distribution shift** — coverage degrades when the test distribution moves.

The report's **size-stratified coverage score (SSCS)** exists precisely to show how far
conditional coverage falls short of the marginal number: it is the *worst* coverage over
groups of equal set size.

```python
from tabtune.uncertainty import size_stratified_coverage

size_stratified_coverage(set_sizes, covered, min_stratum=...)
```

!!! note "An SSCS of 0.000 is not a bug"
    LAC produces **empty** sets for points the model is very confident about, and an empty
    set covers nothing by definition. The marginal guarantee still holds — those rows are
    paid for by the rest — but a whole stratum sits at zero coverage, and no marginal number
    can show you that. Use `'aps'` if you need non-empty sets.

---

## 4. Recalibration

`Recalibrator` reshapes the probabilities and exposes the same `predict` / `predict_proba` /
`classes_` surface, so it can be dropped anywhere the pipeline was used — including as the
input to a `ConformalClassifier`.

```python
recal = Recalibrator(pipeline, method="temperature").fit(X_cal1, y_cal1)
recal.temperature_          # the fitted scalar
recal.predict_proba(X_test)
```

Recalibrate on one split, conformalize on a **second**:

```python
X_cal1, X_cal2, y_cal1, y_cal2 = train_test_split(X_cal, y_cal, test_size=0.5)

recal   = Recalibrator(pipeline, method="temperature").fit(X_cal1, y_cal1)
stacked = ConformalClassifier(recal, alpha=0.1).calibrate(X_cal2, y_cal2)
```

---

## 5. Regression intervals

```python
cr = ConformalRegressor(pipeline, method="absolute", alpha=0.1).calibrate(X_cal, y_cal)
lo, hi = cr.predict_interval(X_test)
cr.coverage(X_test, y_test)
```

| `method` | Requires | Interval width |
|---|---|---|
| `'absolute'` | `predict` only — works for **every** model | Constant across rows |
| `'cqr'` | `predict_quantiles` — in TabTune, the TabPFN regressor family | Adapts per row |

---

## 6. The one-call report

```python
report = uncertainty_report(
    pipeline, X_test, y_test,
    X_cal=X_cal, y_cal=y_cal,
    alpha=0.1, n_bins=15, method="lac",
)
```

Also available as a pipeline method:

```python
pipeline.uncertainty_report(X_test, y_test, X_cal=X_cal, y_cal=y_cal)
```

| Key | Meaning |
|---|---|
| `ece` | Expected calibration error |
| `mce` | Maximum calibration error |
| `brier` | Brier score |
| `coverage` | Empirical marginal coverage of the prediction sets |
| `avg_set_size` | Mean set size (classification) / mean width (regression) |
| `sscs` | Size-stratified coverage score — the worst stratum |

Omitting `X_cal` / `y_cal` gives the calibration metrics only, with no conformal section.

---

## 7. End-to-end

```python
from tabtune import TabularPipeline
from tabtune.uncertainty import ConformalClassifier, Recalibrator

pipe = TabularPipeline("TabICLv2", task_type="classification", cache="memory")
pipe.fit(X_fit, y_fit)

# 1. Diagnose
print(pipe.uncertainty_report(X_test, y_test, X_cal=X_cal, y_cal=y_cal))

# 2. Fix the probabilities
pipe = Recalibrator(pipe, method="temperature").fit(X_cal1, y_cal1)

# 3. Wrap in a guarantee
cp = ConformalClassifier(pipe, method="aps", alpha=0.1).calibrate(X_cal2, y_cal2)
sets = cp.predict_set(X_test)
```

---

## See Also

- [Shift-Aware Evaluation](shift-evaluation.md) — coverage degrades under shift; measure the gap
- [Ensembling](ensembling.md) — `random_init` gives epistemic uncertainty
- Runnable example: `examples/18_uncertainty.py`
