"""Uncertainty quantification: coverage you can promise, not just probabilities.

A tabular foundation model's ``predict_proba`` is an in-context softmax that was
never calibrated to your dataset. This example shows the two model-agnostic
fixes ``tabtune.uncertainty`` ships in 0.2.0 — split conformal prediction
(sets/intervals with a distribution-free marginal coverage guarantee) and
post-hoc recalibration — and the honest failure mode marginal coverage hides.

It uses scikit-learn estimators so it runs in seconds without downloading any
foundation-model weights; every wrapper here consumes only
``predict_proba``/``predict``, so swap the estimator for
``TabularPipeline("TabICLv2").fit(...)`` to run it for real.

Run:
    python examples/18_uncertainty.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from tabtune.uncertainty import (
    ConformalClassifier,
    ConformalRegressor,
    Recalibrator,
    uncertainty_report,
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


rng = np.random.default_rng(0)

# A 3-way split is the minimum for conformal work: the calibration rows must be
# disjoint from training, or the guarantee is void.
X, y = make_classification(
    n_samples=3000, n_features=12, n_informative=6, n_classes=3,
    n_clusters_per_class=1, class_sep=0.8, random_state=0,
)
X_fit, X_rest, y_fit, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0)
model.fit(X_fit, y_fit)


# --------------------------------------------------------------------------
section("1. Prediction sets with a 90% guarantee (LAC)")

cp = ConformalClassifier(model, method="lac", alpha=0.1).calibrate(X_cal, y_cal)
sets = cp.predict_set(X_test)
sizes = cp.set_sizes(X_test)
measured = cp.coverage(X_test, y_test)

print(f"threshold q_hat = {cp.q_hat_:.4f}  (from n_cal = {cp.n_cal_})")
print(f"empirical coverage = {measured['coverage']:.3f}  (target >= 0.90)")
print(f"avg set size       = {measured['avg_set_size']:.2f}  of {sets.shape[1]} classes")
print("set-size histogram :", dict(zip(*np.unique(sizes, return_counts=True), strict=True)))
print("\nA set is the honest answer to 'which labels can you not rule out at 90%?'.")


# --------------------------------------------------------------------------
section("2. Adaptive sets (APS): hard rows get bigger sets")

aps = ConformalClassifier(model, method="aps", alpha=0.1).calibrate(X_cal, y_cal)
aps_sizes = aps.set_sizes(X_test)
confidence = model.predict_proba(X_test).max(axis=1)
easy, hard = confidence > 0.9, confidence < 0.6
print(f"mean set size on confident rows (p_max > 0.9): {aps_sizes[easy].mean():.2f}")
print(f"mean set size on ambiguous rows (p_max < 0.6): {aps_sizes[hard].mean():.2f}")


# --------------------------------------------------------------------------
section("3. What marginal coverage hides: the size-stratified score")

report = uncertainty_report(model, X_test, y_test, X_cal=X_cal, y_cal=y_cal, alpha=0.1)
for key in ("ece", "mce", "brier", "coverage", "avg_set_size", "sscs"):
    print(f"  {key:<12} {report[key]:.4f}")
print(
    "\nsscs is the WORST coverage over groups of equal set size. A marginally\n"
    "valid predictor can still fail a subgroup badly - that gap is exactly\n"
    "what published TFM benchmarks flag, and why the report shows both numbers."
)

# Where that number comes from. If sscs is 0.000 here, it is not a bug: LAC
# produces EMPTY sets for points the model is confident about, and an empty set
# covers nothing by definition. The marginal guarantee still holds - those rows
# are paid for by the rest - but a whole stratum is at zero coverage, and no
# marginal number can show you that.
covered = sets[np.arange(len(y_test)), cp.classes_.searchsorted(y_test)]
print("\nper-stratum breakdown (this is what sscs takes the minimum of):")
for size in np.unique(sizes):
    group = sizes == size
    print(f"  set size {size}: n={group.sum():4d}  coverage={covered[group].mean():.3f}")


# --------------------------------------------------------------------------
section("4. Recalibration, then conformal on top")

# An intentionally overconfident model: too few iterations, big capacity.
overconfident = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=60, random_state=0)
overconfident.fit(X_fit, y_fit)

X_cal1, X_cal2, y_cal1, y_cal2 = train_test_split(X_cal, y_cal, test_size=0.5, random_state=0)
before = uncertainty_report(overconfident, X_test, y_test)
recal = Recalibrator(overconfident, method="temperature").fit(X_cal1, y_cal1)
after = uncertainty_report(recal, X_test, y_test)
print(f"ECE before temperature scaling: {before['ece']:.4f}")
print(f"ECE after  temperature scaling: {after['ece']:.4f}   (T = {recal.temperature_:.3f})")

stacked = ConformalClassifier(recal, alpha=0.1).calibrate(X_cal2, y_cal2)
print(f"conformal-on-recalibrated coverage: {stacked.coverage(X_test, y_test)['coverage']:.3f}")
print("\nRecalibrate on one split, conformalize on a second - never the same rows.")


# --------------------------------------------------------------------------
section("5. Regression intervals")

Xr, yr = make_regression(n_samples=2400, n_features=8, noise=12.0, random_state=0)
Xr_fit, Xr_rest, yr_fit, yr_rest = train_test_split(Xr, yr, test_size=0.5, random_state=0)
Xr_cal, Xr_test, yr_cal, yr_test = train_test_split(Xr_rest, yr_rest, test_size=0.5, random_state=0)

reg = GradientBoostingRegressor(random_state=0).fit(Xr_fit, yr_fit)
cr = ConformalRegressor(reg, method="absolute", alpha=0.1).calibrate(Xr_cal, yr_cal)
lo, hi = cr.predict_interval(Xr_test)
inside = float(np.mean((yr_test >= lo) & (yr_test <= hi)))
print(f"interval coverage = {inside:.3f}  (target >= 0.90)")
print(f"mean width        = {float(np.mean(hi - lo)):.1f}  (constant: the 'absolute' score)")
print("\n'cqr' adapts widths per-row but needs predict_quantiles - in TabTune,")
print("the TabPFN regressor family. 'absolute' works for every model.")


# --------------------------------------------------------------------------
section("6. The same API on a TabTune pipeline")

print("""Everything above consumed only predict_proba/predict, so with weights
available the swap is one line:

    pipe = TabularPipeline("TabICLv2", cache="memory").fit(X_fit, y_fit)
    pipe.uncertainty_report(X_test, y_test, X_cal=X_cal, y_cal=y_cal)
    ConformalClassifier(pipe, alpha=0.1).calibrate(X_cal, y_cal).predict_set(X_test)

One honest caveat for in-context models: the "training data" IS the support
set. The wrapper detects a re-used training frame by fingerprint and raises,
because a guarantee calibrated on support rows would be void.""")

# A cheap sanity check that this example's claims hold: LogisticRegression via
# the identical code path.
lr = LogisticRegression(max_iter=500).fit(X_fit, y_fit)
lr_cov = ConformalClassifier(lr, alpha=0.1).calibrate(X_cal, y_cal).coverage(X_test, y_test)
print(f"\nLogisticRegression through the same path -> coverage {lr_cov['coverage']:.3f}")
