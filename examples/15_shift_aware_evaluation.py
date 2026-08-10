"""Shift-aware evaluation: measuring the gap an IID score hides.

An IID cross-validation score answers "how well does this model fit data drawn
like the training data?" A deployed model faces a different question: "how well
does it hold up on data from next quarter, or from a cohort it has never seen?"

This example builds a dataset with a deliberate temporal drift and shows the
gap between the two answers. It uses scikit-learn estimators so it runs in
seconds without downloading any foundation-model weights; swap the factory for
``lambda: TabularPipeline("TabICLv2", cache="memory")`` to run it for real.

Run:
    python examples/14_shift_aware_evaluation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tabtune.evaluation import (
    GroupedSplit,
    ShiftEvaluator,
    StratifiedGroupedSplit,
    TemporalSplit,
    shift_gap,
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def make_drifting_data(n: int = 1_200, seed: int = 0):
    """Build data whose decision boundary rotates over time.

    Early rows are separable on ``f0``; later rows increasingly depend on
    ``f1``. A model validated on a random split sees both regimes in training
    and scores well; a model validated temporally does not.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n)
    drift = time / n  # 0 -> 1

    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    noise = rng.normal(scale=0.35, size=n)

    logit = (1 - drift) * 2.2 * f0 + drift * 2.2 * f1 + noise
    y = (logit > 0).astype(int)

    frame = pd.DataFrame(
        {
            "f0": f0,
            "f1": f1,
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "application_date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "region": rng.choice([f"r{i}" for i in range(8)], size=n),
        }
    )
    return frame, pd.Series(y, name="default")


X, y = make_drifting_data()


# --------------------------------------------------------------------------
section("1. Splitters enforce their own invariants")

temporal = TemporalSplit(n_splits=4, time_col="application_date", gap=24)
dates = X["application_date"].to_numpy()
print("TemporalSplit - training data is always strictly older than test data:")
for i, (train_idx, test_idx) in enumerate(temporal.split(X)):
    assert dates[train_idx].max() < dates[test_idx].min()
    print(
        f"  fold {i}: train {len(train_idx):>4} rows up to {pd.Timestamp(dates[train_idx].max()).date()}"
        f" | test {len(test_idx):>4} rows from {pd.Timestamp(dates[test_idx].min()).date()}"
    )

grouped = GroupedSplit(n_splits=4, group_col="region")
regions = X["region"].to_numpy()
print("\nGroupedSplit - no region appears on both sides of a fold:")
for i, (train_idx, test_idx) in enumerate(grouped.split(X)):
    held_out = sorted(set(regions[test_idx]))
    assert set(regions[train_idx]).isdisjoint(held_out)
    print(f"  fold {i}: held out {held_out}")


# --------------------------------------------------------------------------
section("2. The shift gap")

evaluator = ShiftEvaluator(
    splits={
        "temporal": TemporalSplit(4, time_col="application_date", gap=24),
        "grouped": GroupedSplit(4, group_col="region"),
        "grouped_stratified": StratifiedGroupedSplit(4, group_col="region"),
    },
    task_type="classification",
    n_splits=4,
)

# A factory, not a fitted model: each fold needs an unfitted estimator, and
# reusing one instance would leak the previous fold's state.
report = evaluator.run(
    lambda: LogisticRegression(max_iter=1_000),
    X,
    y,
    model_name="LogisticRegression",
    drop_split_columns=["application_date", "region"],
)

print(report)
print("\nNegative gap means worse under shift, whichever way the metric runs.")


# --------------------------------------------------------------------------
section("3. Per-split detail")

summary = report.summary_frame()
columns = [c for c in ("split", "folds", "accuracy", "roc_auc_score", "ece", "shift_gap") if c in summary]
print(summary[columns].round(4).to_string(index=False))

print("\nThe gap is available for any computed metric, not just the primary one:")
for metric in ("accuracy", "roc_auc_score", "ece"):
    gaps = report.gap_for(metric)
    rendered = ", ".join(f"{k}={v:+.4f}" for k, v in gaps.items())
    print(f"  {metric:<16} {rendered}")


# --------------------------------------------------------------------------
section("4. Comparing models under shift")

print(f"{'model':<24} {'iid':>8} {'temporal':>10} {'gap':>9}")
print("-" * 54)
for name, factory in [
    ("LogisticRegression", lambda: LogisticRegression(max_iter=1_000)),
    ("RandomForest", lambda: RandomForestClassifier(n_estimators=120, random_state=0)),
]:
    result = ShiftEvaluator(
        splits={"temporal": TemporalSplit(4, time_col="application_date")},
        task_type="classification",
        n_splits=4,
    ).run(factory, X, y, model_name=name, drop_split_columns=["application_date", "region"])

    iid = result.mean_metrics("iid")
    shifted = result.mean_metrics("temporal")
    gap = shift_gap(iid, shifted, "roc_auc_score")
    print(
        f"{name:<24} {iid['roc_auc_score']:>8.4f} {shifted['roc_auc_score']:>10.4f} {gap:>+9.4f}"
    )

print(
    "\nThe model that wins on the IID split is not necessarily the one that\n"
    "holds up under drift. That is the whole point of measuring the gap."
)


# --------------------------------------------------------------------------
section("5. Machine-readable output")

payload = report.to_dict()
print(f"  baseline:       {payload['baseline']}")
print(f"  primary metric: {payload['primary_metric']}")
print(f"  shift_gap:      {payload['shift_gap']}")
print(f"  failed folds:   {payload['n_failures']}")
print("\nUse report.to_frame() for a tidy per-fold DataFrame, or to_dict() for")
print("model cards and CI artefacts.")
