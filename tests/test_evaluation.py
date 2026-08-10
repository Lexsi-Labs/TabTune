"""Tests for shift-aware splits, shared metrics and shift-gap reporting.

The splitters carry correctness properties that are easy to get subtly wrong
and expensive to discover in production - a temporal split that leaks one row
of the future, or a grouped split that lets a patient appear on both sides,
produces optimistic scores that look fine until deployment. Those properties
are asserted directly rather than through end-to-end behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from tabtune.evaluation import (
    GroupedSplit,
    ShiftEvaluator,
    StratifiedGroupedSplit,
    TemporalSplit,
    calibration_metrics,
    classification_metrics,
    compute_metrics,
    expected_calibration_error,
    is_higher_better,
    primary_metric,
    regression_metrics,
    resolve_split,
    shift_gap,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def temporal_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "f1": rng.normal(size=40),
            "f2": rng.normal(size=40),
        }
    )


@pytest.fixture
def grouped_frame():
    return pd.DataFrame(
        {
            "patient": [f"p{i // 4}" for i in range(40)],
            "f1": np.arange(40.0),
        }
    )


# ------------------------------------------------------------- temporal split


def test_temporal_split_never_trains_on_the_future(temporal_frame):
    splitter = TemporalSplit(n_splits=4, time_col="date")
    dates = temporal_frame["date"].to_numpy()
    for train_idx, test_idx in splitter.split(temporal_frame):
        assert dates[train_idx].max() < dates[test_idx].min()


def test_temporal_split_training_window_expands(temporal_frame):
    sizes = [len(tr) for tr, _ in TemporalSplit(4, time_col="date").split(temporal_frame)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1


def test_temporal_split_respects_max_train_size(temporal_frame):
    splitter = TemporalSplit(4, time_col="date", max_train_size=10)
    assert all(len(tr) <= 10 for tr, _ in splitter.split(temporal_frame))


def test_temporal_split_gap_creates_a_hole(temporal_frame):
    dates = temporal_frame["date"].to_numpy()
    for train_idx, test_idx in TemporalSplit(3, time_col="date", gap=5).split(temporal_frame):
        gap_days = (dates[test_idx].min() - dates[train_idx].max()).astype("timedelta64[D]")
        assert gap_days.astype(int) > 1


def test_temporal_split_sorts_unordered_input():
    """A shuffled frame must still split chronologically."""
    frame = pd.DataFrame({"t": [5, 1, 4, 2, 3, 0], "f": range(6)})
    train_idx, test_idx = next(iter(TemporalSplit(2, time_col="t").split(frame)))
    assert frame["t"].to_numpy()[train_idx].max() < frame["t"].to_numpy()[test_idx].min()


def test_temporal_split_uses_row_order_without_a_time_column():
    frame = pd.DataFrame({"f": range(20)})
    folds = list(TemporalSplit(3).split(frame))
    assert len(folds) == 3
    assert folds[0][0].max() < folds[0][1].min()


def test_temporal_split_rejects_too_few_rows():
    with pytest.raises(ValueError, match="Not enough rows"):
        list(TemporalSplit(10).split(pd.DataFrame({"f": range(5)})))


def test_temporal_split_rejects_n_splits_below_two():
    with pytest.raises(ValueError, match="n_splits"):
        TemporalSplit(1)


def test_temporal_split_reports_its_shift_type():
    assert TemporalSplit().shift_type == "temporal"
    assert TemporalSplit(3).get_n_splits() == 3


# -------------------------------------------------------------- grouped split


def test_grouped_split_keeps_groups_disjoint(grouped_frame):
    groups = grouped_frame["patient"].to_numpy()
    for train_idx, test_idx in GroupedSplit(5, group_col="patient").split(grouped_frame):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_grouped_split_covers_every_row_exactly_once(grouped_frame):
    seen: list[int] = []
    for _, test_idx in GroupedSplit(5, group_col="patient").split(grouped_frame):
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(grouped_frame)))


def test_grouped_split_accepts_explicit_group_array(grouped_frame):
    groups = grouped_frame["patient"].to_numpy()
    folds = list(GroupedSplit(5).split(grouped_frame, groups=groups))
    assert len(folds) == 5


def test_grouped_split_rejects_too_few_groups():
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "f": range(4)})
    with pytest.raises(ValueError, match="distinct groups"):
        list(GroupedSplit(5, group_col="g").split(frame))


def test_grouped_split_reports_a_missing_column(grouped_frame):
    with pytest.raises(KeyError, match="not found"):
        list(GroupedSplit(3, group_col="nope").split(grouped_frame))


def test_grouped_split_requires_a_dataframe_for_column_lookup():
    with pytest.raises(TypeError, match="DataFrame"):
        list(GroupedSplit(3, group_col="g").split(np.zeros((10, 2))))


def test_grouped_split_shuffle_is_seeded(grouped_frame):
    a = [t.tolist() for _, t in GroupedSplit(5, group_col="patient", shuffle=True, random_state=1).split(grouped_frame)]
    b = [t.tolist() for _, t in GroupedSplit(5, group_col="patient", shuffle=True, random_state=1).split(grouped_frame)]
    assert a == b


def test_stratified_grouped_split_keeps_groups_disjoint_and_balances(grouped_frame):
    y = pd.Series([0] * 20 + [1] * 20)
    groups = grouped_frame["patient"].to_numpy()
    positives = []
    for train_idx, test_idx in StratifiedGroupedSplit(4, group_col="patient").split(
        grouped_frame, y
    ):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        positives.append(int(y.to_numpy()[test_idx].sum()))
    # Not exact stratification - groups are atomic - but no fold may be empty
    # of the positive class, which is the failure mode being guarded against.
    assert all(count > 0 for count in positives)


def test_stratified_grouped_split_falls_back_without_y(grouped_frame):
    folds = list(StratifiedGroupedSplit(4, group_col="patient").split(grouped_frame))
    assert len(folds) == 4


# ------------------------------------------------------------ split resolution


def test_resolve_split_defaults_to_stratified_kfold():
    from sklearn.model_selection import StratifiedKFold

    assert isinstance(resolve_split(None, task_type="classification"), StratifiedKFold)


def test_resolve_split_defaults_to_kfold_for_regression():
    from sklearn.model_selection import KFold

    assert isinstance(resolve_split(None, task_type="regression"), KFold)


def test_resolve_split_by_name():
    assert isinstance(resolve_split("temporal", n_splits=3), TemporalSplit)


def test_resolve_split_passes_instances_through():
    splitter = TemporalSplit(3)
    assert resolve_split(splitter) is splitter


def test_resolve_split_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown split"):
        resolve_split("diagonal")


def test_resolve_split_rejects_wrong_types():
    with pytest.raises(TypeError):
        resolve_split(42)


# ---------------------------------------------------------------- metrics


def test_classification_metrics_cover_the_documented_bundle():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9], [0.4, 0.6]])
    metrics = classification_metrics(y_true, y_pred, proba)
    for key in ("accuracy", "f1_score", "roc_auc_score", "mcc", "ece", "mce", "brier_score"):
        assert key in metrics


def test_regression_metrics_cover_the_documented_bundle():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    for key in ("r2_score", "rmse", "mae", "mse", "max_error"):
        assert key in metrics
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def test_metrics_return_nan_rather_than_raising_on_degenerate_input():
    """A single-class fold is normal in a sweep and must not abort it."""
    metrics = classification_metrics([0, 0, 0], [0, 0, 0], np.array([[1.0, 0.0]] * 3))
    assert metrics["accuracy"] == 1.0
    assert np.isnan(metrics["roc_auc_score"])


def test_perfect_calibration_scores_near_zero():
    rng = np.random.default_rng(0)
    proba_positive = rng.uniform(size=4000)
    y_true = (rng.uniform(size=4000) < proba_positive).astype(int)
    proba = np.column_stack([1 - proba_positive, proba_positive])
    ece, mce = expected_calibration_error(y_true, proba, n_bins=10)
    assert ece < 0.05


def test_overconfident_predictions_score_badly():
    y_true = np.array([0, 1] * 50)
    proba = np.tile([[0.99, 0.01]], (100, 1))  # always confidently class 0
    ece, _ = expected_calibration_error(y_true, proba)
    assert ece > 0.4


def test_calibration_handles_shape_mismatch_gracefully():
    metrics = calibration_metrics([0, 1, 2], np.array([[0.5, 0.5]] * 3))
    assert np.isnan(metrics["ece"])


def test_quantile_binning_is_supported():
    y_true = np.array([0, 1] * 50)
    proba = np.column_stack([np.linspace(0.01, 0.99, 100), np.linspace(0.99, 0.01, 100)])
    ece, _ = expected_calibration_error(y_true, proba, strategy="quantile")
    assert 0.0 <= ece <= 1.0


def test_compute_metrics_dispatches_on_task_type():
    assert "accuracy" in compute_metrics([0, 1], [0, 1], task_type="classification")
    assert "rmse" in compute_metrics([0.0, 1.0], [0.1, 0.9], task_type="regression")
    with pytest.raises(ValueError, match="task_type"):
        compute_metrics([0], [0], task_type="ranking")


def test_metric_direction_table():
    assert is_higher_better("accuracy") and is_higher_better("r2_score")
    assert not is_higher_better("rmse") and not is_higher_better("ece")
    assert primary_metric("classification") == "roc_auc_score"
    assert primary_metric("regression") == "r2_score"


# ----------------------------------------------------------- shift evaluation


def _classification_frame():
    X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=0)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    frame["t"] = np.arange(300)
    frame["g"] = [f"g{i % 10}" for i in range(300)]
    return frame, pd.Series(y)


@pytest.mark.integration
def test_shift_evaluator_reports_a_gap_against_the_iid_baseline():
    X, y = _classification_frame()
    evaluator = ShiftEvaluator(
        splits={"temporal": TemporalSplit(3, time_col="t")},
        task_type="classification",
        n_splits=3,
    )
    report = evaluator.run(
        lambda: LogisticRegression(max_iter=500),
        X,
        y,
        model_name="logreg",
        drop_split_columns=["t", "g"],
    )
    assert set(report.split_names()) == {"iid", "temporal"}
    assert report.baseline == "iid"
    assert "temporal" in report.shift_gap
    assert not report.failures
    assert 0.0 <= report.mean_metrics("iid")["accuracy"] <= 1.0


@pytest.mark.integration
def test_shift_report_frames_and_dict_are_well_formed():
    X, y = _classification_frame()
    evaluator = ShiftEvaluator(
        splits={"grouped": GroupedSplit(3, group_col="g")}, n_splits=3
    )
    report = evaluator.run(
        lambda: LogisticRegression(max_iter=500), X, y, drop_split_columns=["t", "g"]
    )

    tidy = report.to_frame()
    assert len(tidy) == 6  # two schemes x three folds
    assert {"split", "fold", "status"} <= set(tidy.columns)

    summary = report.summary_frame()
    assert set(summary["split"]) == {"iid", "grouped"}
    assert "shift_gap" in summary.columns

    payload = report.to_dict()
    assert payload["baseline"] == "iid"
    assert "temporal" not in payload["splits"]
    assert isinstance(str(report), str)


@pytest.mark.integration
def test_shift_evaluator_works_for_regression():
    X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame["t"] = np.arange(200)
    evaluator = ShiftEvaluator(
        splits={"temporal": TemporalSplit(3, time_col="t")},
        task_type="regression",
        n_splits=3,
    )
    report = evaluator.run(
        LinearRegression, frame, pd.Series(y), drop_split_columns=["t"]
    )
    assert "r2_score" in report.mean_metrics("iid")


def test_shift_evaluator_records_failures_instead_of_aborting():
    X, y = _classification_frame()

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("checkpoint missing")

    evaluator = ShiftEvaluator(task_type="classification", n_splits=3)
    report = evaluator.run(Broken, X, y, drop_split_columns=["t", "g"])
    assert len(report.failures) == 3
    assert "checkpoint missing" in report.failures[0].error


def test_shift_evaluator_can_reraise():
    X, y = _classification_frame()

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("boom")

    evaluator = ShiftEvaluator(task_type="classification", n_splits=3, error_score="raise")
    with pytest.raises(RuntimeError, match="boom"):
        evaluator.run(Broken, X, y, drop_split_columns=["t", "g"])


def test_shift_evaluator_rejects_bad_arguments():
    with pytest.raises(ValueError, match="task_type"):
        ShiftEvaluator(task_type="ranking")
    with pytest.raises(ValueError, match="error_score"):
        ShiftEvaluator(error_score="explode")


def test_shift_evaluator_accepts_a_list_of_split_names():
    evaluator = ShiftEvaluator(["temporal"], n_splits=3)
    assert set(evaluator.splits) == {"temporal", "iid"}


@pytest.mark.parametrize(
    "metric,baseline,shifted,expected",
    [
        ("accuracy", 0.90, 0.85, -0.05),  # higher-is-better: drop is negative
        ("rmse", 1.00, 1.20, -0.20),  # lower-is-better: rise is also negative
        ("rmse", 1.00, 0.80, 0.20),  # improvement is positive either way
    ],
)
def test_shift_gap_sign_convention(metric, baseline, shifted, expected):
    """Negative always means "worse under shift", whatever the metric's direction."""
    assert shift_gap({metric: baseline}, {metric: shifted}, metric) == pytest.approx(expected)


def test_shift_gap_is_nan_when_a_value_is_missing():
    assert np.isnan(shift_gap({}, {"accuracy": 0.9}, "accuracy"))
