"""Task-aware metric defaults for tabular ensembles and greedy selection."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from tabtune.ensemble import GreedyEnsembleSelection, TabularEnsemble

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("constructor", [TabularEnsemble, GreedyEnsembleSelection])
@pytest.mark.parametrize(
    "task_type, expected", [("classification", "accuracy"), ("regression", "r2")]
)
@pytest.mark.parametrize("metric_kwargs", [{}, {"metric": None}])
def test_task_aware_defaults(constructor, task_type, expected, metric_kwargs):
    kwargs = {"models": [{"model_name": "linear"}]} if constructor is TabularEnsemble else {}
    ensemble = constructor(task_type=task_type, **kwargs, **metric_kwargs)
    assert ensemble.metric == expected


@pytest.mark.parametrize("constructor", [TabularEnsemble, GreedyEnsembleSelection])
@pytest.mark.parametrize(
    "task_type, metric", [("classification", "f1_score"), ("regression", "mae")]
)
def test_explicit_metric_is_preserved(constructor, task_type, metric):
    kwargs = {"models": [{"model_name": "linear"}]} if constructor is TabularEnsemble else {}
    ensemble = constructor(task_type=task_type, metric=metric, **kwargs)
    assert ensemble.metric == metric


def test_greedy_regression_default_fits_and_selects_best_model():
    y = np.array([1.0, 3.0, 5.0, 7.0])
    outputs = {"perfect": y.copy(), "constant": np.full_like(y, y.mean())}
    ensemble = GreedyEnsembleSelection(task_type="regression", ensemble_size=3)
    assert ensemble.fit(outputs, y) is ensemble
    assert ensemble.weights_ == {"perfect": 1.0, "constant": 0.0}
    np.testing.assert_allclose(ensemble.predict(outputs), y)


def test_greedy_classification_default_still_fits():
    y = np.array([0, 1, 0, 1])
    outputs = {"perfect": np.eye(2)[y], "wrong": np.eye(2)[1 - y]}
    ensemble = GreedyEnsembleSelection(ensemble_size=3)
    ensemble.fit(outputs, y)
    assert ensemble.metric == "accuracy"
    assert ensemble.weights_ == {"perfect": 1.0, "wrong": 0.0}
    np.testing.assert_array_equal(ensemble.predict(outputs), y)


@pytest.mark.parametrize("metric_kwargs", [{}, {"metric": None}, {"metric": "mse"}])
def test_tabular_regression_default_fit_and_predict(monkeypatch, metric_kwargs):
    """Run the real ensemble workflow with small, weight-free base estimators."""
    ensemble = TabularEnsemble(
        models=[{"model_name": "linear"}, {"model_name": "constant"}],
        task_type="regression",
        greedy_ensemble_size=3,
        verbose=False,
        **metric_kwargs,
    )
    monkeypatch.setattr(
        ensemble,
        "_build_pipeline",
        lambda config: LinearRegression() if config["model_name"] == "linear" else DummyRegressor(),
    )
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    assert ensemble.fit(X, y) is ensemble
    expected_metric = "mse" if metric_kwargs.get("metric") == "mse" else "r2"
    assert ensemble.strategy_.metric == expected_metric
    np.testing.assert_allclose(ensemble.predict(X), y)
    assert ensemble.ensemble_score_ == pytest.approx(1.0)


def test_tabular_classification_default_fit_and_predict(monkeypatch):
    ensemble = TabularEnsemble(
        models=[{"model_name": "tree"}], greedy_ensemble_size=3, verbose=False
    )
    monkeypatch.setattr(
        ensemble, "_build_pipeline", lambda config: DecisionTreeClassifier(random_state=42)
    )
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 10).astype(int)
    ensemble.fit(X[:], y, X_val=X, y_val=y)
    assert ensemble.strategy_.metric == "accuracy"
    np.testing.assert_array_equal(ensemble.predict(X), y)


def test_explicit_invalid_regression_metric_is_not_overridden():
    y = np.array([1.0, 3.0, 5.0, 7.0])
    ensemble = GreedyEnsembleSelection(task_type="regression", metric="accuracy")
    with pytest.raises(ValueError, match="Unknown metric 'accuracy'.*regression"):
        ensemble.fit({"model": y}, y)
