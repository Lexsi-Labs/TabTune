"""Smoke tests for the ``tabtune.causal`` module.

These cover the lightweight, estimator-fit-free surface of the causal module
(registry wiring, constructor validation, and sklearn-adapter typing). They are
skipped automatically when the optional causal dependencies are not installed
(``pip install tabtune[causal]``), so they never break a core-only CI run but
do exercise the public API once the extra is present.
"""
import pytest

# The causal package imports networkx at import time; skip the whole module
# when the optional causal dependencies are absent.
pytest.importorskip("networkx")

from tabtune.causal import (  # noqa: E402
    CausalAnalysis,
    ESTIMATOR_REGISTRY,
    _TabTuneSklearnAdapter,
    as_sklearn,
)


def test_estimator_registry_complete():
    expected = {
        "dml",
        "s_learner",
        "t_learner",
        "x_learner",
        "r_learner",
        "causal_forest",
    }
    assert expected.issubset(set(ESTIMATOR_REGISTRY))


def test_causal_analysis_rejects_unknown_estimator():
    with pytest.raises(ValueError):
        CausalAnalysis(
            model_name="TabPFNv26",
            task_type="regression",
            treatment="t",
            outcome="y",
            confounders=["x1", "x2"],
            estimator="not_a_real_estimator",
        )


def test_causal_analysis_constructs_and_round_trips_params():
    ca = CausalAnalysis(
        model_name="TabPFNv26",
        task_type="regression",
        treatment="t",
        outcome="y",
        confounders=["x1", "x2"],
        sensitive=["s"],
        estimator="dml",
        verbose=False,
    )
    assert ca.estimator == "dml"
    assert ca.confounders == ["x1", "x2"]
    assert ca.sensitive == ["s"]

    params = ca.get_params()
    assert params["estimator"] == "dml"
    assert params["treatment"] == "t"
    assert params["confounders"] == ["x1", "x2"]


@pytest.mark.parametrize(
    "task_type,expected_type",
    [("classification", "classifier"), ("regression", "regressor")],
)
def test_adapter_estimator_type(task_type, expected_type):
    adapter = as_sklearn(model_name="TabPFNv26", task_type=task_type)
    assert isinstance(adapter, _TabTuneSklearnAdapter)
    assert adapter._estimator_type == expected_type


def test_regression_adapter_has_no_predict_proba():
    adapter = as_sklearn(model_name="TabPFNv26", task_type="regression")
    # The regression adapter must not advertise predict_proba (it routes
    # through __getattr__, which raises AttributeError for regressors).
    assert not hasattr(adapter, "predict_proba")
