"""
Test TabTune Ensemble Module
==============================
Unit tests for all 6 ensemble strategies (no GPU) + integration tests (GPU).

Run:
    pytest tests/test_ensemble.py -v

    # Strategy unit tests only (no GPU):
    pytest tests/test_ensemble.py -k "TestStrategies or TestCascade or TestRandomInit" -v

    # Full integration (GPU):
    pytest tests/test_ensemble.py::TestFullEnsemble -v

    # Smoke test:
    python tests/test_ensemble.py
"""

import pytest
import numpy as np
import pandas as pd


# ======================================================================
# Mock data generators
# ======================================================================

def _make_mock_cls_outputs(n_samples=100, n_classes=3, n_models=4, seed=42):
    """Generate mock probability outputs from multiple classification models.

    Returns
    -------
    outputs : dict[str, ndarray]
        ``{model_name: (n_samples, n_classes) proba array}``
    y : ndarray of shape ``(n_samples,)``
    """
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, n_samples)
    outputs = {}
    for i in range(n_models):
        noise = 0.2 + 0.1 * i
        probas = rng.dirichlet(np.ones(n_classes) * (1 / noise), n_samples)
        # Correlate with true labels
        for j in range(n_samples):
            probas[j, y[j]] += 1.0 - noise
        probas = probas / probas.sum(axis=1, keepdims=True)
        outputs[f"model_{i}"] = probas
    return outputs, y


def _make_mock_reg_outputs(n_samples=100, n_models=4, seed=42):
    """Generate mock regression outputs.

    Returns
    -------
    outputs : dict[str, ndarray]
        ``{model_name: (n_samples,) prediction array}``
    y : ndarray of shape ``(n_samples,)``
    """
    rng = np.random.RandomState(seed)
    y = rng.randn(n_samples) * 10 + 50
    outputs = {}
    for i in range(n_models):
        noise = 1.0 + 0.5 * i
        outputs[f"model_{i}"] = y + rng.randn(n_samples) * noise
    return outputs, y


def _make_mock_seed_outputs(n_samples=100, n_classes=3, n_models=3, n_seeds=3, seed=42):
    """Generate mock per-seed outputs for RandomInitEnsemble testing.

    Returns
    -------
    per_seed : dict[str, list[ndarray]]
    y : ndarray
    """
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, n_samples)
    per_seed = {}
    for i in range(n_models):
        seed_list = []
        for s in range(n_seeds):
            noise = 0.3 + 0.1 * i + 0.05 * s
            probas = rng.dirichlet(np.ones(n_classes) * (1 / noise), n_samples)
            for j in range(n_samples):
                probas[j, y[j]] += 1.0 - noise
            probas = probas / probas.sum(axis=1, keepdims=True)
            seed_list.append(probas)
        per_seed[f"model_{i}"] = seed_list
    return per_seed, y


# ======================================================================
# Strategy 1-4 unit tests (no GPU, no TabTune models)
# ======================================================================

class TestStrategies:
    """Test ensemble strategies 1-4 with mock data (no GPU needed)."""

    # --- Weighted Averaging ---

    def test_weighted_averaging_cls_performance(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        outputs, y = _make_mock_cls_outputs()
        wa = WeightedAveraging(task_type="classification", weight_scheme="performance")
        wa.fit(outputs, y, metric="accuracy")
        assert wa.weights_ is not None
        assert len(wa.weights_) == 4
        assert abs(sum(wa.weights_.values()) - 1.0) < 1e-6
        preds = wa.predict(outputs)
        assert len(preds) == len(y)
        probas = wa.predict_proba(outputs)
        assert probas.shape == (100, 3)

    def test_weighted_averaging_cls_uniform(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        outputs, y = _make_mock_cls_outputs()
        wa = WeightedAveraging(task_type="classification", weight_scheme="uniform")
        wa.fit(outputs, y, metric="accuracy")
        n = len(outputs)
        for w in wa.weights_.values():
            assert w == pytest.approx(1.0 / n)

    def test_weighted_averaging_cls_inverse_error(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        outputs, y = _make_mock_cls_outputs()
        wa = WeightedAveraging(task_type="classification", weight_scheme="inverse_error")
        wa.fit(outputs, y, metric="accuracy")
        assert wa.weights_ is not None
        assert abs(sum(wa.weights_.values()) - 1.0) < 1e-6

    def test_weighted_averaging_reg(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        outputs, y = _make_mock_reg_outputs()
        wa = WeightedAveraging(task_type="regression")
        wa.fit(outputs, y, metric="mse")
        preds = wa.predict(outputs)
        assert len(preds) == len(y)

    def test_weighted_averaging_manual_weights(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        outputs, y = _make_mock_cls_outputs()
        wa = WeightedAveraging(task_type="classification")
        wa.fit(
            outputs, y,
            weights={"model_0": 3, "model_1": 1, "model_2": 1, "model_3": 1},
        )
        assert wa.weights_["model_0"] == pytest.approx(0.5)

    def test_weighted_averaging_invalid_task_type(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        with pytest.raises(ValueError, match="task_type"):
            WeightedAveraging(task_type="invalid")

    def test_weighted_averaging_invalid_weight_scheme(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        with pytest.raises(ValueError, match="weight_scheme"):
            WeightedAveraging(task_type="classification", weight_scheme="bogus")

    def test_weighted_averaging_predict_before_fit(self):
        from tabtune.ensemble.strategies import WeightedAveraging
        wa = WeightedAveraging(task_type="classification")
        outputs, _ = _make_mock_cls_outputs()
        with pytest.raises(RuntimeError, match="not fitted"):
            wa.predict(outputs)

    # --- Greedy Ensemble Selection ---

    def test_greedy_selection_cls(self):
        from tabtune.ensemble.strategies import GreedyEnsembleSelection
        outputs, y = _make_mock_cls_outputs()
        gs = GreedyEnsembleSelection(ensemble_size=20, task_type="classification")
        gs.fit(outputs, y)
        assert gs.weights_ is not None
        assert len(gs.selection_history_) == 20
        preds = gs.predict(outputs)
        assert len(preds) == len(y)

    def test_greedy_selection_reg(self):
        from tabtune.ensemble.strategies import GreedyEnsembleSelection
        outputs, y = _make_mock_reg_outputs()
        gs = GreedyEnsembleSelection(
            ensemble_size=10, task_type="regression", metric="mse"
        )
        gs.fit(outputs, y)
        preds = gs.predict(outputs)
        assert len(preds) == len(y)

    def test_greedy_improves_over_best_single(self):
        """Greedy selection should be >= best single model."""
        from tabtune.ensemble.strategies import GreedyEnsembleSelection
        from sklearn.metrics import accuracy_score
        outputs, y = _make_mock_cls_outputs(n_samples=500, n_models=6)
        gs = GreedyEnsembleSelection(
            ensemble_size=30, task_type="classification"
        )
        gs.fit(outputs, y)

        best_single = max(
            accuracy_score(y, np.argmax(v, axis=1)) for v in outputs.values()
        )
        ensemble_score = accuracy_score(y, gs.predict(outputs))
        assert ensemble_score >= best_single - 0.02

    def test_greedy_without_replacement(self):
        from tabtune.ensemble.strategies import GreedyEnsembleSelection
        outputs, y = _make_mock_cls_outputs(n_models=4)
        gs = GreedyEnsembleSelection(
            ensemble_size=10, task_type="classification", with_replacement=False
        )
        gs.fit(outputs, y)
        # Without replacement, max selections = n_models
        assert len(gs.selection_history_) <= 4

    def test_greedy_invalid_ensemble_size(self):
        from tabtune.ensemble.strategies import GreedyEnsembleSelection
        with pytest.raises(ValueError, match="ensemble_size"):
            GreedyEnsembleSelection(ensemble_size=0)

    # --- Stacking ---

    def test_stacking_cls_lr(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        outputs, y = _make_mock_cls_outputs()
        se = StackingEnsemble(meta_learner="lr", task_type="classification")
        se.fit(outputs, y)
        preds = se.predict(outputs)
        assert len(preds) == len(y)
        probas = se.predict_proba(outputs)
        assert probas.shape[0] == len(y)

    def test_stacking_cls_gbdt(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        se = StackingEnsemble(meta_learner="gbdt", task_type="classification")
        outputs, y = _make_mock_cls_outputs()
        se.fit(outputs, y)
        preds = se.predict(outputs)
        assert len(preds) == len(y)

    def test_stacking_cls_mlp(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        se = StackingEnsemble(meta_learner="mlp", task_type="classification")
        outputs, y = _make_mock_cls_outputs()
        se.fit(outputs, y)
        preds = se.predict(outputs)
        assert len(preds) == len(y)

    def test_stacking_reg(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        outputs, y = _make_mock_reg_outputs()
        se = StackingEnsemble(meta_learner="lr", task_type="regression")
        se.fit(outputs, y)
        preds = se.predict(outputs)
        assert len(preds) == len(y)

    def test_stacking_with_original_features(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        outputs, y = _make_mock_cls_outputs()
        X_orig = np.random.randn(100, 5)
        se = StackingEnsemble(
            meta_learner="lr", task_type="classification",
            use_original_features=True,
        )
        se.fit(outputs, y, X_original=X_orig)
        preds = se.predict(outputs, X_original=X_orig)
        assert len(preds) == len(y)

    def test_stacking_invalid_meta_learner(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        with pytest.raises(ValueError, match="meta_learner"):
            StackingEnsemble(meta_learner="xgb")

    def test_stacking_predict_before_fit(self):
        from tabtune.ensemble.strategies import StackingEnsemble
        se = StackingEnsemble(meta_learner="lr", task_type="classification")
        outputs, _ = _make_mock_cls_outputs()
        with pytest.raises(RuntimeError, match="not fitted"):
            se.predict(outputs)

    # --- Temperature-Scaled Blending ---

    def test_temperature_scaled_cls(self):
        from tabtune.ensemble.strategies import TemperatureScaledBlending
        outputs, y = _make_mock_cls_outputs()
        ts = TemperatureScaledBlending(task_type="classification")
        ts.fit(outputs, y)
        assert len(ts.temperatures_) == 4
        for t in ts.temperatures_.values():
            assert 0.01 <= t <= 20.0
        preds = ts.predict(outputs)
        assert len(preds) == len(y)
        probas = ts.predict_proba(outputs)
        assert probas.shape == (100, 3)

    def test_temperature_scaled_reg_fallback(self):
        from tabtune.ensemble.strategies import TemperatureScaledBlending
        outputs, y = _make_mock_reg_outputs()
        ts = TemperatureScaledBlending(task_type="regression")
        ts.fit(outputs, y)
        preds = ts.predict(outputs)
        assert len(preds) == len(y)

    def test_temperature_scaled_predict_before_fit(self):
        from tabtune.ensemble.strategies import TemperatureScaledBlending
        ts = TemperatureScaledBlending(task_type="classification")
        outputs, _ = _make_mock_cls_outputs()
        with pytest.raises(RuntimeError, match="not fitted"):
            ts.predict(outputs)


# ======================================================================
# Strategy 5: Cascade Stacking tests
# ======================================================================

class TestCascadeStacking:
    """Test CascadeStackingEnsemble with mock data (no GPU needed)."""

    def test_cascade_basic(self):
        from tabtune.ensemble.strategies import CascadeStackingEnsemble
        outputs_l1, y = _make_mock_cls_outputs(n_models=3)
        outputs_l2, _ = _make_mock_cls_outputs(n_models=3, seed=99)

        cs = CascadeStackingEnsemble(
            n_levels=2, ges_size=20, task_type="classification"
        )
        cs.fit([outputs_l1, outputs_l2], y)
        assert cs.weights_ is not None
        assert len(cs.weights_) == 6  # 3 models x 2 levels
        assert len(cs.level_info_) == 2

    def test_cascade_predict(self):
        from tabtune.ensemble.strategies import CascadeStackingEnsemble
        outputs_l1, y = _make_mock_cls_outputs(n_models=3)
        outputs_l2, _ = _make_mock_cls_outputs(n_models=3, seed=99)

        cs = CascadeStackingEnsemble(
            n_levels=2, ges_size=20, task_type="classification"
        )
        cs.fit([outputs_l1, outputs_l2], y)

        # Build flat stacker outputs for predict
        flat = {}
        for name, arr in outputs_l1.items():
            flat[f"L1_{name}"] = arr
        for name, arr in outputs_l2.items():
            flat[f"L2_{name}"] = arr

        preds = cs.predict(flat)
        assert len(preds) == len(y)
        probas = cs.predict_proba(flat)
        assert probas.shape == (100, 3)

    def test_cascade_single_level(self):
        from tabtune.ensemble.strategies import CascadeStackingEnsemble
        outputs, y = _make_mock_cls_outputs(n_models=4)
        cs = CascadeStackingEnsemble(n_levels=1, ges_size=10)
        cs.fit([outputs], y)
        assert len(cs.weights_) == 4

    def test_cascade_empty_input(self):
        from tabtune.ensemble.strategies import CascadeStackingEnsemble
        cs = CascadeStackingEnsemble()
        _, y = _make_mock_cls_outputs()
        with pytest.raises(ValueError, match="No stacker outputs"):
            cs.fit([], y)

    def test_cascade_predict_before_fit(self):
        from tabtune.ensemble.strategies import CascadeStackingEnsemble
        cs = CascadeStackingEnsemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            cs.predict({"L1_model_0": np.zeros((10, 3))})

    def test_cascade_invalid_params(self):
        from tabtune.ensemble.strategies import CascadeStackingEnsemble
        with pytest.raises(ValueError, match="n_levels"):
            CascadeStackingEnsemble(n_levels=0)
        with pytest.raises(ValueError, match="n_folds"):
            CascadeStackingEnsemble(n_folds=1)


# ======================================================================
# Strategy 6: Random-Init Ensemble tests
# ======================================================================

class TestRandomInitEnsemble:
    """Test RandomInitEnsemble with mock data (no GPU needed)."""

    def test_random_init_basic(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        per_seed, y = _make_mock_seed_outputs(n_models=3, n_seeds=3)
        ri = RandomInitEnsemble(n_seeds=3, task_type="classification")
        ri.fit(per_seed, y)
        assert ri.weights_ is not None
        assert len(ri.weights_) == 3

    def test_random_init_predict(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        per_seed, y = _make_mock_seed_outputs(n_models=3, n_seeds=3)
        ri = RandomInitEnsemble(n_seeds=3, task_type="classification")
        ri.fit(per_seed, y)
        preds = ri.predict(per_seed)
        assert len(preds) == len(y)

    def test_random_init_predict_proba(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        per_seed, y = _make_mock_seed_outputs(n_models=3, n_seeds=3)
        ri = RandomInitEnsemble(n_seeds=3, task_type="classification")
        ri.fit(per_seed, y)
        probas = ri.predict_proba(per_seed)
        assert probas.shape == (100, 3)

    def test_random_init_uncertainty(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        per_seed, y = _make_mock_seed_outputs(n_models=3, n_seeds=5)
        ri = RandomInitEnsemble(n_seeds=5, task_type="classification")
        ri.fit(per_seed, y)
        _ = ri.predict_proba(per_seed)
        assert ri.uncertainty_ is not None
        assert ri.uncertainty_.shape == (100,)
        assert np.all(ri.uncertainty_ >= 0)

    def test_random_init_regression(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        rng = np.random.RandomState(42)
        y = rng.randn(100) * 10 + 50
        per_seed = {}
        for i in range(3):
            seeds = []
            for s in range(3):
                seeds.append(y + rng.randn(100) * (1.0 + 0.5 * i))
            per_seed[f"model_{i}"] = seeds

        ri = RandomInitEnsemble(n_seeds=3, task_type="regression")
        ri.fit(per_seed, y)
        preds = ri.predict(per_seed)
        assert len(preds) == len(y)

    def test_random_init_empty(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        ri = RandomInitEnsemble(task_type="classification")
        with pytest.raises(ValueError, match="empty"):
            ri.fit({}, np.array([0, 1, 0]))

    def test_random_init_predict_before_fit(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        ri = RandomInitEnsemble(task_type="classification")
        with pytest.raises(RuntimeError, match="not fitted"):
            ri.predict({"model_0": [np.zeros((10, 3))]})

    def test_random_init_invalid_n_seeds(self):
        from tabtune.ensemble.strategies import RandomInitEnsemble
        with pytest.raises(ValueError, match="n_seeds"):
            RandomInitEnsemble(n_seeds=0)


# ======================================================================
# Strategy factory tests
# ======================================================================

class TestStrategyFactory:
    """Test the get_strategy factory and STRATEGY_MAP."""

    def test_all_strategies_instantiate(self):
        from tabtune.ensemble.strategies import get_strategy, STRATEGY_MAP
        for name in STRATEGY_MAP:
            kwargs = {"task_type": "classification"}
            if name == "greedy_selection":
                kwargs["ensemble_size"] = 10
            elif name == "random_init":
                kwargs["n_seeds"] = 3
            s = get_strategy(name, **kwargs)
            assert s is not None

    def test_unknown_strategy_raises(self):
        from tabtune.ensemble.strategies import get_strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")

    def test_all_six_strategies_present(self):
        from tabtune.ensemble.strategies import STRATEGY_MAP
        expected = {
            "weighted_averaging", "greedy_selection", "stacking",
            "temperature_scaled", "cascade_stacking", "random_init",
        }
        assert set(STRATEGY_MAP.keys()) == expected


# ======================================================================
# Utility function tests
# ======================================================================

class TestUtilities:
    """Test internal utility functions."""

    def test_probas_to_preds(self):
        from tabtune.ensemble.strategies import _probas_to_preds
        probas = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        preds = _probas_to_preds(probas)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_probas_to_preds_1d_raises(self):
        from tabtune.ensemble.strategies import _probas_to_preds
        with pytest.raises(ValueError, match="2-D"):
            _probas_to_preds(np.array([0.1, 0.9]))

    def test_validate_empty_outputs(self):
        from tabtune.ensemble.strategies import _validate_model_outputs
        with pytest.raises(ValueError, match="empty"):
            _validate_model_outputs({})

    def test_validate_inconsistent_sizes(self):
        from tabtune.ensemble.strategies import _validate_model_outputs
        with pytest.raises(ValueError, match="Inconsistent"):
            _validate_model_outputs({
                "a": np.zeros((10, 3)),
                "b": np.zeros((20, 3)),
            })

    def test_get_metric_fn_classification(self):
        from tabtune.ensemble.strategies import _get_metric_fn
        fn, higher = _get_metric_fn("accuracy", "classification")
        assert higher is True
        assert callable(fn)

    def test_get_metric_fn_regression(self):
        from tabtune.ensemble.strategies import _get_metric_fn
        fn, higher = _get_metric_fn("mse", "regression")
        assert higher is False

    def test_get_metric_fn_unknown(self):
        from tabtune.ensemble.strategies import _get_metric_fn
        with pytest.raises(ValueError, match="Unknown metric"):
            _get_metric_fn("f2_score", "classification")


# ======================================================================
# TabularEnsemble API unit tests (no GPU)
# ======================================================================

class TestTabularEnsembleUnit:
    """Test TabularEnsemble API without running actual TFMs."""

    def test_import(self):
        from tabtune.ensemble import TabularEnsemble
        assert TabularEnsemble is not None

    def test_init(self):
        from tabtune.ensemble import TabularEnsemble
        e = TabularEnsemble(
            models=[{"model_name": "TabPFN"}],
            ensemble_strategy="greedy_selection",
        )
        assert e.ensemble_strategy == "greedy_selection"
        assert not e._is_fitted

    def test_init_all_strategies(self):
        from tabtune.ensemble import TabularEnsemble
        for strategy in [
            "weighted_averaging", "greedy_selection", "stacking",
            "temperature_scaled", "cascade_stacking", "random_init",
        ]:
            e = TabularEnsemble(
                models=[{"model_name": "TabPFN"}],
                ensemble_strategy=strategy,
            )
            assert e.ensemble_strategy == strategy

    def test_init_invalid_strategy(self):
        from tabtune.ensemble import TabularEnsemble
        with pytest.raises(ValueError, match="ensemble_strategy"):
            TabularEnsemble(
                models=[{"model_name": "TabPFN"}],
                ensemble_strategy="invalid",
            )

    def test_init_invalid_holdout(self):
        from tabtune.ensemble import TabularEnsemble
        with pytest.raises(ValueError, match="holdout_fraction"):
            TabularEnsemble(
                models=[{"model_name": "TabPFN"}],
                holdout_fraction=0.0,
            )

    def test_get_pipeline_id(self):
        from tabtune.ensemble import TabularEnsemble
        e = TabularEnsemble(models=[])
        assert e._get_pipeline_id(
            {"model_name": "TabPFN", "tuning_strategy": "inference"}
        ) == "TabPFN_inference"
        assert e._get_pipeline_id(
            {"model_name": "OrionMSP", "tuning_strategy": "finetune",
             "finetune_mode": "meta-learning"}
        ) == "OrionMSP_finetune_meta-learning"

    def test_predict_before_fit_raises(self):
        from tabtune.ensemble import TabularEnsemble
        e = TabularEnsemble(models=[{"model_name": "TabPFN"}])
        with pytest.raises(RuntimeError, match="fit"):
            e.predict(np.zeros((10, 5)))

    def test_predict_proba_regression_raises(self):
        from tabtune.ensemble import TabularEnsemble
        e = TabularEnsemble(
            models=[{"model_name": "TabPFN"}],
            task_type="regression",
        )
        e._is_fitted = True  # hack for testing
        with pytest.raises(ValueError, match="classification"):
            e.predict_proba(np.zeros((10, 5)))

    def test_summary(self):
        from tabtune.ensemble import TabularEnsemble
        e = TabularEnsemble(
            models=[{"model_name": "TabPFN"}],
            ensemble_strategy="greedy_selection",
        )
        s = e.summary()
        assert s["ensemble_strategy"] == "greedy_selection"
        assert s["is_fitted"] is False
        assert s["n_models"] == 1


# ======================================================================
# Full integration tests (GPU needed)
# ======================================================================

def _make_cls_data(n=300, d=10, k=3):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n, n_features=d, n_classes=k,
        n_informative=d - 2, random_state=42,
    )
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]),
        pd.Series(y, name="target"),
    )


def _make_reg_data(n=300, d=10):
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=n, n_features=d, random_state=42)
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]),
        pd.Series(y, name="target"),
    )


class TestFullEnsemble:
    """Full integration tests requiring GPU and TabTune models."""

    def test_classification_greedy(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
                {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="greedy_selection",
            task_type="classification",
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)

        result = ensemble.evaluate(Xte, yte)
        assert "ensemble" in result
        assert result["ensemble"]["accuracy"] > 0.5

        lb = ensemble.get_leaderboard()
        assert len(lb) >= 3  # 2 models + 1 ensemble
        print(f"\nLeaderboard:\n{lb}")

    def test_classification_weighted_averaging(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        for scheme in ["uniform", "performance", "inverse_error"]:
            ensemble = TabularEnsemble(
                models=[
                    {"model_name": "TabPFN", "tuning_strategy": "inference"},
                    {"model_name": "TabICLv2", "tuning_strategy": "inference"},
                ],
                ensemble_strategy="weighted_averaging",
                weight_scheme=scheme,
                verbose=False,
            )
            ensemble.fit(Xtr, ytr)
            result = ensemble.evaluate(Xte, yte)
            assert result["ensemble"]["accuracy"] > 0.5
            print(f"\n  WA ({scheme}): acc={result['ensemble']['accuracy']:.4f}")

    def test_classification_with_gbdt(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        ensemble = TabularEnsemble(
            models=[{"model_name": "TabPFN", "tuning_strategy": "inference"}],
            ensemble_strategy="greedy_selection",
            include_gbdt_baselines=True,
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)
        result = ensemble.evaluate(Xte, yte)
        print(f"\nTFM+GBDT ensemble: {result['ensemble']}")

    def test_regression_weighted(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_reg_data(n=300)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
                {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="weighted_averaging",
            task_type="regression",
            metric="mse",
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)
        result = ensemble.evaluate(Xte, yte)
        assert "ensemble" in result
        print(f"\nRegression ensemble: {result['ensemble']}")

    def test_stacking(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
                {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="stacking",
            meta_learner="lr",
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)
        result = ensemble.evaluate(Xte, yte)
        print(f"\nStacking ensemble: {result['ensemble']}")

    def test_temperature_scaled(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
                {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="temperature_scaled",
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)
        result = ensemble.evaluate(Xte, yte)
        print(f"\nTemp-Scaled ensemble: {result['ensemble']}")

    def test_cascade_stacking(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
                {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="cascade_stacking",
            n_cascade_levels=2,
            cv_folds=3,
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)
        result = ensemble.evaluate(Xte, yte)
        print(f"\nCascade Stacking ensemble: {result['ensemble']}")

    def test_random_init_ensemble(self):
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="random_init",
            n_seeds=3,
            base_seed=42,
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)
        result = ensemble.evaluate(Xte, yte)
        print(f"\nDeep Ensemble: {result['ensemble']}")

        uncertainty = ensemble.get_uncertainty()
        if uncertainty is not None:
            print(f"  Mean uncertainty: {uncertainty.mean():.6f}")


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TabTune Ensemble Smoke Test")
    print("=" * 60)

    # Strategy tests (no GPU)
    print("\n--- Strategy unit tests (no GPU) ---")
    outputs, y = _make_mock_cls_outputs(n_samples=200, n_models=5)
    per_seed, y_seed = _make_mock_seed_outputs(n_samples=200, n_models=3, n_seeds=3)

    from tabtune.ensemble.strategies import (
        WeightedAveraging,
        GreedyEnsembleSelection,
        StackingEnsemble,
        TemperatureScaledBlending,
        CascadeStackingEnsemble,
        RandomInitEnsemble,
    )
    from sklearn.metrics import accuracy_score

    # Strategies 1-4
    for StrategyClass, name, kwargs in [
        (WeightedAveraging, "WeightedAveraging (uniform)",
         {"task_type": "classification", "weight_scheme": "uniform"}),
        (WeightedAveraging, "WeightedAveraging (performance)",
         {"task_type": "classification", "weight_scheme": "performance"}),
        (WeightedAveraging, "WeightedAveraging (inverse_error)",
         {"task_type": "classification", "weight_scheme": "inverse_error"}),
        (GreedyEnsembleSelection, "GreedySelection",
         {"task_type": "classification", "ensemble_size": 30}),
        (StackingEnsemble, "Stacking (LR)",
         {"task_type": "classification", "meta_learner": "lr"}),
        (StackingEnsemble, "Stacking (GBDT)",
         {"task_type": "classification", "meta_learner": "gbdt"}),
        (TemperatureScaledBlending, "TempScaled",
         {"task_type": "classification"}),
    ]:
        s = StrategyClass(**kwargs)
        if isinstance(s, WeightedAveraging):
            s.fit(outputs, y, metric="accuracy")
        else:
            s.fit(outputs, y)
        preds = s.predict(outputs)
        acc = accuracy_score(y, preds)
        print(f"  {name:40s}  accuracy={acc:.4f}")

    # Strategy 5: Cascade
    cs = CascadeStackingEnsemble(n_levels=2, ges_size=20, task_type="classification")
    outputs_l2, _ = _make_mock_cls_outputs(n_samples=200, n_models=5, seed=99)
    cs.fit([outputs, outputs_l2], y)
    flat = {}
    for n, a in outputs.items():
        flat[f"L1_{n}"] = a
    for n, a in outputs_l2.items():
        flat[f"L2_{n}"] = a
    preds_cs = cs.predict(flat)
    print(f"  {'CascadeStacking (2 levels)':40s}  accuracy={accuracy_score(y, preds_cs):.4f}")

    # Strategy 6: Random-Init
    ri = RandomInitEnsemble(n_seeds=3, task_type="classification")
    ri.fit(per_seed, y_seed)
    preds_ri = ri.predict(per_seed)
    print(f"  {'RandomInitEnsemble (3 seeds)':40s}  accuracy={accuracy_score(y_seed, preds_ri):.4f}")

    # Full integration test (GPU)
    print("\n--- Full integration test ---")
    try:
        from tabtune.ensemble import TabularEnsemble
        from sklearn.model_selection import train_test_split

        X, y_data = _make_cls_data(n=300, k=2)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_data, test_size=0.25, random_state=42
        )

        ensemble = TabularEnsemble(
            models=[
                {"model_name": "TabPFN", "tuning_strategy": "inference"},
                {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            ],
            ensemble_strategy="greedy_selection",
            verbose=True,
        )
        ensemble.fit(Xtr, ytr)

        result = ensemble.evaluate(Xte, yte)
        print(f"\nEnsemble result: {result['ensemble']}")
        print(f"\nLeaderboard:\n{ensemble.get_leaderboard()}")
        print("\nAll tests PASSED!")

    except Exception as e:
        print(f"GPU test skipped: {e}")
        print("Strategy unit tests PASSED!")