"""
Pure model tests for the vendored xRFM engine + the TabTune XRFM wrappers.

The vendored package (`tabtune/models/xrfm/`) is loaded **standalone** via
importlib (it only uses package-relative imports), so these tests do not
depend on the rest of the TabTune model zoo importing successfully. They
need torch + scikit-learn but NO network downloads and NO GPU: xRFM is a
kernel/Recursive-Feature-Machine method that trains from scratch at fit time.

Run:  pytest tests/test_xrfm_model.py -v
"""
import importlib
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_xrfm_pkg():
    """Load tabtune/models/xrfm as a standalone package named `xrfm_std`."""
    if "xrfm_std" in sys.modules:
        return sys.modules["xrfm_std"]
    pkg_dir = os.path.join(_ROOT, "tabtune", "models", "xrfm")
    spec = importlib.util.spec_from_file_location(
        "xrfm_std", os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["xrfm_std"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


try:
    _pkg = _load_xrfm_pkg()
    xRFM = importlib.import_module("xrfm_std.xrfm").xRFM
    XRFMClassifier = _pkg.XRFMClassifier
    XRFMFeatureEncoder = importlib.import_module("xrfm_std.preprocessing").XRFMFeatureEncoder
except Exception as e:  # pragma: no cover
    pytest.skip(f"vendored xRFM standalone import failed: {e}", allow_module_level=True)


def _tiny_rfm_params(iters=3):
    return {
        "model": {"kernel": "l2", "bandwidth": 10.0, "exponent": 1.0,
                  "diag": False, "bandwidth_mode": "constant"},
        "fit": {"reg": 1e-3, "iters": iters, "verbose": False,
                "early_stop_rfm": True, "return_best_params": True},
    }


def _clf_arrays(n=120, d=5, k=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64) if k == 2 else \
        np.digitize(X[:, 0], [-0.5, 0.5]).astype(np.int64)
    return X, y


@pytest.mark.unit
@pytest.mark.model_xrfm
class TestVendoredEngine:
    def test_engine_classification_fit_predict(self):
        X, y = _clf_arrays()
        model = xRFM(rfm_params=_tiny_rfm_params(), device="cpu",
                     tuning_metric="brier", random_state=0, verbose=False)
        model.fit(X[:90], y[:90], X[90:], y[90:])
        preds = model.predict(X[90:])
        assert preds.shape == (30,)
        assert set(np.unique(preds)).issubset({0, 1})
        proba = model.predict_proba(X[90:])
        assert proba.shape == (30, 2)
        assert np.all(np.isfinite(proba))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_engine_regression_fit_predict(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 4).astype(np.float32)
        y = (X[:, 0] * 2.0 + X[:, 1]).astype(np.float32)
        model = xRFM(rfm_params=_tiny_rfm_params(), device="cpu",
                     tuning_metric="mse", random_state=0, verbose=False)
        model.fit(X[:80], y[:80], X[80:], y[80:])
        preds = np.asarray(model.predict(X[80:])).ravel()
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))
        # kernel regression on a linear target should beat the mean predictor
        assert np.mean((preds - y[80:]) ** 2) < np.var(y[80:])

    def test_engine_learns_mahalanobis_matrix(self):
        X, y = _clf_arrays()
        model = xRFM(rfm_params=_tiny_rfm_params(), device="cpu",
                     tuning_metric="brier", random_state=0, verbose=False)
        model.fit(X[:90], y[:90], X[90:], y[90:])
        leaves = [node["model"] for tree in model.trees
                  for node in model._collect_leaf_nodes(tree)]
        assert len(leaves) >= 1
        for leaf in leaves:
            assert leaf.M is not None
            M = leaf.M
            assert M.shape == (X.shape[1], X.shape[1])
            assert torch.all(torch.isfinite(M))
            # engine max-normalises M
            assert float(M.max()) == pytest.approx(1.0, abs=1e-4)
            # AGOP of new samples has the same shape and is finite
            agop = leaf.update_M(torch.as_tensor(X[:20]))
            assert agop.shape == M.shape and torch.all(torch.isfinite(agop))


@pytest.mark.unit
@pytest.mark.model_xrfm
class TestFeatureEncoder:
    def test_mixed_frame_onehot(self):
        X = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", "b", "a", "c"],
        })
        enc = XRFMFeatureEncoder(categorical_encoding="onehot").fit(X)
        Xt = enc.transform(X)
        assert Xt.dtype == np.float32
        assert Xt.shape == (4, 1 + 3)  # scaled numeric + 3 one-hot columns
        assert np.all(np.isfinite(Xt))
        # unknown category at transform time must not crash (all-zero block)
        Xt2 = enc.transform(pd.DataFrame({"num": [1.0], "cat": ["zzz"]}))
        assert Xt2.shape == (1, 4)

    def test_ordinal_mode(self):
        X = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        enc = XRFMFeatureEncoder(categorical_encoding="ordinal").fit(X)
        assert enc.transform(X).shape == (2, 2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            XRFMFeatureEncoder(categorical_encoding="bogus")


@pytest.mark.unit
@pytest.mark.model_xrfm
class TestXRFMClassifierWrapper:
    def _frame(self, n=120, seed=0):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "cat": rng.choice(["red", "green", "blue"], n),
        })
        y = pd.Series(np.where(X["f1"] + 0.5 * X["f2"] > 0, "pos", "neg"))
        return X, y

    def test_fit_predict_proba_contract(self):
        X, y = self._frame()
        clf = XRFMClassifier(device="cpu", iters=2, verbose=False, random_state=0)
        clf.fit(X, y)
        assert clf._is_fitted and clf.model_ is not None
        assert clf.n_classes_ == 2
        assert set(clf.classes_) == {"neg", "pos"}
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({"neg", "pos"})  # ORIGINAL label space
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        # trained kernel should comfortably beat chance on its own training data
        assert (preds == y.values).mean() > 0.7

    def test_multiclass(self):
        rng = np.random.RandomState(1)
        X = pd.DataFrame({"a": rng.randn(150), "b": rng.randn(150)})
        y = pd.Series(np.digitize(X["a"], [-0.5, 0.5])).map({0: "lo", 1: "mid", 2: "hi"})
        clf = XRFMClassifier(device="cpu", iters=2, verbose=False, random_state=0)
        clf.fit(X, y)
        assert clf.n_classes_ == 3
        assert clf.predict_proba(X).shape == (150, 3)

    def test_unknown_kwargs_tolerated(self):
        clf = XRFMClassifier(device="cpu", task_type="classification",
                             ignore_pretraining_limits=True, n_jobs=1)
        assert "task_type" not in clf._extra_kwargs

    def test_invalid_tuning_strategy_rejected(self):
        with pytest.raises(ValueError):
            XRFMClassifier(tuning_strategy="prompt-tune")

    def test_predict_before_fit_raises(self):
        clf = XRFMClassifier(device="cpu")
        with pytest.raises(RuntimeError):
            clf.predict(pd.DataFrame({"a": [1.0]}))

    def test_adaptation_hooks(self):
        X, y = self._frame(n=80)
        clf = XRFMClassifier(device="cpu", iters=2, verbose=False, random_state=0)
        clf.fit(X, y)
        leaves = clf.leaf_models()
        assert len(leaves) >= 1 and all(leaf.M is not None for leaf in leaves)
        X_num = clf.transform_features(X)
        assert X_num.dtype == np.float32 and X_num.shape[0] == len(X)
        y_num = clf.numeric_targets(y)
        assert y_num.shape[0] == len(y) and y_num.dtype == torch.float32

    def test_tiny_dataset_no_holdout_fallback(self):
        # 6 rows: too small for a stratified holdout -> validate on train split
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        clf = XRFMClassifier(device="cpu", iters=1, verbose=False, random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (6,)
