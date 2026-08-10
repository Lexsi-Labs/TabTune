"""
Integration / wiring tests for the XRFM model in TabTune.

The `unit` tests verify XRFM is registered across every subsystem
(DataProcessor, pipeline dispatch, TuningManager dispatch, package exports).
The `TestXRFMEndToEnd` class runs REAL end-to-end pipelines -- xRFM has no
pretrained checkpoint, so there are NO network downloads and no gating env
var is needed; everything trains from scratch on tiny synthetic/sklearn data
(CPU, seconds).

Run:  pytest tests/test_xrfm_integration.py -v
"""
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _require_full_tabtune():
    """Skip if the full TabTune package (the whole model zoo) cannot import --
    e.g. a pre-existing dependency issue in another model. This keeps XRFM
    wiring tests from failing for reasons unrelated to XRFM."""
    try:
        import tabtune  # noqa: F401
        from tabtune.TuningManager.tuning import TuningManager  # noqa: F401
    except Exception as e:
        pytest.skip(f"full TabTune package not importable (unrelated to XRFM): {e}")


def _clf_frame(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "cat": rng.choice(["red", "green", "blue"], n),
    })
    y = pd.Series(np.where(X["f1"] + 0.5 * X["f2"] > 0, "pos", "neg"))
    return X[:100], X[100:], y[:100], y[100:]


def _reg_frame(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(3.0 * X["f1"] - 2.0 * X["f2"] + 0.1 * rng.randn(n))
    return X[:100], X[100:], y[:100], y[100:]


@pytest.mark.unit
@pytest.mark.model_xrfm
class TestXRFMWiring:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classifier_and_regressor_importable(self):
        from tabtune.models.xrfm import XRFMClassifier
        from tabtune.models.regression.xrfm.regressor import XRFMRegressorWrapper
        assert XRFMClassifier is not None and XRFMRegressorWrapper is not None

    def test_dataprocessor_registers_xrfm(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        dp = DataProcessor(model_name="XRFM", task_type="classification")
        assert dp.categorical_encoding == "xrfm_special"
        from tabtune.Dataprocess.xrfm_preprocessor import XRFMPreprocessor
        assert isinstance(dp._get_custom_preprocessor(), XRFMPreprocessor)

    def test_regression_processor_registered(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        from tabtune.Dataprocess.regression.xrfm_processor import XRFMRegressionProcessor
        dp = DataProcessor(model_name="XRFM", task_type="regression")
        assert isinstance(dp._get_regression_processor(), XRFMRegressionProcessor)

    def test_tuning_manager_imports_xrfm(self):
        from tabtune.TuningManager import tuning
        assert hasattr(tuning, "XRFMClassifier")
        assert hasattr(tuning, "XRFMRegressorWrapper")
        assert hasattr(tuning.TuningManager, "_finetune_xrfm")
        assert hasattr(tuning.TuningManager, "_peft_xrfm")
        assert hasattr(tuning.TuningManager, "_finetune_xrfm_regression")

    def test_preprocessor_fit_transform(self):
        from tabtune.Dataprocess.xrfm_preprocessor import XRFMPreprocessor
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
        y = pd.Series([0, 1, 0, 1])
        pre = XRFMPreprocessor(task_type="classification").fit(X, y)
        Xt, yt = pre.transform(X, y)
        assert Xt.shape == (4, 3) and Xt.dtype == np.float32  # 1 numeric + 2 one-hot
        assert set(np.unique(yt)).issubset({0, 1})
        assert pre.label_encoder_ is not None


@pytest.mark.unit
@pytest.mark.model_xrfm
class TestPipelineConstruction:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classification_pipeline_selects_xrfm(self):
        from tabtune import TabularPipeline
        from tabtune.models.xrfm import XRFMClassifier
        pipe = TabularPipeline(model_name="XRFM", task_type="classification",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(pipe.model, XRFMClassifier)

    def test_regression_pipeline_selects_xrfm(self):
        from tabtune import TabularPipeline
        from tabtune.models.regression.xrfm.regressor import XRFMRegressorWrapper
        pipe = TabularPipeline(model_name="XRFM", task_type="regression",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(pipe.model, XRFMRegressorWrapper)

    def test_regression_finetune_allowed(self):
        from tabtune import TabularPipeline
        # should NOT raise the "regression finetuning not enabled" ValueError
        TabularPipeline(model_name="XRFM", task_type="regression",
                        tuning_strategy="finetune", model_params={"device": "cpu"})

    def test_regression_peft_rejected(self):
        from tabtune import TabularPipeline
        with pytest.raises(ValueError):
            TabularPipeline(model_name="XRFM", task_type="regression",
                            tuning_strategy="peft", model_params={"device": "cpu"})


@pytest.mark.unit
@pytest.mark.model_xrfm
class TestImportSafety:
    def test_importing_package_does_not_load_heavy_engine(self):
        # The vendored engine (torch-heavy) must only load lazily inside
        # XRFMClassifier._load_model / fit, never on package import.
        # Snapshot + restore sys.modules so class identities stay stable for
        # later pickling (joblib) tests in this session.
        saved = {m: sys.modules[m] for m in list(sys.modules) if m.startswith("tabtune.models.xrfm")}
        for m in saved:
            del sys.modules[m]
        try:
            import tabtune.models.xrfm  # noqa: F401
            assert "tabtune.models.xrfm.xrfm" not in sys.modules
            assert "tabtune.models.xrfm.rfm_src.recursive_feature_machine" not in sys.modules
        except ImportError as e:
            pytest.skip(f"base tabtune not importable (unrelated to XRFM): {e}")
        finally:
            for m in list(sys.modules):
                if m.startswith("tabtune.models.xrfm"):
                    del sys.modules[m]
            sys.modules.update(saved)


@pytest.mark.integration
@pytest.mark.model_xrfm
class TestXRFMEndToEnd:
    """Real end-to-end runs: no downloads (xRFM trains from scratch), tiny data."""

    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_inference_classification(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = _clf_frame()
        pipe = TabularPipeline(model_name="XRFM", task_type="classification",
                               tuning_strategy="inference",
                               model_params={"device": "cpu", "iters": 2, "verbose": False})
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        assert preds.shape[0] == len(Xte)
        assert set(np.unique(preds)).issubset({"neg", "pos"})
        proba = pipe.predict_proba(Xte)
        assert proba.shape == (len(Xte), 2)
        metrics = pipe.evaluate(Xte, yte)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert np.isfinite(metrics["roc_auc_score"])

    def test_finetune_classification(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = _clf_frame()
        pipe = TabularPipeline(model_name="XRFM", task_type="classification",
                               tuning_strategy="finetune",
                               model_params={"device": "cpu", "verbose": False},
                               tuning_params={"iters": 3, "bandwidth": 8.0, "device": "cpu"})
        pipe.fit(Xtr, ytr)
        # the finetune config must have reached the wrapper
        assert pipe.model.iters == 3 and pipe.model.bandwidth == 8.0
        metrics = pipe.evaluate(Xte, yte)
        assert metrics["accuracy"] > 0.6  # separable synthetic task

    def test_peft_classification(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = _clf_frame()
        pipe = TabularPipeline(model_name="XRFM", task_type="classification",
                               tuning_strategy="peft",
                               model_params={"device": "cpu", "iters": 2, "verbose": False},
                               tuning_params={"lora_rank": 2, "peft_alpha": 0.5, "device": "cpu"})
        pipe.fit(Xtr, ytr)
        metrics = pipe.evaluate(Xte, yte)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert pipe.predict_proba(Xte).shape == (len(Xte), 2)

    def test_inference_regression(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = _reg_frame()
        pipe = TabularPipeline(model_name="XRFM", task_type="regression",
                               tuning_strategy="inference",
                               model_params={"device": "cpu", "iters": 2, "verbose": False})
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        assert preds.shape[0] == len(Xte) and np.all(np.isfinite(preds))
        metrics = pipe.evaluate(Xte, yte)
        assert metrics["r2_score"] > 0.5  # near-linear synthetic target

    def test_finetune_regression(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = _reg_frame()
        pipe = TabularPipeline(model_name="XRFM", task_type="regression",
                               tuning_strategy="finetune",
                               model_params={"device": "cpu", "verbose": False},
                               tuning_params={"iters": 3, "device": "cpu"})
        pipe.fit(Xtr, ytr)
        assert pipe.model.iters == 3
        metrics = pipe.evaluate(Xte, yte)
        assert metrics["r2_score"] > 0.5

    def test_pipeline_save_load_roundtrip(self, tmp_path):
        import joblib
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = _clf_frame()
        pipe = TabularPipeline(model_name="XRFM", task_type="classification",
                               tuning_strategy="inference",
                               model_params={"device": "cpu", "iters": 1, "verbose": False})
        pipe.fit(Xtr, ytr)
        path = tmp_path / "xrfm_pipeline.joblib"
        joblib.dump(pipe, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.predict(Xte), pipe.predict(Xte))
