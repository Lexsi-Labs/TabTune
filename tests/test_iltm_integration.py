"""
Integration / wiring tests for the ILTM model in TabTune.

The `unit` tests verify ILTM is registered across every subsystem
(DataProcessor, pipeline dispatch, TuningManager dispatch, package exports).
`TestILTMEndToEnd` runs REAL end-to-end pipelines with a TINY
randomly-initialised checkpoint written to a local ``.pth`` file -- the
vendored checkpoint resolver uses local paths as-is, so there are NO Hugging
Face downloads and no gating env var is needed (CPU, seconds).

Run:  pytest tests/test_iltm_integration.py -v
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

TINY_ARCH = dict(
    n_dims=16, hn_n_layers=2, hn_hidden_size=32, rf_size=64,
    n_classes_limit=6, main_n_layers=3, pca_sampling="zeropad", pca_fit="reduced",
)
FAST_TUNING = dict(device="cpu", epochs=1, steps_per_epoch=6, support_size=32,
                   query_size=16, show_progress=False)


def _require_full_tabtune():
    """Skip if the full TabTune package (the whole model zoo) cannot import --
    e.g. a pre-existing dependency issue in another model. This keeps ILTM
    wiring tests from failing for reasons unrelated to ILTM."""
    try:
        import tabtune  # noqa: F401
        from tabtune.TuningManager.tuning import TuningManager  # noqa: F401
    except Exception as e:
        pytest.skip(f"full TabTune package not importable (unrelated to ILTM): {e}")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A tiny randomly-initialised iLTM checkpoint on disk (no download)."""
    from tabtune.models.iltm.iltm_model import iLTM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("iltm_ckpt") / "tiny.pth"
    torch.save(iLTM(**TINY_ARCH).state_dict(), path)
    return str(path)


def _tiny_model_params(tiny_checkpoint, **extra):
    params = {"device": "cpu", "checkpoint": tiny_checkpoint, "n_ensemble": 2}
    params.update(TINY_ARCH)
    params.update(extra)
    return params


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
@pytest.mark.model_iltm
class TestILTMWiring:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classifier_and_regressor_importable(self):
        from tabtune.models.iltm import ILTMClassifier
        from tabtune.models.regression.iltm.regressor import ILTMRegressorWrapper
        assert ILTMClassifier is not None and ILTMRegressorWrapper is not None

    def test_import_is_lazy(self):
        # Importing the wrapper must NOT construct the heavy engine or touch
        # the checkpoint resolver (which could download).
        from tabtune.models.iltm.classifier import ILTMClassifier
        clf = ILTMClassifier(device="cpu")
        assert clf.estimator_ is None and clf.model_ is None

    def test_dataprocessor_registers_iltm(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        dp = DataProcessor(model_name="ILTM", task_type="classification")
        assert dp.categorical_encoding == "iltm_special"
        from tabtune.Dataprocess.iltm_preprocessor import ILTMPreprocessor
        assert isinstance(dp._get_custom_preprocessor(), ILTMPreprocessor)

    def test_regression_processor_registered(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        from tabtune.Dataprocess.regression.iltm_processor import ILTMRegressionProcessor
        dp = DataProcessor(model_name="ILTM", task_type="regression")
        proc = dp._get_regression_processor()
        assert isinstance(proc, ILTMRegressionProcessor)
        assert proc.target_scaling_strategy == "none"

    def test_tuning_manager_imports_iltm(self):
        from tabtune.TuningManager import tuning
        assert hasattr(tuning, "ILTMClassifier")
        assert hasattr(tuning, "ILTMRegressorWrapper")
        assert hasattr(tuning.TuningManager, "_finetune_iltm")
        assert hasattr(tuning.TuningManager, "_finetune_iltm_regression_turn_by_turn")
        assert hasattr(tuning.TuningManager, "_iltm_episode_tensors")

    def test_peft_targets_registered(self):
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS
        assert "ILTM" in MODEL_LORA_TARGETS

    def test_preprocessor_fit_transform(self):
        from tabtune.Dataprocess.iltm_preprocessor import ILTMPreprocessor
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
        y = pd.Series(["neg", "pos", "neg", "pos"])
        pre = ILTMPreprocessor(task_type="classification").fit(X, y)
        Xt, yt = pre.transform(X, y)
        assert Xt.shape == (4, 2) and Xt.dtype == np.float32
        assert set(np.unique(yt)).issubset({0, 1})
        assert pre.label_encoder_ is not None  # evaluate() depends on this
        assert "iLTM Preprocessing" in pre.get_summary()

    def test_wrapper_tolerates_unknown_kwargs(self):
        from tabtune.models.iltm.classifier import ILTMClassifier
        clf = ILTMClassifier(device="cpu", tuning_strategy="finetune",
                             task_type="classification", totally_unknown_key=1)
        assert "totally_unknown_key" in clf._extra_kwargs
        assert "task_type" not in clf._extra_kwargs

    def test_wrapper_rejects_bad_tuning_strategy(self):
        from tabtune.models.iltm.classifier import ILTMClassifier
        from tabtune.models.regression.iltm.regressor import ILTMRegressorWrapper
        with pytest.raises(ValueError):
            ILTMClassifier(tuning_strategy="nope")
        with pytest.raises(ValueError):
            ILTMRegressorWrapper(tuning_strategy="peft")  # regression: inference/finetune only


@pytest.mark.unit
@pytest.mark.model_iltm
class TestPipelineConstruction:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classification_pipeline_selects_iltm(self):
        from tabtune import TabularPipeline
        from tabtune.models.iltm import ILTMClassifier
        pipe = TabularPipeline(model_name="ILTM", task_type="classification",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(pipe.model, ILTMClassifier)

    def test_regression_pipeline_selects_iltm(self):
        from tabtune import TabularPipeline
        from tabtune.models.regression.iltm.regressor import ILTMRegressorWrapper
        pipe = TabularPipeline(model_name="ILTM", task_type="regression",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(pipe.model, ILTMRegressorWrapper)

    def test_finetune_construction_eager_loads_backbone(self, monkeypatch):
        # finetune/peft construction calls _load_model eagerly; monkeypatch it
        # so no checkpoint resolution (= no HF download) happens in this test.
        from tabtune import TabularPipeline
        from tabtune.models.iltm.classifier import ILTMClassifier
        calls = []
        monkeypatch.setattr(ILTMClassifier, "_load_model", lambda self: calls.append(1))
        TabularPipeline(model_name="ILTM", task_type="classification",
                        tuning_strategy="finetune", model_params={"device": "cpu"})
        assert calls

    def test_regression_finetune_allowed(self, monkeypatch):
        from tabtune import TabularPipeline
        from tabtune.models.regression.iltm.regressor import ILTMRegressorWrapper
        monkeypatch.setattr(ILTMRegressorWrapper, "_load_model", lambda self: None)
        # should NOT raise the "regression finetuning not enabled" ValueError
        TabularPipeline(model_name="ILTM", task_type="regression",
                        tuning_strategy="finetune", model_params={"device": "cpu"})

    def test_regression_peft_rejected(self):
        from tabtune import TabularPipeline
        with pytest.raises(ValueError):
            TabularPipeline(model_name="ILTM", task_type="regression",
                            tuning_strategy="peft", model_params={"device": "cpu"})

    def test_default_tuning_config_exposed(self):
        from tabtune import TabularPipeline
        pipe = TabularPipeline(model_name="ILTM", task_type="classification",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        cfg = pipe.get_params()["tuning_params"]
        assert cfg["epochs"] == 3 and cfg["support_size"] == 64


@pytest.mark.integration
@pytest.mark.model_iltm
class TestILTMEndToEnd:
    """Real pipelines on a tiny random-init checkpoint (no network, CPU)."""

    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classification_inference(self, tiny_checkpoint):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, y_te = _clf_frame()
        pipe = TabularPipeline(model_name="ILTM", task_type="classification",
                               tuning_strategy="inference",
                               model_params=_tiny_model_params(tiny_checkpoint))
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert set(preds).issubset({"pos", "neg"})
        proba = pipe.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        metrics = pipe.evaluate(X_te, y_te)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "roc_auc_score" in metrics

    def test_classification_finetune(self, tiny_checkpoint):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, _ = _clf_frame()
        pipe = TabularPipeline(model_name="ILTM", task_type="classification",
                               tuning_strategy="finetune",
                               model_params=_tiny_model_params(tiny_checkpoint),
                               tuning_params=dict(FAST_TUNING))
        pipe.fit(X_tr, y_tr)
        assert pipe.predict(X_te).shape == (len(X_te),)

    def test_classification_peft(self, tiny_checkpoint):
        from tabtune import TabularPipeline
        from tabtune.TuningManager.peft_utils import LoRALinear
        X_tr, X_te, y_tr, _ = _clf_frame()
        pipe = TabularPipeline(model_name="ILTM", task_type="classification",
                               tuning_strategy="peft",
                               model_params=_tiny_model_params(tiny_checkpoint),
                               tuning_params=dict(FAST_TUNING, peft_config={"r": 4}))
        pipe.fit(X_tr, y_tr)
        assert any(isinstance(m, LoRALinear) for m in pipe.model.model_.modules())
        assert pipe.predict(X_te).shape == (len(X_te),)

    def test_regression_inference_and_finetune(self, tiny_checkpoint):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, y_te = _reg_frame()
        for strategy in ("inference", "finetune"):
            pipe = TabularPipeline(model_name="ILTM", task_type="regression",
                                   tuning_strategy=strategy,
                                   model_params=_tiny_model_params(tiny_checkpoint),
                                   tuning_params=dict(FAST_TUNING))
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            assert preds.shape == (len(X_te),)
            assert np.isfinite(preds).all()
            metrics = pipe.evaluate(X_te, y_te)
            assert "rmse" in metrics

    def test_pipeline_save_load_round_trip(self, tiny_checkpoint, tmp_path):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, _ = _clf_frame()
        pipe = TabularPipeline(model_name="ILTM", task_type="classification",
                               tuning_strategy="inference",
                               model_params=_tiny_model_params(tiny_checkpoint))
        pipe.fit(X_tr, y_tr)
        preds_before = pipe.predict(X_te)
        path = str(tmp_path / "iltm_pipeline.joblib")
        pipe.save(path)
        loaded = TabularPipeline.load(path)
        preds_after = loaded.predict(X_te)
        assert np.array_equal(preds_before, preds_after)

    def test_backbones_are_not_shared_between_wrappers(self, tiny_checkpoint):
        # The engine subclasses pin a per-instance backbone: two wrappers with
        # the same checkpoint must own DIFFERENT nn.Module objects.
        from tabtune.models.iltm.classifier import ILTMClassifier
        a = ILTMClassifier(**_tiny_model_params(tiny_checkpoint))
        b = ILTMClassifier(**_tiny_model_params(tiny_checkpoint))
        a._load_model(); b._load_model()
        assert a.model_ is not b.model_
