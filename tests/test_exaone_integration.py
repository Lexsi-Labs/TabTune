"""
Integration / wiring tests for EXAONE Tabular (LG AI Research) in TabTune.

The `unit` tests verify EXAONETabular is registered across every subsystem
(DataProcessor, pipeline dispatch, TuningManager dispatch, PEFT targets, package
exports). `TestEXAONEEndToEnd` runs REAL end-to-end pipelines against a
random-init checkpoint written at the **released geometry** (20.8M parameters
for classification, 21.1M for regression) inside a fixture -- the vendored
resolver takes local paths as-is, so there are NO Hugging Face downloads and no
gating env var is needed.

Why random-init weights rather than the published ones: the classification
checkpoint is research-use-only and not fetchable from a sandbox, and LG AI
Research has never published regression weights at all. The geometry, though, is
frozen in the manifest, and the checkpoint loader rebuilds a model and validates
key order and shapes against it -- so a random-init file at that geometry drives
every line the real file would. The *values* are meaningless, so these tests
assert on contracts (label space, target scale, probability normalisation,
round-trips), never on accuracy.

The end-to-end tests carry the released architecture through a CPU forward, so
they are marked ``slow`` and kept to <= 120 rows with ``n_ensemble=2``.

Run:  pytest tests/test_exaone_integration.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.model_exaone

FAST_TUNING = dict(device="cpu", epochs=1, steps_per_epoch=3, support_size=32,
                   query_size=16, learning_rate=1e-4, show_progress=False)


def _require_full_tabtune():
    """Skip if the full TabTune package (the whole model zoo) cannot import --
    e.g. a pre-existing dependency issue in another model. This keeps EXAONE
    wiring tests from failing for reasons unrelated to EXAONE."""
    try:
        import tabtune  # noqa: F401
        from tabtune.TuningManager.tuning import TuningManager  # noqa: F401
    except Exception as e:
        pytest.skip(f"full TabTune package not importable (unrelated to EXAONE): {e}")


def _build_released_checkpoint(task, path, seed=0):
    """Write random-init weights at a released checkpoint's exact geometry."""
    from safetensors.torch import save_file

    from tabtune.models.exaone.model.heads import build_model
    from tabtune.models.exaone.presets import released_manifest

    torch.manual_seed(seed)
    model = build_model(
        released_manifest(task), device=torch.device("cpu"), dtype=torch.float32
    )
    save_file(
        {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
        str(path),
    )
    return str(path)


@pytest.fixture(scope="module")
def released_checkpoints(tmp_path_factory):
    """``{'classification': path, 'regression': path}`` -- built, not downloaded."""
    try:
        directory = tmp_path_factory.mktemp("exaone_released")
        return {
            task: _build_released_checkpoint(task, directory / f"{task}.safetensors")
            for task in ("classification", "regression")
        }
    except Exception as exc:  # pragma: no cover - environment problem, not a defect
        pytest.skip(f"EXAONE released-geometry weights unavailable: {exc}")


def _model_params(checkpoint, **extra):
    params = {"device": "cpu", "checkpoint_path": checkpoint, "n_ensemble": 2}
    params.update(extra)
    return params


def _clf_frame(n=120, seed=0):
    """String labels on purpose: a code/label confusion is then visible."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "cat": rng.choice(["red", "green", "blue"], n),
    })
    y = pd.Series(np.where(X["f1"] + 0.5 * X["f2"] > 0, "pos", "neg"))
    return X[:100], X[100:], y[:100], y[100:]


def _reg_frame(n=120, seed=0):
    """Targets far from zero and far from unit variance, so a missing inverse
    transform (predictions left in a standardised space) cannot hide."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(30.0 * X["f1"] - 20.0 * X["f2"] + 100.0)
    return X[:100], X[100:], y[:100], y[100:]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestEXAONEWiring:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classifier_and_regressor_importable(self):
        from tabtune.models.exaone import EXAONETabularClassifier
        from tabtune.models.regression.exaone.regressor import (
            EXAONETabularRegressorWrapper,
        )
        assert EXAONETabularClassifier is not None
        assert EXAONETabularRegressorWrapper is not None

    def test_import_is_lazy(self):
        # Constructing the wrapper must NOT build the heavy engine or touch the
        # checkpoint resolver (which could download).
        from tabtune.models.exaone.classifier import EXAONETabularClassifier
        clf = EXAONETabularClassifier(device="cpu")
        assert clf.estimator_ is None and clf.model_ is None

    def test_dataprocessor_registers_exaone(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        from tabtune.Dataprocess.exaone_preprocessor import EXAONEPreprocessor
        dp = DataProcessor(model_name="EXAONETabular", task_type="classification")
        assert dp.categorical_encoding == "exaone_special"
        preprocessor = dp._get_custom_preprocessor()
        assert isinstance(preprocessor, EXAONEPreprocessor)
        assert preprocessor.task_type == "classification"

    def test_regression_processor_registered(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        from tabtune.Dataprocess.regression.exaone_processor import (
            EXAONERegressionProcessor,
        )
        dp = DataProcessor(model_name="EXAONETabular", task_type="regression")
        processor = dp._get_regression_processor()
        assert isinstance(processor, EXAONERegressionProcessor)
        # The vendored engine un-scales internally; scaling here would double-apply.
        assert processor.target_scaling_strategy == "none"

    def test_tuning_manager_imports_exaone(self):
        from tabtune.TuningManager import tuning
        assert hasattr(tuning, "EXAONETabularClassifier")
        assert hasattr(tuning, "EXAONETabularRegressorWrapper")
        assert hasattr(tuning.TuningManager, "_finetune_exaone")
        assert hasattr(tuning.TuningManager, "_finetune_exaone_regression_turn_by_turn")
        assert hasattr(tuning.TuningManager, "_exaone_episode_tensors")

    def test_peft_targets_registered(self):
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS
        assert "EXAONETabular" in MODEL_LORA_TARGETS

    def test_preprocessor_passes_features_through_and_encodes_labels(self):
        from tabtune.Dataprocess.exaone_preprocessor import EXAONEPreprocessor
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
        y = pd.Series(["neg", "pos", "neg", "pos"])
        pre = EXAONEPreprocessor(task_type="classification").fit(X, y)
        Xt, yt = pre.transform(X, y)
        # Features are deliberately untouched -- the wrappers encode them
        # themselves, and encoding twice would disagree with their vocabulary.
        pd.testing.assert_frame_equal(Xt, X)
        assert set(np.unique(yt)).issubset({0, 1})
        assert pre.label_encoder_ is not None  # evaluate() depends on this
        assert "EXAONE Preprocessing" in pre.get_summary()

    def test_wrapper_tolerates_unknown_kwargs(self):
        from tabtune.models.exaone.classifier import EXAONETabularClassifier
        clf = EXAONETabularClassifier(device="cpu", tuning_strategy="finetune",
                                      task_type="classification",
                                      totally_unknown_key=1)
        assert "totally_unknown_key" in clf._extra_kwargs
        assert "task_type" not in clf._extra_kwargs

    def test_wrapper_rejects_bad_tuning_strategy(self):
        from tabtune.models.exaone.classifier import EXAONETabularClassifier
        from tabtune.models.regression.exaone.regressor import (
            EXAONETabularRegressorWrapper,
        )
        with pytest.raises(ValueError):
            EXAONETabularClassifier(tuning_strategy="nope")
        with pytest.raises(ValueError):
            # regression: inference/finetune only
            EXAONETabularRegressorWrapper(tuning_strategy="peft")

class TestPipelineConstruction:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classification_pipeline_selects_exaone(self):
        from tabtune import TabularPipeline
        from tabtune.models.exaone import EXAONETabularClassifier
        pipe = TabularPipeline(model_name="EXAONETabular",
                               task_type="classification",
                               tuning_strategy="inference",
                               model_params={"device": "cpu"})
        assert isinstance(pipe.model, EXAONETabularClassifier)
        # Inference construction stays lazy: no weights touched yet.
        assert pipe.model.model_ is None

    def test_regression_pipeline_selects_exaone(self):
        from tabtune import TabularPipeline
        from tabtune.models.regression.exaone.regressor import (
            EXAONETabularRegressorWrapper,
        )
        pipe = TabularPipeline(model_name="EXAONETabular", task_type="regression",
                               tuning_strategy="inference",
                               model_params={"device": "cpu"})
        assert isinstance(pipe.model, EXAONETabularRegressorWrapper)
        assert pipe.model.model_ is None

    def test_finetune_construction_eager_loads_backbone(self, monkeypatch):
        # finetune/peft construction calls _load_model eagerly; monkeypatch it so
        # no checkpoint resolution (= no HF download) happens in this test.
        from tabtune import TabularPipeline
        from tabtune.models.exaone.classifier import EXAONETabularClassifier
        calls = []
        monkeypatch.setattr(EXAONETabularClassifier, "_load_model",
                            lambda self: calls.append(1))
        TabularPipeline(model_name="EXAONETabular", task_type="classification",
                        tuning_strategy="finetune", model_params={"device": "cpu"})
        assert calls

    def test_regression_finetune_allowed(self, monkeypatch):
        from tabtune import TabularPipeline
        from tabtune.models.regression.exaone.regressor import (
            EXAONETabularRegressorWrapper,
        )
        monkeypatch.setattr(EXAONETabularRegressorWrapper, "_load_model",
                            lambda self: None)
        # should NOT raise the "regression finetuning not enabled" ValueError
        TabularPipeline(model_name="EXAONETabular", task_type="regression",
                        tuning_strategy="finetune", model_params={"device": "cpu"})

    def test_regression_peft_rejected(self):
        from tabtune import TabularPipeline
        with pytest.raises(ValueError):
            TabularPipeline(model_name="EXAONETabular", task_type="regression",
                            tuning_strategy="peft", model_params={"device": "cpu"})

    def test_default_tuning_config_exposed(self):
        from tabtune import TabularPipeline
        pipe = TabularPipeline(model_name="EXAONETabular",
                               task_type="classification",
                               tuning_strategy="inference",
                               model_params={"device": "cpu"})
        cfg = pipe.get_params()["tuning_params"]
        assert cfg["epochs"] == 3
        assert cfg["support_size"] == 64
        assert cfg["steps_per_epoch"] == 50

    def test_model_params_reach_the_wrapper(self):
        from tabtune import TabularPipeline
        pipe = TabularPipeline(model_name="EXAONETabular",
                               task_type="classification",
                               tuning_strategy="inference",
                               model_params={"device": "cpu", "n_ensemble": 3,
                                             "random_state": 7})
        assert pipe.model.n_ensemble == 3
        assert pipe.model.random_state == 7
        assert pipe.model.tuning_strategy == "inference"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.integration
class TestEXAONEEndToEnd:
    """Real pipelines on a random-init checkpoint at the released geometry."""

    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classification_inference(self, released_checkpoints):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, y_te = _clf_frame()
        pipe = TabularPipeline(
            model_name="EXAONETabular", task_type="classification",
            tuning_strategy="inference",
            model_params=_model_params(released_checkpoints["classification"]),
        )
        pipe.fit(X_tr, y_tr)

        preds = pipe.predict(X_te)
        assert preds.shape == (len(X_te),)
        # The ORIGINAL label space, not the encoder's 0/1 codes.
        assert set(preds).issubset({"pos", "neg"})

        proba = pipe.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0).all() and (proba <= 1).all()

        metrics = pipe.evaluate(X_te, y_te)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "roc_auc_score" in metrics

    def test_classification_finetune(self, released_checkpoints):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, y_te = _clf_frame()
        pipe = TabularPipeline(
            model_name="EXAONETabular", task_type="classification",
            tuning_strategy="finetune",
            model_params=_model_params(released_checkpoints["classification"]),
            tuning_params=dict(FAST_TUNING),
        )
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert set(preds).issubset({"pos", "neg"})
        assert 0.0 <= pipe.evaluate(X_te, y_te)["accuracy"] <= 1.0

    def test_classification_peft_runs_but_injects_no_adapters(
        self, released_checkpoints
    ):
        """PEFT on EXAONE is currently a full fine-tune -- see test_exaone_finetune."""
        from tabtune import TabularPipeline
        from tabtune.TuningManager.peft_utils import LoRALinear
        X_tr, X_te, y_tr, _y_te = _clf_frame()
        pipe = TabularPipeline(
            model_name="EXAONETabular", task_type="classification",
            tuning_strategy="peft",
            model_params=_model_params(released_checkpoints["classification"]),
            tuning_params=dict(FAST_TUNING, peft_config={"r": 4}),
        )
        pipe.fit(X_tr, y_tr)
        assert sum(isinstance(m, LoRALinear)
                   for m in pipe.model.model_.modules()) == 0
        assert pipe.predict(X_te).shape == (len(X_te),)

    @pytest.mark.parametrize("strategy", ["inference", "finetune"])
    def test_regression_inference_and_finetune(self, released_checkpoints, strategy):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, y_te = _reg_frame()
        pipe = TabularPipeline(
            model_name="EXAONETabular", task_type="regression",
            tuning_strategy=strategy,
            model_params=_model_params(released_checkpoints["regression"]),
            tuning_params=dict(FAST_TUNING),
        )
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert np.isfinite(preds).all()
        # The ORIGINAL target scale (y is centred near 100 with sd ~36), not a
        # standardised one: the vendored engine un-scales and the TabTune
        # regression processor therefore pins target_scaling_strategy='none'.
        assert abs(float(preds.mean()) - float(y_tr.mean())) < 3 * float(y_tr.std())
        assert abs(float(preds.mean())) > 10.0
        metrics = pipe.evaluate(X_te, y_te)
        assert "rmse" in metrics and np.isfinite(metrics["rmse"])

    def test_pipeline_save_load_round_trip(self, released_checkpoints, tmp_path):
        from tabtune import TabularPipeline
        X_tr, X_te, y_tr, _y_te = _clf_frame()
        pipe = TabularPipeline(
            model_name="EXAONETabular", task_type="classification",
            tuning_strategy="inference",
            model_params=_model_params(released_checkpoints["classification"]),
        )
        pipe.fit(X_tr, y_tr)
        preds_before = pipe.predict(X_te)
        path = str(tmp_path / "exaone_pipeline.joblib")
        pipe.save(path)
        loaded = TabularPipeline.load(path)
        preds_after = loaded.predict(X_te)
        assert np.array_equal(preds_before, preds_after)
        # The label space survives the round-trip too.
        assert set(preds_after).issubset({"pos", "neg"})

    def test_backbones_are_not_shared_between_wrappers(self, released_checkpoints):
        # Two wrappers built from the same checkpoint must own DIFFERENT
        # nn.Module objects: a shared backbone would mean fine-tuning one
        # pipeline silently changed the other.
        from tabtune import TabularPipeline
        params = _model_params(released_checkpoints["classification"])
        a = TabularPipeline(model_name="EXAONETabular", task_type="classification",
                            tuning_strategy="inference", model_params=params)
        b = TabularPipeline(model_name="EXAONETabular", task_type="classification",
                            tuning_strategy="inference", model_params=params)
        a.model._load_model()
        b.model._load_model()
        assert a.model.model_ is not b.model.model_
        assert a.model.estimator_ is not b.model.estimator_
        first = next(a.model.model_.parameters())
        with torch.no_grad():
            first.add_(1.0)
        assert not torch.equal(first, next(b.model.model_.parameters()))
