"""Fine-tuned weights must survive the refits TabTune performs before predicting.

Every vendored TabPFN tree (v2, v2.6, v3, v3.5) rebuilds its module from
``model_path`` in ``fit()``, and TabLDM and TabICLv2 rebuild theirs through
``_load_model()``. TabTune refits after fine-tuning (to rebuild the inference
state) and again in ``TabularPipeline.predict``, so before 0.4.0 fine-tuned
runs predicted with the pretrained weights. These tests use tiny, randomly
initialised checkpoints written to disk, so they run offline in seconds. Each
test first shows the weight update being lost without the fix, so a
regression cannot pass silently.

Also covered: the smaller fine-tuning fixes of 0.4.0 (v2.6 PEFT no longer
advertised, v2.6 native fine-tuning pinned to v2.6 weights, ``finetune_mode``
read from ``tuning_params``, a missing-function fallback, a never-set
attribute, and prediction after a native fine-tune).
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from tabtune.TuningManager.tuning import (
    TuningManager,
    _pin_trained_tabpfn,
    _refit_keeping_weights,
    _refit_without_reload,
)

pytestmark = [pytest.mark.unit, pytest.mark.finetuning]

X = np.random.RandomState(0).randn(60, 4).astype(np.float32)
Y = (X[:, 0] > 0).astype(int)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


def _tiny_tabpfn_checkpoint(tree: str, tmp_path) -> tuple[type, str]:
    """Write a tiny random TabPFN classifier checkpoint for ``tree``; return (class, path)."""
    pkg = f"tabtune.models.{tree}"
    base = importlib.import_module(f"{pkg}.base")
    loading = importlib.import_module(f"{pkg}.model_loading")
    classifier = importlib.import_module(f"{pkg}.classifier").TabPFNClassifier
    torch.manual_seed(0)
    if tree == "tabpfn":
        arch = importlib.import_module(f"{pkg}.architectures.base")
        cfg, _ = arch.parse_config({"max_num_classes": 10, "emsize": 16, "nlayers": 1, "nhead": 2,
                                    "features_per_group": 2, "num_buckets": 100})
        model = arch.get_architecture(cfg, n_out=10, cache_trainset_representation=False)
        seed = classifier(model_path=base.ClassifierModelSpecs(model, cfg), device="cpu", n_estimators=1)
        seed.fit(X, Y)
        path = tmp_path / "tabpfn-v2-classifier-tiny.ckpt"
        loading.save_tabpfn_model(seed, path)
        return classifier, str(path)
    # v2.6 uses its own architecture; the v3 / v3.5 trees also load v2.5-format checkpoints
    version = "V2_6" if tree == "tabpfnv26" else "V2_5"
    arch = importlib.import_module(f"{pkg}.architectures." + ("tabpfn_v2_6" if version == "V2_6" else "tabpfn_v2_5"))
    icfg = importlib.import_module(f"{pkg}.inference_config")
    consts = importlib.import_module(f"{pkg}.constants")
    cfg, _ = arch.parse_config({"max_num_classes": 10, "emsize": 16, "nlayers": 1, "nhead": 2,
                                "num_thinking_rows": 2, "features_per_group": 2})
    model = arch.get_architecture(cfg, cache_trainset_representation=False)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, 0.0, 0.02)
    inference = icfg.InferenceConfig.get_default("multiclass", getattr(consts.ModelVersion, version))
    seed = classifier(model_path=base.ClassifierModelSpecs(model, cfg, inference), device="cpu", n_estimators=1)
    seed.fit(X, Y)
    name = "tabpfn-v2.6-classifier-tiny.ckpt" if version == "V2_6" else "tabpfn-v2.5-classifier-tiny.ckpt"
    path = tmp_path / name
    loading.save_tabpfn_model(seed, path, additional_fields={"inference_config": dataclasses.asdict(seed.inference_config_)})
    return classifier, str(path)


def _first_parameter(estimator) -> torch.Tensor:
    module = estimator.model_ if hasattr(estimator, "model_") and not hasattr(estimator, "models_") else estimator.models_[0]
    return next(module.parameters())


def _fitted_and_perturbed(tree, tmp_path):
    classifier, path = _tiny_tabpfn_checkpoint(tree, tmp_path)
    estimator = classifier(model_path=path, device="cpu", n_estimators=1)
    estimator.fit(X, Y)
    parameter = _first_parameter(estimator)
    with torch.no_grad():
        parameter.add_(1.0)  # stands in for a fine-tuning update
    return estimator, parameter.detach().clone()


@pytest.mark.parametrize("tree", ["tabpfn", "tabpfnv26", "tabpfnv3", "tabpfnv35"])
def test_plain_refit_reloads_the_checkpoint(tree, tmp_path):
    """The bug: fit() replaces an updated module with the checkpoint's weights."""
    estimator, updated = _fitted_and_perturbed(tree, tmp_path)
    estimator.fit(X, Y)
    assert not torch.equal(_first_parameter(estimator).detach(), updated)


@pytest.mark.parametrize("tree", ["tabpfn", "tabpfnv26", "tabpfnv3", "tabpfnv35"])
def test_pinned_refit_keeps_the_trained_weights(tree, tmp_path):
    estimator, updated = _fitted_and_perturbed(tree, tmp_path)
    _refit_keeping_weights(estimator, X, Y)
    assert torch.equal(_first_parameter(estimator).detach(), updated)
    assert np.isfinite(estimator.predict_proba(X[:5])).all()
    estimator.fit(X, Y)  # later refits (TabularPipeline.predict) keep them too
    assert torch.equal(_first_parameter(estimator).detach(), updated)


def test_tune_pins_tabpfn_after_finetuning_only(tmp_path, monkeypatch):
    estimator, updated = _fitted_and_perturbed("tabpfnv3", tmp_path)
    manager = TuningManager()
    monkeypatch.setattr(manager, "_tune", lambda model, *a, **k: model)
    manager.tune(estimator, X, Y, strategy="inference")
    assert isinstance(estimator.model_path, str)  # inference: untouched
    manager.tune(estimator, X, Y, strategy="finetune")
    estimator.fit(X, Y)
    assert torch.equal(_first_parameter(estimator).detach(), updated)


def test_regressor_pin_carries_the_bar_distribution():
    from tabtune.models.tabpfnv3.base import RegressorModelSpecs
    from tabtune.models.tabpfnv3.regressor import TabPFNRegressor

    regressor = TabPFNRegressor.__new__(TabPFNRegressor)
    regressor.models_, regressor.configs_ = [torch.nn.Linear(1, 1)], ["config"]
    regressor.inference_config_, regressor.znorm_space_bardist_ = "inference", "bardist"
    assert _pin_trained_tabpfn(regressor)
    assert isinstance(regressor.model_path, RegressorModelSpecs)
    assert regressor.model_path.norm_criterion == "bardist"
    assert regressor.model_path.model is regressor.models_[0]


def test_pin_ignores_other_and_unfitted_models():
    from tabtune.models.tabpfnv3.classifier import TabPFNClassifier

    assert _pin_trained_tabpfn(object()) is False
    assert _pin_trained_tabpfn(TabPFNClassifier.__new__(TabPFNClassifier)) is False


def _finetuned_tabiclv2_regressor(tmp_path):
    from tabtune.models.tabiclv2.model.tabicl import TabICL
    from tabtune.models.tabiclv2.sklearn.regressor import TabICLRegressor

    cfg = dict(max_classes=0, num_quantiles=17, embed_dim=16, col_num_blocks=1, col_nhead=2,
               col_num_inds=4, row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1,
               icl_nhead=2, ff_factor=1)
    torch.manual_seed(0)
    path = tmp_path / "tiny-regressor.ckpt"
    torch.save({"config": cfg, "state_dict": TabICL(**cfg).state_dict()}, path)
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(120, 5)).astype(np.float32)
    yr = (Xr[:, 0] * 2 + Xr[:, 1]).astype(np.float32)
    regressor = TabICLRegressor(model_path=str(path), allow_auto_download=False, n_estimators=1, device="cpu")
    TuningManager()._finetune_tabiclv2_regression(
        regressor, Xr, yr, params={"epochs": 1, "steps_per_epoch": 5, "learning_rate": 1e-2,
                                   "support_size": 40, "query_size": 20, "show_progress": False,
                                   "device": "cpu"})
    return regressor, torch.load(path)["state_dict"], Xr


def _finetuned_tabldm_classifier(tmp_path):
    from test_tabldm_integration import make_classifier, tiny_config, tiny_model

    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(140, 6))
    yc = (Xc[:, 0] + Xc[:, 1] > 0).astype(int)
    path = tmp_path / "classification.ckpt"
    torch.save({"config": tiny_config(4), "state_dict": tiny_model(4).state_dict()}, path)
    classifier = make_classifier(str(path))
    TuningManager()._finetune_tabldm(classifier, Xc[:100], yc[:100], task="classification",
                                     params={"epochs": 2, "steps_per_epoch": 3, "learning_rate": 1e-2,
                                             "support_size": 45, "query_size": 19})
    return classifier, torch.load(path)["state_dict"], Xc


def _differs(state, pretrained) -> bool:
    return any(not torch.equal(state[k].cpu(), pretrained[k]) for k in pretrained)


def test_tabiclv2_regression_finetune_keeps_its_weights(tmp_path):
    regressor, pretrained, Xr = _finetuned_tabiclv2_regressor(tmp_path)
    assert _differs(regressor.model_.state_dict(), pretrained)
    quantiles = np.asarray(regressor.predict(Xr[:4], output_type="quantiles", alphas=[0.1, 0.9]))
    assert np.isfinite(quantiles).all()


def test_tabldm_finetune_keeps_its_weights(tmp_path):
    classifier, pretrained, Xc = _finetuned_tabldm_classifier(tmp_path)
    assert _differs(classifier.model_.state_dict(), pretrained)
    assert np.isfinite(classifier.predict_proba(Xc[100:105])).all()


@pytest.mark.parametrize("build", [_finetuned_tabiclv2_regressor, _finetuned_tabldm_classifier])
def test_finetuned_weights_survive_pickle_and_deepcopy(build, tmp_path):
    """Their __getstate__ used to drop the module and __setstate__ reload the checkpoint."""
    import copy
    import pickle

    estimator, pretrained, X = build(tmp_path)
    predict = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    expected = np.asarray(predict(X[:4]))
    for twin in (pickle.loads(pickle.dumps(estimator)), copy.deepcopy(estimator)):
        assert _differs(twin.model_.state_dict(), pretrained)
        twin_predict = twin.predict_proba if hasattr(twin, "predict_proba") else twin.predict
        np.testing.assert_allclose(np.asarray(twin_predict(X[:4])), expected, rtol=1e-5, atol=1e-6)
        again = pickle.loads(pickle.dumps(twin))  # the flag survives a second round trip
        assert _differs(again.model_.state_dict(), pretrained)


def test_a_second_finetune_starts_from_the_checkpoint(tmp_path):
    """One pipeline reused across folds must not carry fold A's training into fold B."""
    from tabtune import TabularPipeline

    classifier, path = _tiny_tabpfn_checkpoint("tabpfn", tmp_path)
    rng = np.random.default_rng(0)
    Xa = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    Xb = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    ya, yb = pd.Series((Xa.a > 0).astype(int)), pd.Series((Xb.b > 0).astype(int))
    checkpoint = next(classifier(model_path=path, device="cpu", n_estimators=1).fit(Xa.values, ya.values)
                      .model_.parameters()).detach().clone()
    pipe = TabularPipeline(model_name="TabPFN", task_type="classification", tuning_strategy="finetune",
                           model_params={"model_path": path, "n_estimators": 1, "device": "cpu"},
                           tuning_params={"epochs": 1, "device": "cpu", "show_progress": False,
                                          "learning_rate": 1e-2, "batch_size": 64})
    pipe.fit(Xa, ya)
    fold_a = next(pipe.model.model_.parameters()).detach().clone()
    assert not torch.equal(fold_a, checkpoint)
    pipe.tuning_params["learning_rate"] = 0.0
    pipe.fit(Xb, yb)
    assert torch.equal(next(pipe.model.model_.parameters()).detach(), checkpoint)


def test_pinned_native_finetuners_keep_sklearn_parameters():
    from sklearn.base import clone

    from tabtune.models.tabpfnv26.finetuning._tabtune_v26_pin import V26PinnedFinetunedClassifier

    finetuner = V26PinnedFinetunedClassifier(epochs=3, device="cpu")
    assert finetuner.get_params()["epochs"] == 3 and clone(finetuner).epochs == 3
    assert "V26PinnedFinetunedClassifier" in repr(finetuner)


def test_tabicl_sft_zeroes_gradients_every_step(tmp_path, monkeypatch):
    """Without zero_grad each step applied the sum of all earlier gradients."""
    from tabtune.models.tabiclv2.model.tabicl import TabICL
    from tabtune.models.tabiclv2.sklearn.classifier import TabICLClassifier

    calls = {"zero": 0, "step": 0}

    class SpyAdam(torch.optim.Adam):
        def zero_grad(self, set_to_none=True):
            calls["zero"] += 1
            return super().zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            calls["step"] += 1
            return super().step(closure)

    cfg = dict(max_classes=10, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
               row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2, ff_factor=1)
    torch.manual_seed(0)
    path = tmp_path / "tiny-classifier.ckpt"
    torch.save({"config": cfg, "state_dict": TabICL(**cfg).state_dict()}, path)
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(96, 5)).astype(np.float32)
    yc = (Xc[:, 0] > 0).astype(np.int64)
    classifier = TabICLClassifier(model_path=str(path), allow_auto_download=False, n_estimators=1, device="cpu")
    monkeypatch.setattr(torch.optim, "Adam", SpyAdam)
    TuningManager()._finetune_tabicl_simple_sft(
        classifier, Xc, yc, params={"epochs": 1, "batch_size": 32, "device": "cpu", "show_progress": False,
                                    "support_size": 8, "query_size": 8})
    assert calls["step"] > 0 and calls["zero"] >= calls["step"]


def test_refit_without_reload_restores_load_model():
    calls = []

    class Estimator:
        model_ = "trained"

        def _load_model(self):
            calls.append("load")
            self.model_ = "pretrained"

        def fit(self, X, y):
            self._load_model()
            return self

    estimator = Estimator()
    _refit_without_reload(estimator, X, Y)
    assert estimator.model_ == "trained" and calls == []
    estimator.fit(X, Y)  # the class method is back afterwards
    assert calls == ["load"]


def test_tabpfnv26_no_longer_offers_peft():
    from tabtune.registry import get_model_spec, validate_request
    from tabtune.registry.errors import UnsupportedStrategyError

    spec = get_model_spec("TabPFNv26")
    assert "peft" not in spec.classification_strategies
    with pytest.raises(UnsupportedStrategyError):
        validate_request("TabPFNv26", "classification", "peft")


def test_tabpfnv26_native_finetuners_build_v26_estimators():
    from tabtune.models.tabpfnv26.finetuning._tabtune_v26_pin import (
        V26PinnedFinetunedClassifier,
        V26PinnedFinetunedRegressor,
    )

    for cls in (V26PinnedFinetunedClassifier, V26PinnedFinetunedRegressor):
        estimator = cls(device="cpu")._create_estimator({"device": "cpu"})
        assert "v2.6" in str(estimator.model_path)
        assert estimator.fit_mode == "batched"


def test_finetune_mode_is_read_from_tuning_params():
    from tabtune import TabularPipeline

    pipeline = TabularPipeline("XRFM", task_type="classification", tuning_strategy="finetune",
                               tuning_params={"finetune_mode": "sft"})
    assert pipeline.finetune_mode == "sft"
    assert pipeline.tuning_params["finetune_mode"] == "sft"
    default = TabularPipeline("XRFM", task_type="classification", tuning_strategy="finetune")
    assert default.finetune_mode == "meta-learning"


def test_v35_regression_default_mode_is_native():
    from tabtune.models.regression.tabpfnv35.regressor import TabPFNv35RegressorWrapper

    manager = TuningManager()
    manager._finetune_tabpfnv35_native_regressor = lambda m, *a, **k: "native"
    manager._finetune_tabpfnv3_regression_turn_by_turn = lambda m, *a, **k: "turn_by_turn"
    stub = TabPFNv35RegressorWrapper.__new__(TabPFNv35RegressorWrapper)
    Xr, yr = np.zeros((10, 2)), np.arange(10.0)
    assert manager.tune(stub, Xr, yr, strategy="finetune", params={}) == "native"
    assert manager.tune(stub, Xr, yr, strategy="finetune", params={"finetune_mode": "turn_by_turn"}) == "turn_by_turn"


def test_tabpfnv26_native_regression_reports_a_missing_finetuner(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "tabpfnv26.finetuning" in (name or ""):
            raise ImportError("simulated")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ImportError, match="TabPFNv26 regression fine-tuning"):
        TuningManager()._finetune_tabpfnv26_native_regressor(object(), np.zeros((4, 2)), np.zeros(4), params={})


class _Processor:
    def transform(self, X, y=None):
        return X if y is None else (X, y)


def _stub_pipeline(model, task):
    from tabtune.TabularPipeline.pipeline import TabularPipeline

    pipeline = TabularPipeline.__new__(TabularPipeline)
    pipeline._is_fitted, pipeline.task_type, pipeline.model = True, task, model
    pipeline.processor, pipeline.tuning_strategy, pipeline.model_name = _Processor(), "finetune", "TabPFNv3"
    return pipeline


@pytest.mark.parametrize("path", [
    "tabtune.models.tabpfnv3.classifier:TabPFNClassifier",
    "tabtune.models.tabpfnv35.finetuning.finetuned_classifier:FinetunedTabPFNClassifier",
])
def test_predict_proba_after_native_classification_finetune(path):
    module, name = path.split(":")
    cls = getattr(importlib.import_module(module), name)
    model = cls.__new__(cls)
    model.models_ = [torch.nn.Linear(1, 1)]  # a fitted estimator has its module
    model.predict_proba = lambda X: np.full((len(X), 2), 0.5)
    model.fit = lambda *a, **k: pytest.fail("a native result must not be refitted")
    out = _stub_pipeline(model, "classification")._predict_proba_uncached(pd.DataFrame({"a": [1.0, 2.0]}))
    assert np.shape(out) == (2, 2)


@pytest.mark.parametrize("path", [
    "tabtune.models.tabpfnv26.regressor:TabPFNRegressor",
    "tabtune.models.tabpfnv3.regressor:TabPFNRegressor",
    "tabtune.models.tabpfnv35.finetuning.finetuned_regressor:FinetunedTabPFNRegressor",
])
def test_predict_quantiles_after_native_regression_finetune(path):
    module, name = path.split(":")
    cls = getattr(importlib.import_module(module), name)
    model = cls.__new__(cls)
    model.predict = lambda X, output_type="mean", quantiles=None: [np.zeros(len(X)) for _ in quantiles]
    out = _stub_pipeline(model, "regression").predict_quantiles(pd.DataFrame({"a": [1.0, 2.0]}), [0.1, 0.9])
    assert list(out) == [0.1, 0.9]
