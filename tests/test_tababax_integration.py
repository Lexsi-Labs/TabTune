"""TabAbax integration: vendored tree, estimators, fine-tuning, LoRA, persistence and pipeline.

Every test runs on a tiny randomly initialised TabAbax built from the real
architecture code, so nothing is downloaded.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from tabtune.Dataprocess.data_processor import DataProcessor
from tabtune.Dataprocess.tababax_preprocessor import TabAbaxPreprocessor
from tabtune.models.regression.tababax.regressor import TabAbaxRegressorWrapper
from tabtune.models.tababax import (
    TABABAX_LORA_EXCLUDE,
    TABABAX_LORA_TARGETS,
    TabAbaxTabTuneClassifier,
    TabAbaxTabTuneRegressor,
    finetune,
)
from tabtune.models.tababax import tabtune_support as support
from tabtune.models.tababax._model.model import TabAbax
from tabtune.registry import MODEL_REGISTRY, get_model_spec, resolve_model_name
from tabtune.TuningManager.peft_utils import (
    MODEL_LORA_TARGETS,
    FunctionalWeightLoRALinear,
    LoRALinear,
)

pytestmark = [pytest.mark.integration, pytest.mark.model_tababax]

TINY = dict(
    num_quantiles=19, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
    row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2, ff_factor=2,
)
FT_PARAMS = {"epochs": 1, "steps_per_epoch": 2, "support_size": 60, "query_size": 30, "learning_rate": 1e-3}
VAL_PARAMS = {**FT_PARAMS, "epochs": 2, "min_validation_rows": 8, "patience": 1}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("tababax")
    paths = {}
    for task, max_classes in (("classification", 10), ("regression", 0)):
        torch.manual_seed(0)
        model = TabAbax(max_classes=max_classes, **TINY)
        # Residual branches are zero-initialised; randomise them so outputs depend on the inputs.
        with torch.no_grad():
            for parameter in model.parameters():
                if not parameter.any():
                    parameter.normal_(0, 0.05)
        path = root / f"{task}.ckpt"
        torch.save({"config": {"max_classes": max_classes, **TINY}, "state_dict": model.state_dict()}, path)
        paths[task] = str(path)
    return paths


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 5)), columns=[f"f{i}" for i in range(5)])
    X["cat"] = rng.choice(list("abc"), size=160)
    X.loc[X.index[:5], "f3"] = np.nan
    y_cls = np.array(["low", "mid", "high"])[(X["f0"] > 0).astype(int) + (X["f1"] > 0.8)]
    y_reg = (X["f0"] * 3 + X["f2"]).to_numpy()
    return X, y_cls, y_reg


def make(task, checkpoints, **kwargs):
    kwargs.setdefault("n_estimators", 2)
    estimator = TabAbaxTabTuneClassifier if task == "classification" else TabAbaxTabTuneRegressor
    return estimator(model_path=checkpoints[task], allow_auto_download=False, device="cpu", **kwargs)


def fitted(task, checkpoints, frame, **kwargs):
    X, y_cls, y_reg = frame
    return make(task, checkpoints, **kwargs).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])


def predict(estimator, X):
    return estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else estimator.predict(X)


class TestRegistry:
    def test_spec(self) -> None:
        assert "TabAbax" in MODEL_REGISTRY
        for alias in ("Tab-Abax", "TabAbax-v1", "BaiRong-TabAbax", "tababax"):
            assert resolve_model_name(alias) == "TabAbax"
        spec = get_model_spec("TabAbax")
        assert spec.classification_strategies == {"inference", "finetune", "peft"}
        assert spec.regression_strategies == {"inference", "finetune", "peft"}
        assert spec.finetune_modes == {"meta-learning"}
        assert spec.license.commercial_use_ok is False
        assert spec.weights == "tababax-ai/TabAbax"

    def test_data_processor_routes_to_passthrough(self) -> None:
        processor = DataProcessor(model_name="TabAbax", task_type="classification")
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": list("xyxy")})
        processor.fit(X, pd.Series(["p", "q", "p", "q"]))
        assert isinstance(processor.custom_preprocessor_, TabAbaxPreprocessor)
        out = processor.transform(X)
        assert list(out.columns) == ["a", "b"] and np.isnan(out["a"].iloc[1])

    def test_lora_table(self) -> None:
        config = MODEL_LORA_TARGETS["TabAbax"]
        assert tuple(config.target_substrings) == TABABAX_LORA_TARGETS
        assert tuple(config.functional_weight_substrings) == support.TABABAX_FUNCTIONAL_WEIGHT_LEAVES


class TestVendoredTree:
    def test_upstream_package_name_is_not_imported(self) -> None:
        assert "tababax" not in sys.modules

    def test_licence_files_ship_with_the_tree(self) -> None:
        from pathlib import Path

        import tabtune.models.tababax as package

        root = Path(package.__file__).parent
        for name in ("LICENSE", "NOTICE", "PROVENANCE.md", "LICENSES/BSD-3-Clause.txt"):
            assert (root / name).is_file(), name


class TestEstimators:
    def test_classifier(self, checkpoints, frame) -> None:
        X, y, _ = frame
        clf = fitted("classification", checkpoints, frame)
        proba = clf.predict_proba(X[120:])
        assert proba.shape == (40, 3) and np.allclose(proba.sum(axis=1), 1.0)
        assert set(clf.predict(X[120:])) <= set(y)

    def test_regressor(self, checkpoints, frame) -> None:
        X, _, _ = frame
        reg = fitted("regression", checkpoints, frame)
        prediction = reg.predict(X[120:])
        assert prediction.shape == (40,) and np.isfinite(prediction).all()

    def test_many_classes_at_inference(self, checkpoints, frame) -> None:
        X, _, _ = frame
        clf = make("classification", checkpoints).fit(X[:120], np.arange(120) % 13)
        assert clf.predict_proba(X[120:]).shape == (40, 13)

    def test_model_property_before_fit(self) -> None:
        with pytest.raises(AttributeError, match="fit"):
            TabAbaxTabTuneClassifier().model

    def test_sklearn_params_round_trip(self) -> None:
        clf = TabAbaxTabTuneClassifier(n_estimators=3, tuning_strategy="peft")
        assert clone(clf).get_params()["n_estimators"] == 3
        assert clone(clf).get_params()["tuning_strategy"] == "peft"


class TestFineTuning:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_a_fixed_episode_is_learnable(self, checkpoints, frame, task) -> None:
        estimator = fitted(task, checkpoints, frame)
        model = estimator.model
        features, targets = support.ensemble_views(estimator)[0]
        support_rows, query_rows = np.arange(80), np.arange(80, 120)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        losses = []
        try:
            for _ in range(25):
                loss = support.episode_loss(model, features, targets, support_rows, query_rows, task, torch.device("cpu"))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                losses.append(float(loss.detach()))
        finally:
            model.eval()
        assert losses[-1] < 0.9 * losses[0]

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_lora_wraps_the_declared_layers_and_trains_adapters_only(self, checkpoints, frame, task) -> None:
        estimator = fitted(task, checkpoints, frame)
        support.apply_lora(estimator.model, {"r": 4, "lora_alpha": 8})
        wrapped = {n: m for n, m in estimator.model.named_modules() if isinstance(m, LoRALinear)}
        for target in TABABAX_LORA_TARGETS:
            assert any(target in n for n in wrapped), target
        assert not any(tok in n for n in wrapped for tok in TABABAX_LORA_EXCLUDE)
        for name, module in wrapped.items():
            if "out_proj" in name or "projection" in name:
                assert isinstance(module, FunctionalWeightLoRALinear), name
        trainable = [n for n, p in estimator.model.named_parameters() if p.requires_grad]
        assert trainable and all("lora" in n.lower() for n in trainable)

    @pytest.mark.parametrize("leaf", ["attn.out_proj", "in_linear.projection"])
    def test_adapters_on_functionally_read_weights_change_outputs(self, checkpoints, frame, leaf) -> None:
        X, _, _ = frame
        clf = fitted("classification", checkpoints, frame)
        support.apply_lora(clf.model, {"r": 4, "lora_alpha": 8})
        before = clf.predict_proba(X[120:])
        with torch.no_grad():
            for name, module in clf.model.named_modules():
                if isinstance(module, LoRALinear) and name.endswith(leaf):
                    module.lora_B.weight.normal_(0, 0.5)
        assert not np.allclose(clf.predict_proba(X[120:]), before)

    def test_requires_fit(self) -> None:
        with pytest.raises(RuntimeError, match="Fit"):
            finetune(TabAbaxTabTuneClassifier(), task="classification")


class TestValidatedLoop:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_report_and_selection(self, checkpoints, frame, task) -> None:
        estimator = fitted(task, checkpoints, frame)
        before = {k: v.clone() for k, v in estimator.model.state_dict().items()}
        finetune(estimator, task=task, params=VAL_PARAMS)
        report = estimator.finetune_report_
        assert report.model == "TabAbax" and report.strategy == "full"
        assert report.n_val > 0 and report.baseline is not None and np.isfinite(report.baseline)
        assert report.kept in ("fine-tuned", "pretrained")
        assert len(report.history) == report.epochs_run >= 1
        if report.kept == "pretrained":
            assert all(torch.equal(v, before[k]) for k, v in estimator.model.state_dict().items())
        else:
            assert report.best < report.baseline
        assert estimator.n_samples_in_ == 120 and not estimator.model.training

    def test_diverging_run_keeps_the_pretrained_weights(self, checkpoints, frame) -> None:
        X, _, _ = frame
        clf = fitted("classification", checkpoints, frame)
        before = clf.predict_proba(X[120:])
        finetune(clf, task="classification",
                 params={**VAL_PARAMS, "learning_rate": 1e6, "grad_clip": 0.0, "patience": 5})
        assert clf.finetune_report_.kept == "pretrained"
        np.testing.assert_allclose(clf.predict_proba(X[120:]), before, atol=1e-6)

    def test_finetuned_weights_survive_a_refit(self, checkpoints, frame) -> None:
        X, y, _ = frame
        clf = fitted("classification", checkpoints, frame)
        finetune(clf, task="classification", params={**FT_PARAMS, "validation_fraction": 0.0})
        weights = {k: v.clone() for k, v in clf.model.state_dict().items()}
        clf.fit(X[:120], y[:120])
        assert all(torch.equal(v, weights[k]) for k, v in clf.model.state_dict().items())

    def test_too_many_classes_for_finetuning(self, checkpoints, frame) -> None:
        X, _, _ = frame
        clf = make("classification", checkpoints).fit(X[:120], np.arange(120) % 13)
        with pytest.raises(ValueError, match="up to 10 classes"):
            finetune(clf, task="classification", params=FT_PARAMS)

    def test_second_finetune_starts_from_the_checkpoint(self, checkpoints, frame) -> None:
        from tabtune.TuningManager.tuning import TuningManager

        X, y, _ = frame
        clf = make("classification", checkpoints)
        manager = TuningManager()
        manager.tune(clf, X[:120], y[:120], strategy="finetune", params=dict(FT_PARAMS))
        first = clf.model_
        assert clf._tabtune_finetuned
        manager.tune(clf, X[:120], y[:120], strategy="finetune", params=dict(FT_PARAMS))
        assert clf.model_ is not first


class TestPersistence:
    @pytest.mark.parametrize("peft", [False, True])
    def test_pickle_keeps_finetuned_weights(self, checkpoints, frame, peft) -> None:
        X, _, _ = frame
        clf = fitted("classification", checkpoints, frame)
        finetune(clf, task="classification", params={**FT_PARAMS, "validation_fraction": 0.0},
                 peft_config={"r": 4, "lora_alpha": 8} if peft else None)
        restored = pickle.loads(pickle.dumps(clf))
        np.testing.assert_allclose(restored.predict_proba(X[120:]), clf.predict_proba(X[120:]), atol=1e-6)

    def test_pickle_of_an_untuned_estimator_reloads_the_checkpoint(self, checkpoints, frame) -> None:
        X, _, _ = frame
        clf = fitted("classification", checkpoints, frame)
        restored = pickle.loads(pickle.dumps(clf))
        np.testing.assert_allclose(restored.predict_proba(X[120:]), clf.predict_proba(X[120:]), atol=1e-6)


class TestPipeline:
    @staticmethod
    def build(checkpoints, task, strategy):
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        params = dict(FT_PARAMS)
        if strategy == "peft":
            params["peft_config"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        return TabularPipeline(
            model_name="TabAbax", task_type=task, tuning_strategy=strategy,
            model_params={"model_path": checkpoints[task], "allow_auto_download": False,
                          "device": "cpu", "n_estimators": 2},
            tuning_params=params,
        )

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_classification(self, checkpoints, frame, strategy) -> None:
        X, _, _ = frame
        y = pd.Series(np.where(X["f0"] + X["f1"] > 0, "yes", "no"))
        pipeline = self.build(checkpoints, "classification", strategy)
        pipeline.fit(X[:120], y[:120])
        assert set(np.unique(pipeline.predict(X[120:]))) <= {"yes", "no"}
        proba = pipeline.predict_proba(X[120:])
        assert proba.shape == (40, 2) and np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_regression(self, checkpoints, frame, strategy) -> None:
        X, _, y = frame
        pipeline = self.build(checkpoints, "regression", strategy)
        pipeline.fit(X[:120], pd.Series(y[:120]))
        prediction = np.asarray(pipeline.predict(X[120:]), dtype=float)
        assert prediction.shape == (40,) and np.isfinite(prediction).all()

    def test_save_and_load(self, checkpoints, frame, tmp_path) -> None:
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        X, y, _ = frame
        pipeline = self.build(checkpoints, "classification", "peft")
        pipeline.fit(X[:120], y[:120])
        path = tmp_path / "tababax.joblib"
        pipeline.save(str(path))
        loaded = TabularPipeline.load(str(path))
        np.testing.assert_allclose(loaded.predict_proba(X[120:]), pipeline.predict_proba(X[120:]), atol=1e-6)


class TestTuningManagerDispatch:
    @pytest.mark.parametrize("estimator, task", [
        (TabAbaxTabTuneClassifier, "classification"),
        (TabAbaxRegressorWrapper, "regression"),
    ])
    def test_peft_reaches_the_tababax_loop(self, monkeypatch, estimator, task) -> None:
        from tabtune.TuningManager import tuning as tuning_module

        seen = {}

        def fake(self, model, X, y, params=None, peft_config=None, task="classification"):
            seen.update(task=task, peft=peft_config)
            return model

        monkeypatch.setattr(tuning_module.TuningManager, "_finetune_tababax", fake)
        tuning_module.TuningManager().tune(
            estimator(), np.zeros((4, 2)), np.array([0, 1, 0, 1]),
            strategy="peft", params={"peft_config": {"r": 4}},
        )
        assert seen == {"task": task, "peft": {"r": 4}}
