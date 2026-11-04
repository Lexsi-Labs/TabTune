"""Kumo Tabular integration: vendored sdm tree, estimators, fine-tuning, LoRA and pipeline.

Every test runs on a tiny randomly initialised KumoTabular built from the real
architecture code, so nothing is downloaded.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from tabtune.Dataprocess.data_processor import DataProcessor
from tabtune.Dataprocess.kumo_preprocessor import KumoTabularPreprocessor
from tabtune.models.kumo import (
    KUMO_LORA_TARGETS,
    KumoTabularClassifier,
    KumoTabularRegressor,
    finetune,
    resolve_checkpoint,
)
from tabtune.models.kumo import tabtune_support as support
from tabtune.models.kumo._sdm.models import KumoTabular
from tabtune.models.kumo._sdm.models.kumo.tabular import model as kumo_model
from tabtune.models.regression.kumo.regressor import KumoTabularRegressorWrapper
from tabtune.registry import MODEL_REGISTRY, get_model_spec, resolve_model_name
from tabtune.TuningManager.peft_utils import (
    MODEL_LORA_TARGETS,
    FunctionalWeightLoRALinear,
    LoRALinear,
)

pytestmark = [pytest.mark.integration, pytest.mark.model_kumo]

TINY = dict(
    cell_channels=16, num_embedding_layers=1, num_embedding_heads=2,
    num_inducing_points=8, group_size=3, num_frequencies=4, num_readout_tokens=2,
    icl_channels=32, num_icl_layers=2, num_icl_heads=2,
    num_icl_key_value_heads_for_query=None,
)
FT_PARAMS = {
    "epochs": 1, "steps_per_epoch": 2, "support_size": 60,
    "query_size": 30, "learning_rate": 1e-3,
}
VAL_PARAMS = {**FT_PARAMS, "epochs": 2, "min_validation_rows": 8, "patience": 1}


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory) -> str:
    kumo_model.MODEL_KWARGS["tiny"] = TINY
    root = tmp_path_factory.mktemp("kumo")
    (root / "tiny").mkdir()
    for task, name in (("classification", "classifier"), ("regression", "regressor")):
        torch.manual_seed(0)
        model = KumoTabular(task=task, size="tiny", pretrained=False)
        # Residual branches are zero-initialised; randomise them so outputs
        # depend on the inputs, as upstream's own tests do.
        for parameter in model.parameters():
            if not parameter.any():
                torch.nn.init.normal_(parameter, std=0.02)
        torch.save(model.models[task].state_dict(), root / "tiny" / f"{name}.pt")
    yield str(root)
    kumo_model.MODEL_KWARGS.pop("tiny", None)


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 5)), columns=[f"f{i}" for i in range(5)])
    X["cat"] = rng.choice(list("abc"), size=160)
    X.loc[X.index[:5], "f3"] = np.nan
    y_cls = np.array(["low", "mid", "high"])[(X["f0"] > 0).astype(int) + (X["f1"] > 0.8)]
    y_reg = (X["f0"] * 3 + X["f2"]).to_numpy()
    return X, y_cls, y_reg


def make(estimator, checkpoint_dir, **kwargs):
    kwargs.setdefault("num_estimators", 2)
    return estimator(size="tiny", model_path=checkpoint_dir, device="cpu", **kwargs)


class TestRegistry:
    def test_spec(self) -> None:
        assert "KumoTabular" in MODEL_REGISTRY
        for alias in ("Kumo-Tabular", "Kumo", "NVIDIA-Kumo-Tabular"):
            assert resolve_model_name(alias) == "KumoTabular"
        spec = get_model_spec("KumoTabular")
        assert spec.classification_strategies == {"inference", "finetune", "peft"}
        assert spec.regression_strategies == {"inference", "finetune", "peft"}
        assert spec.license.commercial_use_ok is True
        assert spec.weights == "nvidia/Kumo-Tabular"

    def test_data_processor_routes_to_passthrough(self) -> None:
        processor = DataProcessor(model_name="KumoTabular", task_type="classification")
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": list("xyxy")})
        processor.fit(X, pd.Series(["p", "q", "p", "q"]))
        assert isinstance(processor.custom_preprocessor_, KumoTabularPreprocessor)
        out = processor.transform(X)
        assert list(out.columns) == ["a", "b"] and np.isnan(out["a"].iloc[1])

    def test_lora_table(self) -> None:
        config = MODEL_LORA_TARGETS["KumoTabular"]
        assert tuple(config.target_substrings) == KUMO_LORA_TARGETS
        assert tuple(config.functional_weight_substrings) == ("qkv_lin",)


class TestVendoredTree:
    def test_no_optional_gpu_dependencies_are_imported(self) -> None:
        import subprocess

        script = (
            "import sys, tabtune; before = set(sys.modules);"
            "import tabtune.models.kumo; new = set(sys.modules) - before;"
            "bad = [m for m in ('cudf', 'cupy', 'pyg_lib', 'pylibcugraph', 'sentence_transformers')"
            " if m in new];"
            "print(bad)"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().endswith("[]")

    def test_strenum_shim(self) -> None:
        from tabtune.models.kumo._sdm import Stype, Task

        assert str(Task.classification) == "classification"
        assert f"{Stype.numerical}" == "numerical"
        assert Task("regression") is Task.regression

    def test_checkpoint_resolution(self, checkpoint_dir, tmp_path) -> None:
        path = resolve_checkpoint("regression", "tiny", checkpoint_dir, "", None, None)
        assert path.endswith("tiny/regressor.pt")
        assert resolve_checkpoint("regression", "tiny", path, "", None, None) == path
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint("classification", "tiny", str(tmp_path), "", None, None)


class TestEstimators:
    def test_classifier(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120])
        proba = clf.predict_proba(X[120:])
        assert proba.shape == (40, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert list(clf.classes_) == sorted(set(y[:120]))
        assert set(clf.predict(X[120:])) <= set(y)

    def test_many_classes_use_ecoc(self, checkpoint_dir, frame) -> None:
        X, _, _ = frame
        y = np.arange(len(X)) % 13
        proba = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120]).predict_proba(X[120:])
        assert proba.shape == (40, 13) and np.allclose(proba.sum(axis=1), 1.0)

    def test_query_batching_is_exact(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        full = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120]).predict_proba(X[120:])
        chunked = make(KumoTabularClassifier, checkpoint_dir, query_batch_size=7)
        chunked.fit(X[:120], y[:120])
        np.testing.assert_allclose(chunked.predict_proba(X[120:]), full, atol=1e-6)

    def test_max_context_size_subsamples_per_estimator(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir, max_context_size=50).fit(X[:120], y[:120])
        proba = clf.predict_proba(X[120:])
        assert proba.shape == (40, 3) and np.isfinite(proba).all()

    def test_datetime_and_numpy_input(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        X = X.copy()
        X["when"] = pd.date_range("2020-01-01", periods=len(X), freq="D")
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120])
        assert clf.predict_proba(X[120:]).shape == (40, 3)
        numeric = X.select_dtypes("number").to_numpy()
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(numeric[:120], y[:120])
        assert clf.predict_proba(numeric[120:]).shape == (40, 3)

    def test_predict_accepts_positional_columns(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        numeric = X.drop(columns=["cat"])
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(numeric[:120], y[:120])
        np.testing.assert_allclose(
            clf.predict_proba(numeric[120:].to_numpy()), clf.predict_proba(numeric[120:])
        )
        with pytest.raises(ValueError, match="features"):
            clf.predict_proba(numeric[120:].to_numpy()[:, :3])

    def test_regressor(self, checkpoint_dir, frame) -> None:
        X, _, y = frame
        reg = make(KumoTabularRegressor, checkpoint_dir).fit(X[:120], y[:120])
        quantiles = reg.predict_quantiles(X[120:])
        assert quantiles.shape == (40, 999)
        assert (np.diff(quantiles, axis=1) >= -1e-6).all()
        np.testing.assert_allclose(reg.predict(X[120:]), quantiles.mean(axis=1))

    def test_model_property_before_fit(self) -> None:
        assert not hasattr(KumoTabularClassifier(), "model")

    def test_sklearn_params_round_trip(self) -> None:
        copy = clone(KumoTabularClassifier(size="small", num_estimators=3, seed=7))
        assert copy.get_params()["num_estimators"] == 3 and copy.size == "small"

    def test_default_estimators_follow_size(self) -> None:
        assert KumoTabularClassifier(size="large").n_estimators_ == 16
        assert KumoTabularClassifier(size="small").n_estimators_ == 8


class TestFineTuning:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_full_finetune_updates_weights_and_predictions(self, checkpoint_dir, frame, task) -> None:
        X, y_cls, y_reg = frame
        estimator_cls = KumoTabularClassifier if task == "classification" else KumoTabularRegressor
        estimator = make(estimator_cls, checkpoint_dir).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        predict = estimator.predict_proba if task == "classification" else estimator.predict
        before = predict(X[120:])
        weights = {k: v.clone() for k, v in estimator.model.state_dict().items()}
        finetune(estimator, task=task, params=FT_PARAMS)
        assert any(not torch.equal(v, weights[k]) for k, v in estimator.model.state_dict().items())
        assert not estimator.kumo_.training
        assert not np.allclose(predict(X[120:]), before)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_a_fixed_episode_is_learnable(self, checkpoint_dir, frame, task) -> None:
        X, y_cls, y_reg = frame
        estimator_cls = KumoTabularClassifier if task == "classification" else KumoTabularRegressor
        estimator = make(estimator_cls, checkpoint_dir).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        kumo, x, y = estimator.kumo_, estimator.x_context_, estimator.y_context_
        recipe = support.train_recipe(kumo)
        scale = float(y.numerical.float().std()) if task == "regression" else 1.0
        optimiser = torch.optim.Adam(kumo.parameters(), lr=1e-3)
        kumo.train()
        losses = []
        try:
            for _ in range(25):
                loss = support.episode_loss(
                    kumo, task, recipe, x[:80], y[:80], x[80:], y[80:], scale,
                    generator=torch.Generator().manual_seed(0),
                )
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                losses.append(float(loss.detach()))
        finally:
            kumo.eval()
        assert losses[-1] < 0.9 * losses[0]

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_lora_trains_adapters_only(self, checkpoint_dir, frame, task) -> None:
        X, y_cls, y_reg = frame
        estimator_cls = KumoTabularClassifier if task == "classification" else KumoTabularRegressor
        estimator = make(estimator_cls, checkpoint_dir).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        predict = estimator.predict_proba if task == "classification" else estimator.predict
        before = predict(X[120:])
        finetune(estimator, task=task, params=FT_PARAMS, peft_config={"r": 4, "lora_alpha": 8})

        wrapped = {n: m for n, m in estimator.model.named_modules() if isinstance(m, LoRALinear)}
        for target in KUMO_LORA_TARGETS:
            assert any(target in n for n in wrapped), target
        assert all(isinstance(m, FunctionalWeightLoRALinear) for n, m in wrapped.items() if "qkv_lin" in n)
        assert not any(tok in n for n in wrapped for tok in ("cell_embedding", "y_lin", "query_scaling"))
        trainable = [n for n, p in estimator.model.named_parameters() if p.requires_grad]
        assert trainable and all("lora" in n.lower() for n in trainable)
        assert not np.allclose(predict(X[120:]), before)

    def test_qkv_adapters_reach_the_cross_attention_path(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120])
        support.apply_lora(clf.model, {"r": 4, "lora_alpha": 8})
        before = clf.predict_proba(X[120:])
        with torch.no_grad():
            for name, module in clf.model.named_modules():
                if isinstance(module, LoRALinear) and "icl_block" in name and "qkv_lin" in name:
                    module.lora_B.weight.normal_(0, 0.5)
        assert not np.allclose(clf.predict_proba(X[120:]), before)

    def test_requires_fit(self) -> None:
        with pytest.raises(RuntimeError, match="Fit"):
            finetune(KumoTabularClassifier(), task="classification")


class TestPipeline:
    @staticmethod
    def build(checkpoint_dir, task, strategy):
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        params = dict(FT_PARAMS)
        if strategy == "peft":
            params["peft_config"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        return TabularPipeline(
            model_name="KumoTabular", task_type=task, tuning_strategy=strategy,
            finetune_mode="meta-learning",
            model_params={"size": "tiny", "model_path": checkpoint_dir,
                          "device": "cpu", "num_estimators": 2},
            tuning_params=params,
        )

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_classification(self, checkpoint_dir, frame, strategy) -> None:
        X, _, _ = frame
        y = pd.Series(np.where(X["f0"] + X["f1"] > 0, "yes", "no"))
        pipeline = self.build(checkpoint_dir, "classification", strategy)
        pipeline.fit(X[:120], y[:120])
        assert set(np.unique(pipeline.predict(X[120:]))) <= {"yes", "no"}
        proba = pipeline.predict_proba(X[120:])
        assert proba.shape == (40, 2) and np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_regression(self, checkpoint_dir, frame, strategy) -> None:
        X, _, y = frame
        pipeline = self.build(checkpoint_dir, "regression", strategy)
        pipeline.fit(X[:120], pd.Series(y[:120]))
        prediction = np.asarray(pipeline.predict(X[120:]), dtype=float)
        assert prediction.shape == (40,) and np.isfinite(prediction).all()


class TestTuningManagerDispatch:
    @pytest.mark.parametrize("estimator, task", [
        (KumoTabularClassifier, "classification"),
        (KumoTabularRegressorWrapper, "regression"),
    ])
    def test_peft_reaches_the_kumo_loop(self, monkeypatch, estimator, task) -> None:
        from tabtune.TuningManager import tuning as tuning_module

        seen = {}

        def fake(self, model, X, y, params=None, peft_config=None, task="classification"):
            seen.update(task=task, peft=peft_config)
            return model

        monkeypatch.setattr(tuning_module.TuningManager, "_finetune_kumo", fake)
        tuning_module.TuningManager().tune(
            estimator(), np.zeros((4, 2)), np.array([0, 1, 0, 1]),
            strategy="peft", params={"peft_config": {"r": 4}},
        )
        assert seen == {"task": task, "peft": {"r": 4}}


class TestValidatedLoop:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_report_and_selection(self, checkpoint_dir, frame, task) -> None:
        X, y_cls, y_reg = frame
        estimator_cls = KumoTabularClassifier if task == "classification" else KumoTabularRegressor
        estimator = make(estimator_cls, checkpoint_dir).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        before = {k: v.clone() for k, v in estimator.model.state_dict().items()}
        finetune(estimator, task=task, params=VAL_PARAMS)
        report = estimator.finetune_report_
        assert report.model == "KumoTabular" and report.strategy == "full"
        assert report.n_val > 0 and report.baseline is not None and np.isfinite(report.baseline)
        assert report.kept in ("fine-tuned", "pretrained")
        assert len(report.history) == report.epochs_run >= 1 and "val" in report.history[0]
        if report.kept == "pretrained":
            assert all(torch.equal(v, before[k]) for k, v in estimator.model.state_dict().items())
        else:
            assert report.best < report.baseline
        assert estimator.x_context_.size(0) == 120 and estimator.y_context_.size(0) == 120
        assert not estimator.kumo_.training

    def test_diverging_run_keeps_the_pretrained_weights(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120])
        before = clf.predict_proba(X[120:])
        finetune(clf, task="classification", params={**VAL_PARAMS, "learning_rate": 1e6, "grad_clip": 0.0, "patience": 5})
        assert clf.finetune_report_.kept == "pretrained"
        np.testing.assert_allclose(clf.predict_proba(X[120:]), before, atol=1e-6)

    def test_lora_report(self, checkpoint_dir, frame) -> None:
        X, y, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], y[:120])
        finetune(clf, task="classification", params=VAL_PARAMS, peft_config={"r": 4, "lora_alpha": 8})
        report = clf.finetune_report_
        assert report.strategy == "lora" and report.trainable_params < report.total_params
        assert report.config["learning_rate"] == 1e-3

    def test_too_many_classes_for_finetuning(self, checkpoint_dir, frame) -> None:
        X, _, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir).fit(X[:120], np.arange(120) % 13)
        with pytest.raises(ValueError, match="up to 10 classes"):
            finetune(clf, task="classification", params=FT_PARAMS)

    def test_second_finetune_starts_from_the_checkpoint(self, checkpoint_dir, frame) -> None:
        from tabtune.TuningManager.tuning import TuningManager

        X, y, _ = frame
        clf = make(KumoTabularClassifier, checkpoint_dir)
        manager = TuningManager()
        manager.tune(clf, X[:120], y[:120], strategy="finetune", params=dict(FT_PARAMS))
        first = clf.kumo_
        assert clf._tabtune_finetuned
        manager.tune(clf, X[:120], y[:120], strategy="finetune", params=dict(FT_PARAMS))
        assert clf.kumo_ is not first
