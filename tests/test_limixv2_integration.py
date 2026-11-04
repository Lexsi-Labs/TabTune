"""LimiX-2 integration: vendored tree, estimators, fine-tuning, LoRA and pipeline.

Every test runs on a tiny randomly initialised LimiX-2 built from the real
architecture code, so nothing is downloaded.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

from tabtune.Dataprocess.data_processor import DataProcessor
from tabtune.Dataprocess.limixv2_preprocessor import LimiXV2Preprocessor
from tabtune.models.limixv2 import (
    LIMIXV2_LORA_TARGETS,
    LimiXV2Classifier,
    LimiXV2Regressor,
    finetune,
    resolve_checkpoint,
)
from tabtune.models.limixv2._upstream.convert_v20_ckpt import build_inference_config
from tabtune.models.limixv2._upstream.model.loading import build_model
from tabtune.models.regression.limixv2.regressor import LimiXV2RegressorWrapper
from tabtune.registry import MODEL_REGISTRY, get_model_spec, resolve_model_name
from tabtune.TuningManager.peft_utils import LoRALinear, MODEL_LORA_TARGETS

pytestmark = [pytest.mark.integration, pytest.mark.model_limixv2]

FT_PARAMS = {
    "epochs": 1, "steps_per_epoch": 2, "support_size": 60,
    "query_size": 30, "learning_rate": 1e-3,
}
VAL_PARAMS = {**FT_PARAMS, "epochs": 2, "min_validation_rows": 8, "patience": 1}


def tiny_config() -> dict:
    config = build_inference_config(
        {"config": {"emsize": 32, "nlayers": 2, "nhead": 2, "features_per_group": 2,
                    "rmsnorm_impl": "torch"}}
    )
    for key in ("encoder_config_x", "encoder_config_y"):
        config[key]["mask_token_emb_type"] = "uniform_mask_emb_type"
    config["encoder_config_y"].update(
        y_encoder_type="none_embedding", reg_y_numeric_embed_type="MLP_GELU"
    )
    config.update(
        feature_mse_decoder_type="MLP_GELU",
        feature_attention_pipeline="decoupled",
        dsti_use_qk_norm=True, dsti_use_softmax_scaling_mlp=True,
        dsti_y_token_aggregation="concat_linear",
        dsti_relation_dim=32, dsti_value_dim=32, dsti_num_heads=2,
        induce_use_qk_norm=True, induce_use_softmax_scaling_mlp=True,
        cls_y_decoder_type="Free_MLP", cls_y_decoder_mlp_pattern="linear-64_GELU_linear-10",
        reg_y_decoder_type="Free_MLP", reg_y_decoder_mlp_pattern="linear-64_GELU_linear-5000",
        num_buckets=5000, tf_attention_layer_type="bert", arch_version="2.0",
    )
    return config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    torch.manual_seed(0)
    config = tiny_config()
    state = build_model(config).state_dict()
    state["_reg_borders"] = torch.linspace(-5, 5, 5001)
    state["reg_log_widths"] = torch.log(state["_reg_borders"].diff())
    path = tmp_path_factory.mktemp("limixv2") / "tiny_v2.ckpt"
    torch.save({"config": config, "state_dict": state, "arch_version": "2.0"}, path)
    return str(path)


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 6)).astype(np.float32)
    y_cls = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.8)
    y_reg = X[:, 0] * 3 + X[:, 2]
    return X, y_cls, y_reg


def make(estimator, checkpoint, **kwargs):
    return estimator(model_path=checkpoint, device="cpu", enable_preprocess_parallel=False, **kwargs)


class TestRegistry:
    def test_spec(self) -> None:
        assert "LimiXv2" in MODEL_REGISTRY
        for alias in ("LimiX-2", "LimiX2", "LimiXV2"):
            assert resolve_model_name(alias) == "LimiXv2"
        spec = get_model_spec("LimiXv2")
        assert spec.classification_strategies == {"inference", "finetune", "peft"}
        assert spec.regression_strategies == {"inference", "finetune", "peft"}
        assert spec.envelope.max_classes == 10
        assert spec.license.commercial_use_ok is False
        assert "Built with StableAI LimiX" in spec.license.notes

    def test_v1_entry_is_untouched(self) -> None:
        assert resolve_model_name("LimiX") == "Limix"

    def test_data_processor_routes_to_passthrough(self) -> None:
        processor = DataProcessor(model_name="LimiXv2", task_type="classification")
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": list("xyxy")})
        processor.fit(X, pd.Series(["p", "q", "p", "q"]))
        assert isinstance(processor.custom_preprocessor_, LimiXV2Preprocessor)
        out = processor.transform(X)
        assert list(out.columns) == ["a", "b"] and np.isnan(out["a"].iloc[1])

    def test_lora_table(self) -> None:
        config = MODEL_LORA_TARGETS["LimiXv2"]
        assert tuple(config.target_substrings) == LIMIXV2_LORA_TARGETS
        assert not config.functional_weight_substrings


class TestVendoredTree:
    def test_runs_without_optional_gpu_dependencies(self, checkpoint) -> None:
        script = textwrap.dedent(f"""
            import sys, warnings
            for name in ("nvtx", "triton", "kditransform", "flash_attn"):
                sys.modules[name] = None
            import numpy as np, torch
            from tabtune.models.limixv2 import LimiXV2Classifier
            from tabtune.models.limixv2._upstream.model.operators.rmsnorm import HAS_TRITON, build_rmsnorm
            from tabtune.models.limixv2._upstream.model.operators.rmsnorm.torch_rmsnorm import RMSNormMixedPrecision
            assert not HAS_TRITON
            assert isinstance(build_rmsnorm(8, norm_impl="triton"), RMSNormMixedPrecision)
            rng = np.random.default_rng(0)
            X = rng.normal(size=(60, 4)); y = (X[:, 0] > 0).astype(int)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf = LimiXV2Classifier(model_path={checkpoint!r}, device="cpu",
                                        enable_preprocess_parallel=False).fit(X[:40], y[:40])
                proba = clf.predict_proba(X[40:])
            assert proba.shape == (20, 2) and np.allclose(proba.sum(1), 1)
            assert any("kditransform" in str(w.message) for w in caught)
            print("ok")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        assert result.returncode == 0, result.stderr[-3000:]
        assert "ok" in result.stdout

    def test_rmsnorm_implementations_share_state_dict_keys(self) -> None:
        pytest.importorskip("triton")
        from tabtune.models.limixv2._upstream.model.operators.rmsnorm import (
            TritonQKNorm, TritonRMSNorm,
        )
        from tabtune.models.limixv2._upstream.model.operators.rmsnorm.torch_rmsnorm import (
            RMSNormMixedPrecision,
        )

        reference = RMSNormMixedPrecision(16, eps=1e-5, elementwise_affine=True).state_dict()
        for cls in (TritonRMSNorm, TritonQKNorm):
            other = cls(hs=16, eps=1e-5, elementwise_affine=True).state_dict()
            assert {k: v.shape for k, v in other.items()} == {k: v.shape for k, v in reference.items()}

    def test_v1_checkpoint_is_rejected(self, tmp_path) -> None:
        from tabtune.models.limixv2._upstream.utils.loading import load_from_checkpoint

        with pytest.raises(ValueError, match="model_name='Limix'"):
            load_from_checkpoint({"config": {}, "state_dict": {}})

    def test_missing_checkpoint_path(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint(str(tmp_path / "absent.ckpt"), None, "LimiX-2.ckpt", None)


class TestEstimators:
    def test_classifier(self, checkpoint, table) -> None:
        X, y, _ = table
        labels = np.array(["low", "mid", "high"])[y]
        clf = make(LimiXV2Classifier, checkpoint).fit(X[:120], labels[:120])
        proba = clf.predict_proba(X[120:])
        assert proba.shape == (40, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert list(clf.classes_) == sorted(set(labels[:120]))
        assert set(clf.predict(X[120:])) <= set(labels)

    def test_classifier_accepts_mixed_frame(self, checkpoint, table) -> None:
        X, y, _ = table
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        frame["cat"] = np.random.default_rng(1).choice(list("abc"), size=len(frame))
        frame.loc[frame.index[:5], "f3"] = np.nan
        clf = make(LimiXV2Classifier, checkpoint).fit(frame[:120], y[:120])
        assert clf.predict_proba(frame[120:]).shape == (40, 3)

    def test_regressor(self, checkpoint, table) -> None:
        X, _, y = table
        reg = make(LimiXV2Regressor, checkpoint).fit(X[:120], y[:120])
        prediction = reg.predict(X[120:])
        assert prediction.shape == (40,)
        assert np.isfinite(prediction).all()

    def test_too_many_classes(self, checkpoint) -> None:
        X = np.random.default_rng(0).normal(size=(60, 3))
        with pytest.raises(ValueError, match="2 to 10 classes"):
            make(LimiXV2Classifier, checkpoint).fit(X, np.arange(60) % 11)

    def test_model_property_before_fit(self) -> None:
        clf = LimiXV2Classifier()
        assert not hasattr(clf, "model")

    def test_sklearn_params_round_trip(self) -> None:
        from sklearn.base import clone

        clf = LimiXV2Classifier(model_path="x.ckpt", seed=3, softmax_temperature=1.0)
        copy = clone(clf)
        assert copy.get_params()["seed"] == 3 and copy.model_path == "x.ckpt"


class TestFineTuning:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_full_finetune_updates_weights_and_predictions(self, checkpoint, table, task) -> None:
        X, y_cls, y_reg = table
        estimator = make(
            LimiXV2Classifier if task == "classification" else LimiXV2Regressor, checkpoint
        ).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        predict = estimator.predict_proba if task == "classification" else estimator.predict
        before = predict(X[120:])
        weights = {k: v.clone() for k, v in estimator.model.state_dict().items()}
        finetune(estimator, task=task, params=FT_PARAMS)
        changed = [k for k, v in estimator.model.state_dict().items() if not torch.equal(v, weights[k])]
        assert changed
        assert not estimator.model.training
        assert not np.allclose(predict(X[120:]), before)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_a_fixed_episode_is_learnable(self, checkpoint, table, task) -> None:
        from tabtune.models.limixv2 import tabtune_support as support
        from tabtune.models.limixv2._upstream.model.autobatch import AutobatchConfig

        X, y_cls, y_reg = table
        y = y_cls if task == "classification" else y_reg
        estimator = make(
            LimiXV2Classifier if task == "classification" else LimiXV2Regressor, checkpoint
        ).fit(X[:120], y[:120])
        predictor, model, device = estimator.predictor_, estimator.model, torch.device("cpu")
        predictor.reuse_frozen_model = False
        if task == "classification":
            x_, y_, classes, permutation = support._classification_member(
                predictor, X[:80], y[:80], X[80:120], 0, np.random.default_rng(1)
            )
            targets = torch.as_tensor(np.searchsorted(classes, y[80:120]))

            def loss_fn(output):
                return support._classification_loss(
                    output, predictor.model_config, len(classes), permutation, targets
                )
        else:
            x_, y_, query = support._regression_member(predictor, X[:80], y[:80], X[80:120], y[80:120], 0)
            targets = torch.as_tensor(query, dtype=torch.float32)

            def loss_fn(output):
                return support._regression_loss(output, model, targets)

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        autobatch = AutobatchConfig.ENABLE_AUTOBATCH
        AutobatchConfig.ENABLE_AUTOBATCH = False
        model.train()
        losses = []
        try:
            for _ in range(30):
                loss = loss_fn(support._forward(model, x_, y_, 80, estimator._upstream_task, device, None))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                losses.append(float(loss.detach()))
        finally:
            AutobatchConfig.ENABLE_AUTOBATCH = autobatch
            model.eval()
        assert losses[-1] < 0.9 * losses[0]

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_lora_trains_adapters_only(self, checkpoint, table, task) -> None:
        X, y_cls, y_reg = table
        estimator = make(
            LimiXV2Classifier if task == "classification" else LimiXV2Regressor, checkpoint
        ).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        predict = estimator.predict_proba if task == "classification" else estimator.predict
        before = predict(X[120:])
        finetune(estimator, task=task, params=FT_PARAMS, peft_config={"r": 4, "lora_alpha": 8})

        wrapped = {n for n, m in estimator.model.named_modules() if isinstance(m, LoRALinear)}
        for target in LIMIXV2_LORA_TARGETS:
            assert any(n.endswith(target) for n in wrapped), target
        assert not any("scale_linear" in n for n in wrapped)
        trainable = [n for n, p in estimator.model.named_parameters() if p.requires_grad]
        assert trainable and all("lora" in n.lower() for n in trainable)
        assert not np.allclose(predict(X[120:]), before)

    def test_requires_fit(self) -> None:
        with pytest.raises(RuntimeError, match="Fit"):
            finetune(LimiXV2Classifier(), task="classification")


class TestPipeline:
    @staticmethod
    def frame():
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(160, 5)), columns=[f"f{i}" for i in range(5)])
        X["cat"] = rng.choice(list("abc"), size=160)
        X.loc[X.index[:5], "f3"] = np.nan
        y_cls = pd.Series(np.where(X["f0"] + X["f1"] > 0, "yes", "no"))
        y_reg = pd.Series(X["f0"] * 2 + X["f1"] + rng.normal(scale=0.1, size=160))
        return X, y_cls, y_reg

    @staticmethod
    def build(checkpoint, task, strategy):
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        params = dict(FT_PARAMS)
        if strategy == "peft":
            params["peft_config"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        return TabularPipeline(
            model_name="LimiXv2", task_type=task, tuning_strategy=strategy,
            finetune_mode="meta-learning",
            model_params={"model_path": checkpoint, "device": "cpu",
                          "enable_preprocess_parallel": False},
            tuning_params=params,
        )

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_classification(self, checkpoint, strategy) -> None:
        X, y, _ = self.frame()
        pipeline = self.build(checkpoint, "classification", strategy)
        pipeline.fit(X[:120], y[:120])
        assert set(np.unique(pipeline.predict(X[120:]))) <= {"yes", "no"}
        proba = pipeline.predict_proba(X[120:])
        assert proba.shape == (40, 2) and np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_regression(self, checkpoint, strategy) -> None:
        X, _, y = self.frame()
        pipeline = self.build(checkpoint, "regression", strategy)
        pipeline.fit(X[:120], y[:120])
        prediction = np.asarray(pipeline.predict(X[120:]), dtype=float)
        assert prediction.shape == (40,) and np.isfinite(prediction).all()


class TestTuningManagerDispatch:
    @pytest.mark.parametrize("estimator, task", [
        (LimiXV2Classifier, "classification"),
        (LimiXV2RegressorWrapper, "regression"),
    ])
    def test_peft_reaches_the_limixv2_loop(self, monkeypatch, estimator, task) -> None:
        from tabtune.TuningManager import tuning as tuning_module

        seen = {}

        def fake(self, model, X, y, params=None, peft_config=None, task="classification"):
            seen.update(task=task, peft=peft_config)
            return model

        monkeypatch.setattr(tuning_module.TuningManager, "_finetune_limixv2", fake)
        tuning_module.TuningManager().tune(
            estimator(), np.zeros((4, 2)), np.array([0, 1, 0, 1]),
            strategy="peft", params={"peft_config": {"r": 4}},
        )
        assert seen == {"task": task, "peft": {"r": 4}}


class TestValidatedLoop:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_report_and_selection(self, checkpoint, table, task) -> None:
        X, y_cls, y_reg = table
        estimator = make(
            LimiXV2Classifier if task == "classification" else LimiXV2Regressor, checkpoint
        ).fit(X[:120], (y_cls if task == "classification" else y_reg)[:120])
        before = {k: v.clone() for k, v in estimator.model.state_dict().items()}
        finetune(estimator, task=task, params=VAL_PARAMS)
        report = estimator.finetune_report_
        assert report.model == "LimiXv2" and report.strategy == "full"
        assert report.n_val > 0 and report.baseline is not None and np.isfinite(report.baseline)
        assert report.kept in ("fine-tuned", "pretrained")
        assert len(report.history) == report.epochs_run >= 1 and "val" in report.history[0]
        if report.kept == "pretrained":
            assert all(torch.equal(v, before[k]) for k, v in estimator.model.state_dict().items())
        else:
            assert report.best < report.baseline
        assert len(estimator.X_train_) == 120 and len(estimator.y_train_) == 120
        assert not estimator.model.training
        assert estimator.predictor_.reuse_frozen_model is False

    def test_diverging_run_keeps_the_pretrained_weights(self, checkpoint, table) -> None:
        X, y, _ = table
        clf = make(LimiXV2Classifier, checkpoint).fit(X[:120], y[:120])
        before = clf.predict_proba(X[120:])
        finetune(clf, task="classification", params={**VAL_PARAMS, "learning_rate": 1e6, "grad_clip": 0.0, "patience": 5})
        assert clf.finetune_report_.kept == "pretrained"
        np.testing.assert_allclose(clf.predict_proba(X[120:]), before, atol=1e-5)

    def test_lora_report(self, checkpoint, table) -> None:
        X, y, _ = table
        clf = make(LimiXV2Classifier, checkpoint).fit(X[:120], y[:120])
        finetune(clf, task="classification", params=VAL_PARAMS, peft_config={"r": 4, "lora_alpha": 8})
        report = clf.finetune_report_
        assert report.strategy == "lora" and report.trainable_params < report.total_params

    def test_second_finetune_starts_from_the_checkpoint(self, checkpoint, table) -> None:
        from tabtune.TuningManager.tuning import TuningManager

        X, y, _ = table
        clf = make(LimiXV2Classifier, checkpoint)
        manager = TuningManager()
        manager.tune(clf, X[:120], y[:120], strategy="finetune", params=dict(FT_PARAMS))
        first = clf.predictor_
        assert clf._tabtune_finetuned
        manager.tune(clf, X[:120], y[:120], strategy="finetune", params=dict(FT_PARAMS))
        assert clf.predictor_ is not first
