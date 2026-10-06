"""Integration tests for Xiaomi TabLDM in TabTune.

No checkpoint is downloaded. A tiny model with the same architecture is built
and saved in the shape ``_load_model`` expects, so the real estimator code path
runs end to end: checkpoint load, preprocessing, ensemble, forward, fine-tuning
and LoRA.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tabtune.Dataprocess.data_processor import DataProcessor
from tabtune.Dataprocess.tabldm_preprocessor import TabLDMPreprocessor
from tabtune.TuningManager.peft_utils import (
    FunctionalWeightLoRALinear,
    LoRALinear,
    MODEL_LORA_TARGETS,
    inject_custom_lora_into_linear_layers,
    resolve_lora_targets,
)
from tabtune.models.tabldm import (
    TABLDM_LORA_EXCLUDE,
    TABLDM_LORA_TARGETS,
    VENDORED_FROM,
    TabLDMTabTuneClassifier,
    TabLDMTabTuneRegressor,
)
from tabtune.models.tabldm._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabtune.models.tabldm._model.embedding_dual_stream import ColEmbeddingDualStream
from tabtune.models.tabldm.tabtune_support import (
    TABLDM_FUNCTIONAL_WEIGHT_LEAVES,
    _ensemble_members,
    _episode_loss,
    apply_lora,
    finetune,
    iter_episodes,
    torch_module,
)
from tabtune.models.regression.tabldm.regressor import TabLDMRegressorWrapper
from tabtune.registry import MODEL_REGISTRY, get_model_spec

ALIASES = ("TabLDM", "tabldm", "Xiaomi-TabLDM", "xiaomi_tabldm", "TabLDM-v1", "tab ldm")


def tiny_config(max_classes: int) -> dict:
    """A miniature of the real checkpoint config, small enough to run on CPU."""
    return dict(
        max_classes=max_classes, num_quantiles=17, embed_dim=16, col_num_blocks=1,
        col_nhead=2, col_num_inds=4, col_affine=False, col_feature_group="same",
        col_feature_group_size=3, col_target_aware=True, col_ssmax=False,
        row_num_blocks=2, row_nhead=2, row_num_cls=2, icl_num_blocks=2, icl_nhead=2,
        ff_factor=1, dropout=0.0, activation="gelu", norm_first=True,
        bias_free_ln=True, zero_init=False, recompute=False,
        moe_num_experts=2, moe_top_k=1, moe_num_shared_experts=1,
    )


def tiny_model(max_classes: int = 4, seed: int = 0) -> nn.Module:
    """Build the miniature model exactly as ``_load_model`` assembles the real one."""
    torch.manual_seed(seed)
    config = tiny_config(max_classes)
    model = TabLDMSparseMoE(**config)
    model.col_embedder = ColEmbeddingDualStream(
        embed_dim=16, num_blocks=1, nhead=2, dim_feedforward=16, num_inds=4,
        dropout=0.0, activation="gelu", norm_first=True, bias_free_ln=True,
        affine=False, feature_group="same", feature_group_size=3,
        global_dilation="adaptive", global_max_span=32, target_aware=True,
        max_classes=max_classes, reserve_cls_tokens=2, ssmax=False,
        zero_init=False, mixed_radix_ensemble=True, recompute=False,
    )
    model.drop_dense_ffn()
    for parameter in model.parameters():
        nn.init.normal_(parameter, 0.0, 0.02)
    return model


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, str]:
    """A classification and a regression checkpoint on disk."""
    directory = tmp_path_factory.mktemp("tabldm_ckpt")
    paths = {}
    for task, max_classes in (("classification", 4), ("regression", 0)):
        model = tiny_model(max_classes)
        path = directory / f"{task}.ckpt"
        torch.save(
            {"config": tiny_config(max_classes), "state_dict": model.state_dict()}, path
        )
        paths[task] = str(path)
    return paths


@pytest.fixture
def table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(140, 6))
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] * 2.0 + X[:, 1] + rng.normal(scale=0.1, size=140)
    return X, y_cls, y_reg


def make_classifier(path: str, **kwargs):
    return TabLDMTabTuneClassifier(
        model_path=path, allow_auto_download=False, n_estimators=2, device="cpu",
        random_state=0, use_amp=False, enhance_candidates=False, **kwargs
    )


def make_regressor(path: str, **kwargs):
    return TabLDMRegressorWrapper(
        model_path=path, allow_auto_download=False, n_estimators=2, device="cpu",
        random_state=0, use_amp=False, enhance_candidates=False, **kwargs
    )


class TestRegistry:
    def test_spec_registered(self) -> None:
        spec = get_model_spec("TabLDM")
        assert spec.name == "TabLDM"
        assert spec.family == "icl"
        assert spec.preprocessor_key == "tabldm_special"

    @pytest.mark.parametrize("alias", ALIASES)
    def test_aliases_resolve(self, alias: str) -> None:
        assert get_model_spec(alias).name == "TabLDM"

    def test_supports_both_tasks_with_finetune_and_peft(self) -> None:
        spec = get_model_spec("TabLDM")
        for task in ("classification", "regression"):
            assert {"inference", "finetune", "peft"} <= spec.strategies_for(task)

    def test_no_unverified_envelope_limits_are_declared(self) -> None:
        """max_classes and min_rows are HARD constraints; TabLDM has neither.

        Its head has a 10-class capacity but handles more by hierarchical
        grouping, and max_num_features subsamples rather than rejecting. A
        declared limit would refuse datasets the model handles by design.
        """
        envelope = get_model_spec("TabLDM").envelope
        assert envelope.max_classes is None
        assert envelope.max_features is None
        assert envelope.max_rows is None
        # 2 is CapabilityEnvelope's universal default, i.e. nothing declared.
        assert envelope.min_rows == 2
        assert envelope.native_nan is True
        assert envelope.native_categorical is True

    def test_weight_license_is_recorded_as_unverified(self) -> None:
        # The code is Apache-2.0, but upstream's statements about the weights conflict
        # (0.4.0 license check), so commercial use is neither granted nor refused.
        spec = get_model_spec("TabLDM")
        assert "Apache-2.0" in spec.license.name
        assert spec.license.commercial_use_ok is None

    def test_listed(self) -> None:
        assert "TabLDM" in MODEL_REGISTRY


class TestVendoredTree:
    def test_version_is_pinned_not_read_from_metadata(self) -> None:
        assert VENDORED_FROM == "0.1.0"

    def test_no_bare_upstream_import_survives(self) -> None:
        """Every ``tabldm.`` import must point at the vendored root.

        A surviving bare import would resolve to a co-installed upstream
        ``tabldm`` distribution, or fail outright when none is installed.
        """
        import pathlib

        import tabtune.models.tabldm as package

        root = pathlib.Path(package.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("from tabldm", "import tabldm")):
                    offenders.append(f"{path.relative_to(root)}:{number}")
        assert offenders == []

    def test_package_root_does_not_squat_a_global_module_name(self) -> None:
        import sys

        assert "tabldm" not in sys.modules or sys.modules["tabldm"].__name__ == "tabldm"

    def test_licence_is_vendored_alongside_the_code(self) -> None:
        import pathlib

        import tabtune.models.tabldm as package

        licence = pathlib.Path(package.__file__).parent / "LICENSE"
        assert licence.exists()
        assert "Apache License" in licence.read_text()


class TestWiring:
    def test_preprocessor_registered(self) -> None:
        processor = DataProcessor(model_name="TabLDM", task_type="classification")
        assert isinstance(processor._get_custom_preprocessor(), TabLDMPreprocessor)

    def test_preprocessor_passes_features_through(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
        preprocessor = TabLDMPreprocessor(task_type="classification")
        out = preprocessor.fit(frame, np.array(["p", "q", "p"])).transform(frame)
        pd.testing.assert_frame_equal(out, frame)
        assert list(preprocessor.transform_target(np.array(["p", "q", "p"]))) == [0, 1, 0]

    def test_regression_target_is_not_scaled(self) -> None:
        """TabLDM predicts in the original space and nothing inverse-transforms.

        The pipeline never undoes a regression target scaling, so any scaling
        chosen here would leave predictions in a space the caller never
        reverses.
        """
        processor = DataProcessor(model_name="TabLDM", task_type="regression")
        assert processor._get_regression_processor().target_scaling_strategy == "none"

    def test_estimators_accept_tabtune_kwargs(self) -> None:
        for estimator in (
            TabLDMTabTuneClassifier(tuning_strategy="finetune", task_type="classification"),
            TabLDMTabTuneRegressor(tuning_strategy="peft", task_type="regression"),
        ):
            assert estimator.tuning_strategy in ("finetune", "peft")

    def test_regression_wrapper_rejects_an_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="inference"):
            TabLDMRegressorWrapper(tuning_strategy="distill")

    def test_torch_module_requires_a_fit(self) -> None:
        with pytest.raises(RuntimeError, match="call fit"):
            torch_module(TabLDMTabTuneClassifier())

    def test_model_property_is_absent_rather_than_exploding(self) -> None:
        """TabTune probes ``hasattr(model, 'model')``; hasattr only eats AttributeError."""
        estimator = TabLDMTabTuneClassifier()
        assert not hasattr(estimator, "model")
        with pytest.raises(AttributeError, match="call fit"):
            estimator.model

    def test_finetune_requires_a_fitted_ensemble(self) -> None:
        with pytest.raises(RuntimeError, match="call fit"):
            _ensemble_members(TabLDMTabTuneClassifier())


class TestLoraTargets:
    def test_every_target_matches_a_real_linear_leaf(self) -> None:
        model = tiny_model()
        leaves = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name
        ]
        for token in TABLDM_LORA_TARGETS:
            assert any(token in leaf for leaf in leaves), f"{token} matches nothing"

    def test_exclusions_match_real_leaves_too(self) -> None:
        """An exclusion that matches nothing is a comment, not a guard."""
        model = tiny_model()
        leaves = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name
        ]
        for token in TABLDM_LORA_EXCLUDE:
            assert any(token in leaf for leaf in leaves), f"{token} matches nothing"

    def test_peft_table_agrees_with_the_support_module(self) -> None:
        assert MODEL_LORA_TARGETS["TabLDM"].target_substrings == TABLDM_LORA_TARGETS

    def test_moe_experts_are_reached(self) -> None:
        model = tiny_model()
        resolved = resolve_lora_targets("TabLDM", model)
        assert any("moe_ffn.experts" in name for name in resolved)
        assert any("shared_experts" in name for name in resolved)

    def test_router_and_y_encoder_are_excluded(self) -> None:
        model = tiny_model()
        apply_lora(model, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0})
        wrapped = [
            name for name, module in model.named_modules()
            if isinstance(module, LoRALinear)
        ]
        assert wrapped, "nothing was wrapped"
        assert not [n for n in wrapped if "router" in n or "y_encoder" in n]


class TestFunctionalWeightLora:
    """``out_proj`` is read as a tensor, not called; a plain wrapper is a no-op."""

    @staticmethod
    def _episode():
        torch.manual_seed(1)
        return torch.randn(1, 40, 6), torch.randint(0, 4, (1, 30)).float()

    def test_out_proj_weight_is_consumed_functionally(self) -> None:
        """Pin the upstream fact the whole wrapper exists for."""
        import inspect

        from tabtune.models.tabldm._model import layers

        source = inspect.getsource(layers.MultiheadAttention.forward)
        assert "self.out_proj.weight" in source

    def test_the_peft_table_declares_the_functional_leaves(self) -> None:
        entry = MODEL_LORA_TARGETS["TabLDM"]
        assert entry.functional_weight_substrings == TABLDM_FUNCTIONAL_WEIGHT_LEAVES
        for token in TABLDM_FUNCTIONAL_WEIGHT_LEAVES:
            assert token in entry.target_substrings

    def test_out_proj_gets_the_functional_wrapper(self) -> None:
        model = tiny_model()
        apply_lora(model, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0})
        wrapped = {
            name: isinstance(module, FunctionalWeightLoRALinear)
            for name, module in model.named_modules()
            if isinstance(module, LoRALinear)
        }
        assert wrapped, "nothing was wrapped"
        for name, is_functional in wrapped.items():
            assert is_functional == name.endswith("out_proj"), name

    def test_zero_init_adapters_are_the_identity(self) -> None:
        model = tiny_model()
        X, y = self._episode()
        model.train()
        with torch.no_grad():
            before = model(X=X, y_train=y, return_logits=True).clone()
        apply_lora(model, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0})
        with torch.no_grad():
            after = model(X=X, y_train=y, return_logits=True)
        assert torch.allclose(before, after, atol=1e-6)

    def test_out_proj_adapters_alone_change_the_output(self) -> None:
        model = tiny_model()
        X, y = self._episode()
        model.train()
        with torch.no_grad():
            before = model(X=X, y_train=y, return_logits=True).clone()
        apply_lora(model, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0})
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, FunctionalWeightLoRALinear):
                    module.lora_B.weight.normal_(0.0, 0.5)
            after = model(X=X, y_train=y, return_logits=True)
        assert (after - before).abs().max() > 1e-6

    def test_the_plain_wrapper_would_have_been_silent(self) -> None:
        """The control: what LoRALinear alone on these leaves would have done."""
        model = tiny_model()
        X, y = self._episode()
        model.train()
        inject_custom_lora_into_linear_layers(
            model, target_names=resolve_lora_targets("TabLDM", model),
            r=2, alpha=4, dropout=0.0,
            exclude_patterns=list(TABLDM_LORA_EXCLUDE),
            functional_weight_patterns=None,
        )
        out_projs = [
            module for name, module in model.named_modules()
            if isinstance(module, LoRALinear) and name.endswith("out_proj")
        ]
        assert out_projs, "out_proj was not wrapped"
        assert not any(
            isinstance(m, FunctionalWeightLoRALinear) for m in out_projs
        )
        with torch.no_grad():
            for module in out_projs:
                module.lora_B.weight.normal_(0.0, 0.5)
            perturbed = model(X=X, y_train=y, return_logits=True).clone()
            for module in out_projs:
                module.lora_B.weight.zero_()
            zeroed = model(X=X, y_train=y, return_logits=True)
        assert torch.equal(perturbed, zeroed)

    def test_merged_weight_stays_differentiable_in_the_adapters(self) -> None:
        model = tiny_model()
        apply_lora(model, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0})
        layer = next(
            m for m in model.modules() if isinstance(m, FunctionalWeightLoRALinear)
        )
        assert layer.weight.requires_grad
        layer.weight.sum().backward()
        assert layer.lora_A.weight.grad is not None


class TestDifferentiablePath:
    def test_training_forward_carries_gradients(self) -> None:
        model = tiny_model()
        model.train()
        X = torch.randn(1, 20, 5)
        y = torch.randint(0, 4, (1, 14)).float()
        out = model(X=X, y_train=y, return_logits=True)
        assert out.requires_grad
        assert out.shape == (1, 6, 4)
        out.sum().backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_sklearn_call_sites_are_no_grad(self) -> None:
        """Why a dedicated loop is needed: the public paths are all no_grad."""
        import inspect

        from tabtune.models.tabldm._sklearn import classifier, regressor

        for module in (classifier, regressor):
            assert "torch.no_grad()" in inspect.getsource(module)

    def test_episodes_split_support_from_query(self) -> None:
        rng = np.random.default_rng(0)
        members = [(rng.normal(size=(50, 4)), rng.integers(0, 3, size=50))]
        episodes = list(iter_episodes(
            members, device=torch.device("cpu"), n_episodes=3,
            support_size=24, query_size=16, rng=rng,
        ))
        assert len(episodes) == 3
        for episode in episodes:
            total = episode.table.shape[1]
            assert episode.context_targets.shape[1] == episode.n_context
            assert episode.query_targets.shape[0] == total - episode.n_context
            assert 0 < episode.n_context < total

    def test_episodes_draw_across_every_ensemble_member(self) -> None:
        """Every normalisation method must be trained against, not just one."""
        rng = np.random.default_rng(0)
        members = [
            (np.full((40, 3), float(i)), np.zeros(40)) for i in range(4)
        ]
        seen = {
            float(e.table[0, 0, 0])
            for e in iter_episodes(
                members, device=torch.device("cpu"), n_episodes=60,
                support_size=22, query_size=10, rng=rng,
            )
        }
        assert seen == {0.0, 1.0, 2.0, 3.0}

    def test_episodes_are_reproducible_from_the_seed(self) -> None:
        members = [(np.random.default_rng(1).normal(size=(40, 3)),
                    np.zeros(40))]
        def draw():
            return [
                e.table.clone() for e in iter_episodes(
                    members, device=torch.device("cpu"), n_episodes=3,
                    support_size=22, query_size=10,
                    rng=np.random.default_rng(7),
                )
            ]
        for first, second in zip(draw(), draw()):
            assert torch.equal(first, second)

    def test_classification_loss_is_cross_entropy_on_clamped_labels(self) -> None:
        model = tiny_model(max_classes=4)
        logits = torch.randn(5, 4, requires_grad=True)
        targets = torch.tensor([0.0, 1.0, 2.0, 3.0, 99.0])
        loss = _episode_loss(logits, targets, "classification", model)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_regression_loss_is_pinball_at_the_models_own_levels(self) -> None:
        """The head emits quantiles; the loss has to be the one that makes them so."""
        model = tiny_model(max_classes=0)
        levels = model.quantile_dist.alpha_levels
        assert len(levels) == model.num_quantiles
        quantiles = torch.zeros(64, len(levels), requires_grad=True)
        targets = torch.ones(64)
        loss = _episode_loss(quantiles, targets, "regression", model)
        assert torch.isfinite(loss)
        loss.backward()
        # Every prediction is below its target, so the subgradient per level is
        # -level / n: strictly negative and increasing in magnitude with level.
        gradient = quantiles.grad[0]
        assert torch.all(gradient < 0)
        assert torch.all(gradient.diff() < 0)

    def test_regression_loss_matches_an_independent_pinball(self) -> None:
        model = tiny_model(max_classes=0)
        levels = model.quantile_dist.alpha_levels
        torch.manual_seed(3)
        quantiles = torch.randn(32, len(levels))
        targets = torch.randn(32)
        expected = torch.stack([
            torch.stack([
                torch.maximum(
                    level * (targets[row] - quantiles[row, index]),
                    (level - 1.0) * (targets[row] - quantiles[row, index]),
                )
                for index, level in enumerate(levels)
            ]).mean()
            for row in range(32)
        ]).mean()
        actual = _episode_loss(quantiles, targets, "regression", model)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_regression_loss_prefers_the_true_quantiles(self) -> None:
        model = tiny_model(max_classes=0)
        levels = model.quantile_dist.alpha_levels
        targets = torch.randn(4096)
        truth = torch.quantile(targets, levels).expand(4096, -1)
        flat = torch.zeros(4096, len(levels))
        assert (
            _episode_loss(truth, targets, "regression", model)
            < _episode_loss(flat, targets, "regression", model)
        )


class TestEndToEnd:
    def test_classification_inference_produces_a_simplex(self, checkpoints, table) -> None:
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        proba = model.predict_proba(X[100:])
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(model.predict(X[100:])) <= {0, 1}

    def test_regression_inference_returns_finite_predictions(self, checkpoints, table) -> None:
        X, _, y = table
        model = make_regressor(checkpoints["regression"])
        model.fit(X[:100], y[:100])
        predictions = model.predict(X[100:])
        assert np.shape(predictions) == (40,)
        assert np.isfinite(predictions).all()

    def test_ensemble_members_match_what_the_model_is_fed(self, checkpoints, table) -> None:
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        members = _ensemble_members(model)
        assert members
        for features, targets in members:
            assert features.shape == (100, 6)
            assert targets.shape == (100,)
            assert np.isfinite(features).all()

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_finetune_updates_weights_and_stays_finite(self, checkpoints, table, task) -> None:
        X, y_cls, y_reg = table
        y = y_cls if task == "classification" else y_reg
        model = (make_classifier if task == "classification" else make_regressor)(
            checkpoints[task]
        )
        model.fit(X[:100], y[:100])
        before = {k: v.detach().clone() for k, v in model.model_.state_dict().items()}
        finetune(model, task=task, params={
            "epochs": 2, "steps_per_epoch": 3, "learning_rate": 1e-3,
            "support_size": 45, "query_size": 19,
        })
        after = model.model_.state_dict()
        moved = sum(1 for k in before if not torch.equal(before[k], after[k]))
        assert moved > len(before) // 2, f"only {moved}/{len(before)} tensors moved"
        assert all(torch.isfinite(v).all() for v in after.values())

    def test_finetune_leaves_the_module_in_eval_mode(self, checkpoints, table) -> None:
        """Train mode would send predict() down the training branch."""
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        assert not model.model_.training
        finetune(model, task="classification", params={
            "epochs": 1, "steps_per_epoch": 2, "support_size": 34, "query_size": 14,
        })
        assert not model.model_.training

    def test_finetune_drops_the_stale_kv_cache(self, checkpoints, table) -> None:
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        model.model_kv_cache_ = {"stale": True}
        finetune(model, task="classification", params={
            "epochs": 1, "steps_per_epoch": 2, "support_size": 34, "query_size": 14,
        })
        assert model.model_kv_cache_ is None

    def test_refit_after_finetune_still_predicts(self, checkpoints, table) -> None:
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        finetune(model, task="classification", params={
            "epochs": 1, "steps_per_epoch": 3, "learning_rate": 1e-3,
            "support_size": 45, "query_size": 19,
        })
        model.fit(X[:100], y[:100])
        proba = model.predict_proba(X[100:])
        assert np.isfinite(proba).all()
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_peft_trains_adapters_and_freezes_the_base(self, checkpoints, table) -> None:
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        base_before = {
            name: parameter.detach().clone()
            for name, parameter in model.model_.named_parameters()
        }
        finetune(model, task="classification",
                 params={"epochs": 2, "steps_per_epoch": 3,
                         "learning_rate": 1e-2, "support_size": 45, "query_size": 19},
                 peft_config={"r": 4, "lora_alpha": 8, "lora_dropout": 0.0})

        after = dict(model.model_.named_parameters())
        adapters = [n for n in after if "lora_A" in n or "lora_B" in n]
        assert adapters, "no adapters were injected"

        # Wrapping renames a base leaf `x.out_proj.weight` to
        # `x.out_proj.base.weight`; strip that to pair them up again.
        rewrapped = {n.replace(".base.", "."): p for n, p in after.items()}
        matched = 0
        for name, value in base_before.items():
            current = rewrapped.get(name)
            if current is None:
                continue
            matched += 1
            assert torch.equal(current, value), f"{name} changed under LoRA"
        assert matched == len(base_before), "some base tensors went missing"

        moved = sum(
            1 for n in adapters if "lora_B" in n and after[n].abs().sum() > 0
        )
        assert moved > 0, "no adapter moved off its zero init"

    def test_peft_reports_a_small_trainable_fraction(self, checkpoints, table) -> None:
        X, y, _ = table
        model = make_classifier(checkpoints["classification"])
        model.fit(X[:100], y[:100])
        trainable = apply_lora(model.model_, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0})
        tuned = sum(p.numel() for p in trainable)
        total = sum(p.numel() for p in model.model_.parameters())
        assert 0 < tuned < total // 2


class TestTuningManagerDispatch:
    def test_classification_finetune_reaches_the_tabldm_loop(self, monkeypatch) -> None:
        from tabtune.TuningManager import tuning as tuning_module

        seen = {}

        def fake(self, model, X, y, params=None, peft_config=None, task="classification"):
            seen.update(task=task, peft=peft_config)
            return model

        monkeypatch.setattr(tuning_module.TuningManager, "_finetune_tabldm", fake)
        manager = tuning_module.TuningManager()
        manager.tune(
            TabLDMTabTuneClassifier(), np.zeros((4, 2)), np.array([0, 1, 0, 1]),
            strategy="peft", params={"peft_config": {"r": 4}},
        )
        assert seen["task"] == "classification"
        assert seen["peft"] == {"r": 4}

    def test_regression_finetune_reaches_the_tabldm_loop(self, monkeypatch) -> None:
        from tabtune.TuningManager import tuning as tuning_module

        seen = {}

        def fake(self, model, X, y, params=None, peft_config=None, task="classification"):
            seen.update(task=task, peft=peft_config)
            return model

        monkeypatch.setattr(tuning_module.TuningManager, "_finetune_tabldm", fake)
        manager = tuning_module.TuningManager()
        manager.tune(
            TabLDMRegressorWrapper(), np.zeros((4, 2)), np.array([0.0, 1.0, 2.0, 3.0]),
            strategy="peft", params={"peft_config": {"r": 4}},
        )
        assert seen["task"] == "regression"
        assert seen["peft"] == {"r": 4}


class TestPipelineEndToEnd:
    """The whole TabularPipeline, not just the estimator.

    Unit tests on the estimator pass even when the pipeline's own dispatch
    chains have no arm for the model: ``_predict_proba_uncached`` falls through
    to a raw-tensor path that calls ``self.model.parameters()``, and
    ``_predict_uncached`` falls through to a branch that skips the label
    inverse-transform. Both failures only show up from this level, so the
    labels here are strings - encoded integers coming back would be silent
    with 0/1 labels.
    """

    @staticmethod
    def frame():
        rng = np.random.default_rng(0)
        X = __import__("pandas").DataFrame(
            rng.normal(size=(160, 6)), columns=[f"f{i}" for i in range(6)]
        )
        X["cat"] = rng.choice(list("abc"), size=160)
        X.loc[X.index[:5], "f3"] = np.nan  # exercise native NaN handling
        import pandas as pd

        y_cls = pd.Series(np.where(X["f0"] + X["f1"] > 0, "yes", "no"))
        y_reg = pd.Series(X["f0"] * 2 + X["f1"] + rng.normal(scale=0.1, size=160))
        return X, y_cls, y_reg

    @staticmethod
    def build(checkpoints, task, strategy):
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        params = {
            "epochs": 1, "steps_per_epoch": 2,
            "support_size": 34, "query_size": 14, "learning_rate": 1e-3,
        }
        if strategy == "peft":
            params["peft_config"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        # finetune_mode is a constructor argument, not a tuning_params key -
        # the pipeline overwrites the tuning_params entry with its own value.
        # TabLDM implements only "meta-learning"; the regression default is
        # "turn_by_turn", which the registry would (correctly) warn about.
        return TabularPipeline(
            model_name="TabLDM", task_type=task, tuning_strategy=strategy,
            finetune_mode="meta-learning",
            model_params={
                "n_estimators": 2, "device": "cpu", "random_state": 0,
                "use_amp": False, "enhance_candidates": False,
                "allow_auto_download": False, "model_path": checkpoints[task],
            },
            tuning_params=params,
        )

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_classification_returns_labels_in_the_original_space(
        self, checkpoints, strategy
    ) -> None:
        X, y, _ = self.frame()
        pipeline = self.build(checkpoints, "classification", strategy)
        pipeline.fit(X[:120], y[:120])
        predictions = pipeline.predict(X[120:])
        assert set(np.unique(predictions)) <= {"yes", "no"}
        proba = pipeline.predict_proba(X[120:])
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_regression_returns_finite_predictions(self, checkpoints, strategy) -> None:
        _, _, y = self.frame()
        X, _, _ = self.frame()
        pipeline = self.build(checkpoints, "regression", strategy)
        pipeline.fit(X[:120], y[:120])
        predictions = pipeline.predict(X[120:])
        assert np.shape(predictions) == (40,)
        assert np.isfinite(np.asarray(predictions, dtype=float)).all()


import tabtune.models.tabldm.tabtune_support as _support


class TestParameterVocabulary:
    """The loop speaks TabTune's episodic vocabulary; the old names still work."""

    def test_defaults_use_the_house_names(self) -> None:
        import inspect

        source = inspect.getsource(_support.finetune)
        for key in ("support_size", "query_size", "steps_per_epoch", "grad_clip", "seed"):
            assert f'"{key}"' in source, key

    def test_legacy_names_are_translated(self) -> None:
        resolved = _support._normalise_params({
            "episodes_per_epoch": 5,
            "grad_clip_value": 0.5,
            "random_state": 3,
            "max_episode_rows": 100,
            "context_ratio": 0.8,
        })
        assert resolved == {
            "steps_per_epoch": 5, "grad_clip": 0.5, "seed": 3,
            "support_size": 80, "query_size": 20,
        }

    def test_the_canonical_name_wins_when_both_are_given(self) -> None:
        resolved = _support._normalise_params(
            {"episodes_per_epoch": 5, "steps_per_epoch": 9}
        )
        assert resolved["steps_per_epoch"] == 9

    def test_explicit_sizes_are_not_overwritten_by_a_legacy_row_cap(self) -> None:
        resolved = _support._normalise_params(
            {"max_episode_rows": 100, "support_size": 10}
        )
        assert resolved["support_size"] == 10
        assert "query_size" not in resolved
