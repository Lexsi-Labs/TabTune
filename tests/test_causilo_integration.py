"""Causilo wiring, and the fine-tuning path TabTune adds on top of it.

The fine-tuning tests build a tiny randomly-initialised Causilo network instead
of downloading the real checkpoint, so gradient flow is verified without weights
and without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune.registry import get_model_spec, list_model_names

torch = pytest.importorskip("torch")

from tabtune.models.causilo.execution.runner import ModelRunner  # noqa: E402
from tabtune.models.causilo.model import Model, ModelConfig  # noqa: E402
from tabtune.models.causilo.tabtune_support import (  # noqa: E402
    CAUSILO_LORA_TARGETS,
    CausiloTabTuneClassifier,
    CausiloTabTuneRegressor,
    iter_episodes,
    torch_module,
)


def tiny_model(task: str = "classification", outputs: int = 10, seed: int = 0) -> Model:
    """A Causilo network at toy dimensions, with every parameter initialised.

    ``Model.__init__`` documents that some latent and embedding parameters use
    empty initialisation and expects a checkpoint to fill them. Left alone they
    hold raw memory - magnitudes around 1e36 were observed - which overflows to
    NaN downstream and makes any test built on them non-deterministic. Tests get
    no checkpoint, so they initialise the weights themselves.
    """
    config = ModelConfig(
        task=task, width=32, expansion=2, group_size=4, frequencies=8,
        column_latents=4, column_heads=4, column_depths=(1, 1), row_heads=4,
        row_latents=2, row_depths=(1, 1), prediction_heads=4, prediction_depth=1,
        outputs=outputs,
    )
    torch.manual_seed(seed)
    model = Model(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
    return model


class TestRegistry:
    def test_spec_registered(self) -> None:
        spec = get_model_spec("Causilo")
        assert spec.name == "Causilo"
        assert spec.preprocessor_key == "causilo_special"

    @pytest.mark.parametrize("alias", ["causilo", "Causilo-v1", "CAUSILO"])
    def test_aliases_resolve(self, alias: str) -> None:
        assert get_model_spec(alias).name == "Causilo"

    def test_supports_both_tasks_and_peft(self) -> None:
        spec = get_model_spec("Causilo")
        assert {"inference", "finetune", "peft"} <= spec.classification_strategies
        assert {"inference", "finetune", "peft"} <= spec.regression_strategies

    def test_class_ceiling_not_declared(self) -> None:
        # Causilo decomposes >10 classes with output codes, so declaring a hard
        # class cap would reject datasets it handles by design.
        assert get_model_spec("Causilo").envelope.max_classes is None

    def test_listed(self) -> None:
        assert "Causilo" in set(list_model_names())


class TestWiring:
    def test_preprocessor_registered(self) -> None:
        from tabtune.Dataprocess.data_processor import DataProcessor
        import inspect

        source = inspect.getsource(DataProcessor)
        assert "'causilo_special': CausiloPreprocessor" in source
        assert "'Causilo': {'categorical_encoding': 'causilo_special'}" in source

    def test_preprocessor_passes_features_through(self) -> None:
        from tabtune.Dataprocess.causilo_preprocessor import CausiloPreprocessor

        X = pd.DataFrame({"num": [1.0, 2.0, np.nan], "cat": ["a", "b", "a"]})
        y = pd.Series(["yes", "no", "yes"])
        pre = CausiloPreprocessor(task_type="classification").fit(X, y)
        out = pre.transform(X)
        # Untouched: the NaN and the string column must both survive, because
        # Causilo reads them itself.
        pd.testing.assert_frame_equal(out, X)
        assert out["num"].isna().sum() == 1
        encoded = pre.transform_target(y)
        assert set(encoded) == {0, 1}
        assert list(pre.inverse_transform_target(encoded)) == list(y)

    def test_lora_targets_match_real_module_names(self) -> None:
        model = tiny_model()
        names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        for target in CAUSILO_LORA_TARGETS:
            assert any(target in name for name in names), f"no Linear matching {target!r}"

    def test_peft_config_registered(self) -> None:
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS

        assert "Causilo" in MODEL_LORA_TARGETS
        assert set(MODEL_LORA_TARGETS["Causilo"].target_substrings) == set(CAUSILO_LORA_TARGETS)

    def test_packed_qkv_is_declared_as_a_functional_weight(self) -> None:
        """Attention reads the packed weight; a plain wrapper would do nothing."""
        import inspect

        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS
        from tabtune.models.causilo.nn.layers import attention

        source = inspect.getsource(attention.Attention)
        assert "self.projection.weight[" in source
        assert MODEL_LORA_TARGETS["Causilo"].functional_weight_substrings == ("projection",)

    def test_every_projection_adapter_actually_changes_the_output(self) -> None:
        """The regression test for the no-op: each adapter must matter."""
        from tabtune.TuningManager.peft_utils import LoRALinear, apply_tabular_lora
        from tabtune.models.causilo.execution.runner import ModelRunner

        model = tiny_model("classification", outputs=4, seed=0)
        model.eval()
        table = torch.randn(1, 30, 5)
        context = torch.randint(0, 4, (1, 20)).float()
        runner = ModelRunner(model)
        apply_tabular_lora("Causilo", model, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0})

        projections = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, LoRALinear) and name.endswith("projection")
        ]
        assert projections
        with torch.no_grad():
            base = runner.predict(table, context).clone()
            dead = []
            for name, module in projections:
                module.lora_B.weight.normal_(0.0, 0.5)
                if float((runner.predict(table, context) - base).abs().max()) == 0.0:
                    dead.append(name)
                module.lora_B.weight.zero_()
        assert dead == [], f"adapters with no effect: {dead}"

    def test_estimators_accept_tabtune_kwargs(self) -> None:
        clf = CausiloTabTuneClassifier(device="cpu", n_estimators=2, tuning_strategy="finetune")
        assert clf.tuning_strategy == "finetune" and clf.n_estimators == 2
        reg = CausiloTabTuneRegressor(device="cpu", n_estimators=2)
        assert reg.tuning_strategy == "inference"

    def test_torch_module_requires_a_fit(self) -> None:
        with pytest.raises(RuntimeError, match="call fit"):
            torch_module(CausiloTabTuneClassifier(device="cpu"))


class TestDifferentiablePath:
    """Causilo's own API is inference-only; TabTune's fine-tuning is not."""

    def test_engine_prediction_is_inference_only(self) -> None:
        # Documents why the fine-tuning path exists at all: every public
        # prediction route is wrapped in inference_mode, whose outputs autograd
        # refuses to track.
        import inspect

        from tabtune.models.causilo import engine

        assert "torch.inference_mode()" in inspect.getsource(engine)

    def test_runner_forward_carries_gradients(self) -> None:
        model = tiny_model().train()
        runner = ModelRunner(model)
        table = torch.randn(1, 52, 8)
        targets = torch.randint(0, 10, (1, 40)).float()
        logits = runner.predict(table, targets)
        assert logits.shape == (1, 12, 10)
        assert logits.requires_grad
        assert torch.isfinite(logits).all()
        torch.nn.functional.cross_entropy(logits[0], torch.randint(0, 10, (12,))).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_episodes_split_support_from_query(self) -> None:
        features = np.random.default_rng(0).normal(size=(80, 6))
        targets = np.random.default_rng(1).integers(0, 3, 80).astype(float)
        episodes = list(iter_episodes(
            features, targets, device=torch.device("cpu"), n_episodes=3,
            support_size=35, query_size=15, rng=np.random.default_rng(2),
        ))
        assert len(episodes) == 3
        for episode in episodes:
            assert episode.table.shape[1] == episode.n_context + len(episode.query_targets)
            assert episode.context_targets.shape == (1, episode.n_context)
            assert 0 < episode.n_context < episode.table.shape[1]

    def test_episodes_are_reproducible_from_the_seed(self) -> None:
        features = np.random.default_rng(0).normal(size=(60, 4))
        targets = np.zeros(60)
        def build():
            return [e.table.clone() for e in iter_episodes(
                features, targets, device=torch.device("cpu"), n_episodes=2,
                support_size=24, query_size=16, rng=np.random.default_rng(7))]
        for a, b in zip(build(), build()):
            torch.testing.assert_close(a, b)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_episode_loss_is_finite_and_differentiable(self, task: str) -> None:
        from tabtune.models.causilo.tabtune_support import _episode_loss

        logits = torch.randn(12, 10, requires_grad=True)
        targets = (
            torch.randint(0, 10, (12,)).float() if task == "classification"
            else torch.randn(12)
        )
        loss = _episode_loss(logits, targets, task, 10)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_out_of_range_labels_are_clamped_not_crashed(self) -> None:
        # ECOC can hand the head a label id at the capacity boundary; the loss
        # must not raise on it.
        from tabtune.models.causilo.tabtune_support import _episode_loss

        logits = torch.randn(4, 10)
        loss = _episode_loss(logits, torch.tensor([0.0, 9.0, 25.0, -3.0]), "classification", 10)
        assert torch.isfinite(loss)


def stub_checkpoint(monkeypatch, task: str = "classification", seed: int = 0) -> Model:
    """Install a small initialised network in place of the Hub download.

    Causilo leaves its latent and embedding parameters uninitialised on purpose
    - ``model.py`` says so, and the checkpoint is what fills them. A stub that
    skips that step holds raw memory on the order of 1e36 and every downstream
    number becomes NaN, so the stub initialises every parameter itself.
    """
    from tabtune.models.causilo import checkpoints

    model = tiny_model(task=task, seed=seed).eval()
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda _task: model)
    return model


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X.f0 + X.f1 > 0).astype(int))
    return X, y


class TestEndToEnd:
    """Fit, fine-tune and PEFT against a stubbed checkpoint: no weights, no network."""

    def test_inference_produces_a_probability_simplex(self, monkeypatch, table) -> None:
        stub_checkpoint(monkeypatch)
        X, y = table
        clf = CausiloTabTuneClassifier(device="cpu", n_estimators=2, random_state=0).fit(X, y)
        proba = clf.predict_proba(X.iloc[:20])
        assert proba.shape == (20, 2)
        assert np.isfinite(proba).all()
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(20), rtol=1e-5)
        assert clf.predict(X.iloc[:20]).shape == (20,)

    def test_stub_is_actually_initialised(self, monkeypatch, table) -> None:
        """Guards the guard: the stub must fill the parameters, not inherit memory.

        ``Model.__init__`` documents that some latent and embedding parameters
        use empty initialisation and expects a checkpoint to fill them. Asserting
        anything about what uninitialised memory *contains* would be a flaky
        test, so this asserts what the stub guarantees instead: every parameter
        finite and on a sane scale. If that ever fails, the other tests in this
        class are running on garbage.
        """
        model = stub_checkpoint(monkeypatch)
        for name, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all(), name
            assert float(parameter.detach().abs().max()) < 1.0, name

    def test_finetune_updates_weights_and_stays_finite(self, monkeypatch, table) -> None:
        stub_checkpoint(monkeypatch)
        X, y = table
        clf = CausiloTabTuneClassifier(device="cpu", n_estimators=2, random_state=0).fit(X, y)
        module = torch_module(clf)
        before = {n: p.detach().clone() for n, p in module.named_parameters()}

        from tabtune.models.causilo.tabtune_support import finetune

        finetune(clf, task="classification",
                 params={"epochs": 2, "steps_per_epoch": 3, "support_size": 45, "query_size": 19,
                         "learning_rate": 1e-3})
        after = dict(module.named_parameters())
        moved = [n for n in before if not torch.equal(before[n], after[n].detach())]
        assert len(moved) > len(before) // 2, "most weights should have moved"
        assert all(torch.isfinite(p).all() for p in after.values())

    def test_finetune_clears_the_stale_context(self, monkeypatch, table) -> None:
        stub_checkpoint(monkeypatch)
        X, y = table
        from tabtune.models.causilo.tabtune_support import finetune

        clf = CausiloTabTuneClassifier(device="cpu", n_estimators=2, random_state=0).fit(X, y)
        finetune(clf, task="classification",
                 params={"epochs": 1, "steps_per_epoch": 1, "support_size": 34, "query_size": 14})
        # The fitted caches came from the pre-fine-tuning weights; predicting
        # through them would mix two models.
        assert clf._engine.state is None
        clf.fit(X, y)
        assert clf.predict(X.iloc[:5]).shape == (5,)

    def test_peft_trains_adapters_and_freezes_the_base(self, monkeypatch, table) -> None:
        stub_checkpoint(monkeypatch, seed=1)
        X, y = table
        from tabtune.models.causilo.tabtune_support import finetune

        clf = CausiloTabTuneClassifier(device="cpu", n_estimators=2, random_state=0).fit(X, y)
        module = torch_module(clf)
        before = {n: p.detach().clone() for n, p in module.named_parameters()}

        finetune(clf, task="classification",
                 params={"epochs": 1, "steps_per_epoch": 2, "support_size": 45, "query_size": 19,
                         "learning_rate": 1e-3},
                 peft_config={"r": 4, "alpha": 8, "dropout": 0.0})

        adapters = [n for n, _ in module.named_parameters() if "lora" in n.lower()]
        assert adapters, "no LoRA adapters were injected"
        shared = {n: p for n, p in module.named_parameters() if n in before}
        assert shared, "LoRA renamed every parameter; the freeze check would be vacuous"
        assert all(torch.equal(before[n], p.detach()) for n, p in shared.items())

    def test_finetune_needs_a_fitted_estimator(self, monkeypatch) -> None:
        stub_checkpoint(monkeypatch)
        from tabtune.models.causilo.tabtune_support import finetune

        with pytest.raises(RuntimeError, match="call fit"):
            finetune(CausiloTabTuneClassifier(device="cpu"), task="classification")


class TestPipelineEndToEnd:
    """The whole TabularPipeline, not just the estimator.

    Estimator-level tests pass even when the pipeline's dispatch chains have no
    arm for the model: ``_predict_proba_uncached`` falls through to a raw-tensor
    path that calls ``self.model.parameters()`` on what is an sklearn wrapper,
    and ``_predict_uncached`` falls through to a branch that skips the label
    inverse-transform. Labels here are strings, so encoded integers coming back
    are visible rather than silent.
    """

    @pytest.fixture(autouse=True)
    def _weights(self, monkeypatch):
        from tabtune.models.causilo import checkpoints

        models = {
            task: tiny_model(task=task, seed=0).eval()
            for task in ("classification", "regression")
        }
        monkeypatch.setattr(
            checkpoints, "load_pretrained_model", lambda task: models[task]
        )

    @staticmethod
    def frame():
        import pandas as pd

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(160, 6)), columns=[f"f{i}" for i in range(6)])
        X["cat"] = rng.choice(list("abc"), size=160)
        X.loc[X.index[:5], "f3"] = np.nan
        y_cls = pd.Series(np.where(X["f0"] + X["f1"] > 0, "yes", "no"))
        y_reg = pd.Series(X["f0"] * 2 + X["f1"] + rng.normal(scale=0.1, size=160))
        return X, y_cls, y_reg

    @staticmethod
    def build(task, strategy):
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        params = {"epochs": 1, "steps_per_epoch": 2,
                  "support_size": 34, "query_size": 14, "learning_rate": 1e-3}
        if strategy == "peft":
            params["peft_config"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        # finetune_mode is a constructor argument, not a tuning_params key -
        # the pipeline overwrites the tuning_params entry with its own value.
        # Causilo implements only "meta-learning"; the regression default is
        # "turn_by_turn", which the registry would (correctly) warn about.
        return TabularPipeline(
            model_name="Causilo", task_type=task, tuning_strategy=strategy,
            finetune_mode="meta-learning",
            model_params={"device": "cpu"}, tuning_params=params,
        )

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_classification_returns_labels_in_the_original_space(self, strategy) -> None:
        X, y, _ = self.frame()
        pipeline = self.build("classification", strategy)
        pipeline.fit(X[:120], y[:120])
        assert set(np.unique(pipeline.predict(X[120:]))) <= {"yes", "no"}
        proba = pipeline.predict_proba(X[120:])
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_regression_returns_finite_predictions(self, strategy) -> None:
        X, _, y = self.frame()
        pipeline = self.build("regression", strategy)
        pipeline.fit(X[:120], y[:120])
        predictions = pipeline.predict(X[120:])
        assert np.shape(predictions) == (40,)
        assert np.isfinite(np.asarray(predictions, dtype=float)).all()


import tabtune.models.causilo.tabtune_support as _support


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
