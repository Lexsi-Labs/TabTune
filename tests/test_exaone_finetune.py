"""
Tests for the EXAONE Tabular fine-tuning path in ``TuningManager``.

Everything runs on a TINY randomly-initialised Cross-axis Summary Transformer
(~52k parameters) wired into the real TabTune wrappers and driven through the
**real** ``TuningManager`` methods -- ``_exaone_episode_tensors``,
``_finetune_exaone`` and ``_finetune_exaone_regression_turn_by_turn``. No
Hugging Face download, no GPU, seconds on CPU.

What is pinned here:

* the episode builder's shapes and dtypes, including the leading ensemble axis
  (EXAONE takes support and query as separate ``(E, ...)`` tensors);
* that a real gradient loop **reduces the loss on a fixed held-out episode**.
  The per-step training loss is not evidence: meta-learning resamples an episode
  every step, so that number wanders on its own. The measurement is taken before
  and after, on rows the loop never trains on;
* that the query logits are sliced to the dataset's class count before
  cross-entropy (the head is as wide as the architectural class capacity);
* that ``y_support`` is never optimised and never requires grad -- the label
  encoder ranks labels by comparison-and-count, which has zero gradient;
* the regression loop's median-quantile readout;
* ``TuningManager.tune`` dispatch for both tasks and both strategies;
* the honest state of PEFT: it currently injects **zero** adapters (see
  ``TestPeftInjectsNothing``).

Run:  pytest tests/test_exaone_finetune.py -v
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tabtune.models.exaone.classifier import EXAONETabularClassifier
from tabtune.models.exaone.config import (
    InferenceManifest,
    ModelConfig,
    RegressionConfig,
    RuntimeConfig,
)
from tabtune.models.exaone.model.heads import ClassificationModel, RegressionModel
from tabtune.models.regression.exaone.regressor import EXAONETabularRegressorWrapper

pytestmark = [pytest.mark.model_exaone, pytest.mark.finetuning]

TINY_MODEL = dict(
    width=32, head_count=2, block_count=2, feedforward_widths=64,
    columns_per_group=1, item_summary_count=2, feature_summary_count=4,
    feature_attention_repeats=1, class_capacity=3,
)
TINY_REGRESSION = dict(quantile_count=5, decoder_hidden_width=32)

#: One epoch of a short episodic loop. lr 1e-4 is the largest step that still
#: improves the *held-out* episode reliably at this (random-init) geometry;
#: bigger steps overfit the sampled episodes within a few dozen updates.
FAST_TUNING = dict(
    device="cpu", epochs=1, steps_per_epoch=30, support_size=40, query_size=20,
    learning_rate=1e-4, show_progress=False,
)

CPU = torch.device("cpu")


def _tuning_manager():
    try:
        from tabtune.TuningManager.tuning import TuningManager
    except Exception as exc:  # heavy import chain; skip if a dep is missing
        pytest.skip(f"could not import TuningManager: {exc}")
    return TuningManager


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_tiny_classifier(seed=0, n_ensemble=1):
    from tabtune.models.exaone._upstream_classifier import (
        EXAONETabularClassifier as VendoredClassifier,
    )

    config = ModelConfig(**TINY_MODEL)
    manifest = InferenceManifest(
        task="classification", model=config,
        runtime=RuntimeConfig(compute_dtype="float32",
                              ensemble_count=n_ensemble, seed=seed),
    )
    torch.manual_seed(seed)
    model = ClassificationModel(config, device=CPU, dtype=torch.float32)
    clf = EXAONETabularClassifier(device="cpu", n_ensemble=n_ensemble,
                                  random_state=seed)
    clf.model_ = model
    clf.manifest_ = manifest
    clf.estimator_ = VendoredClassifier(manifest, device="cpu", model=model)
    return clf


def build_tiny_regressor(seed=0, n_ensemble=1):
    from tabtune.models.exaone._upstream_regressor import (
        EXAONETabularRegressor as VendoredRegressor,
    )

    config = ModelConfig(**TINY_MODEL)
    regression = RegressionConfig(**TINY_REGRESSION)
    manifest = InferenceManifest(
        task="regression", model=config, regression=regression,
        runtime=RuntimeConfig(compute_dtype="float32",
                              ensemble_count=n_ensemble, seed=seed),
    )
    torch.manual_seed(seed)
    model = RegressionModel(config, regression, device=CPU, dtype=torch.float32)
    reg = EXAONETabularRegressorWrapper(device="cpu", n_ensemble=n_ensemble,
                                        random_state=seed)
    reg.model_ = model
    reg.manifest_ = manifest
    reg.estimator_ = VendoredRegressor(manifest, device="cpu", model=model)
    return reg


def _clf_frames(n=200, seed=0, holdout=50):
    """A learnable binary rule plus a distractor column; train / held-out split."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.randn(n), "f2": rng.randn(n), "f3": rng.randn(n),
    })
    y = pd.Series(np.where(X["f1"] + 0.5 * X["f2"] > 0, "pos", "neg"))
    cut = n - holdout
    return (X[:cut].reset_index(drop=True), y[:cut].reset_index(drop=True),
            X[cut:].reset_index(drop=True), y[cut:].reset_index(drop=True))


def _reg_frames(n=200, seed=0, holdout=50):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(30.0 * X["f1"] - 20.0 * X["f2"] + 100.0)
    cut = n - holdout
    return (X[:cut].reset_index(drop=True), y[:cut].reset_index(drop=True),
            X[cut:].reset_index(drop=True), y[cut:].reset_index(drop=True))


def _fixed_episodes(build, X_np, y_np, *, is_reg=False, n_support=20,
                    query=10, count=3):
    """Several fixed episodes over held-out rows, to average out measurement noise."""
    episodes = []
    for k in range(count):
        start = n_support + query * k
        episodes.append(build(
            X_np, y_np, np.arange(0, n_support), np.arange(start, start + query),
            CPU, is_reg,
        ))
    return episodes


# --------------------------------------------------------------------------- #
# Episode tensors
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestEpisodeTensors:
    def _builder(self):
        return _tuning_manager()._exaone_episode_tensors

    def test_classification_shapes_and_dtypes(self):
        build = self._builder()
        X = np.random.RandomState(0).randn(50, 4).astype(np.float32)
        y = np.random.RandomState(1).randint(0, 3, 50)
        xs, ys, xq, yq = build(X, y, np.arange(0, 30), np.arange(30, 45), CPU, False)

        # A LEADING ensemble axis on support and query, fine-tuning uses E=1.
        assert xs.shape == (1, 30, 4)
        assert ys.shape == (1, 30)
        assert xq.shape == (1, 15, 4)
        assert yq.shape == (15,)
        # Support labels ride in as floats: the vendored input validator rejects
        # integer label tensors. Query targets are int64 for cross-entropy.
        assert xs.dtype == torch.float32 and xq.dtype == torch.float32
        assert ys.dtype == torch.float32
        assert yq.dtype == torch.int64
        np.testing.assert_array_equal(ys[0].numpy(), y[:30].astype(np.float32))

    def test_regression_shapes_and_dtypes(self):
        build = self._builder()
        X = np.random.RandomState(0).randn(40, 3).astype(np.float32)
        y = np.random.RandomState(1).randn(40).astype(np.float32)
        xs, ys, xq, yq = build(X, y, np.arange(0, 24), np.arange(24, 36), CPU, True)
        assert xs.shape == (1, 24, 3) and xq.shape == (1, 12, 3)
        assert ys.shape == (1, 24) and ys.dtype == torch.float32
        assert yq.shape == (12,) and yq.dtype == torch.float32

    def test_non_contiguous_index_selection_is_handled(self):
        build = self._builder()
        X = np.random.RandomState(0).randn(20, 2).astype(np.float32)
        y = np.random.RandomState(1).randint(0, 2, 20)
        picks = np.array([9, 2, 17, 0])
        xs, ys, _xq, _yq = build(X, y, picks, np.array([1, 3]), CPU, False)
        np.testing.assert_array_equal(xs[0].numpy(), X[picks])
        assert xs.is_contiguous()

    def test_episode_tensors_feed_the_real_forward(self):
        build = self._builder()
        clf = build_tiny_classifier()
        X_train, y_train, _X, _y = _clf_frames(n=60, holdout=0)
        clf.fit(X_train, y_train)
        features, _ = clf.prepare_episode_features(X_train)
        codes = clf.y_encoder_.transform(y_train.values)
        xs, ys, xq, yq = build(features, codes, np.arange(0, 40),
                               np.arange(40, 55), CPU, False)
        logits = clf.episode_logits(xs, ys, xq)
        assert logits.shape == (1, 15, clf.max_classes)
        assert torch.isfinite(logits).all()
        assert yq.shape[0] == logits.shape[1]


# --------------------------------------------------------------------------- #
# The gradient loop
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestFinetuneReducesHeldOutLoss:
    """Measured on rows the loop never sees, before and after.

    The per-step training loss is deliberately *not* the signal: meta-learning
    draws a fresh support/query split every step, so that number moves for
    reasons unrelated to whether anything was learned.
    """

    def test_classification_finetune_reduces_held_out_loss(self):
        manager = _tuning_manager()
        np.random.seed(0)
        torch.manual_seed(0)
        X_train, y_train, X_hold, y_hold = _clf_frames()
        clf = build_tiny_classifier(seed=0)
        clf.fit(X_train, y_train)

        held_np, _ = clf.prepare_episode_features(X_hold)
        held_codes = clf.y_encoder_.transform(y_hold.values)
        episodes = _fixed_episodes(manager._exaone_episode_tensors,
                                   held_np, held_codes)

        def held_out_loss():
            total = 0.0
            with torch.no_grad():
                for xs, ys, xq, yq in episodes:
                    logits = clf.episode_logits(xs, ys, xq)[..., :2]
                    total += float(torch.nn.functional.cross_entropy(
                        logits.reshape(-1, 2).float(), yq
                    ))
            return total / len(episodes)

        before = held_out_loss()
        manager()._finetune_exaone(clf, X_train, y_train, params=dict(FAST_TUNING))
        after = held_out_loss()

        assert after < before, f"held-out loss did not fall: {before:.4f} -> {after:.4f}"
        # The engine is re-fitted from the tuned weights, so it stays usable.
        assert clf.predict(X_hold).shape == (len(X_hold),)
        assert clf.model_.training is False

    def test_regression_finetune_reduces_held_out_loss(self):
        manager = _tuning_manager()
        np.random.seed(0)
        torch.manual_seed(0)
        X_train, y_train, X_hold, y_hold = _reg_frames()
        reg = build_tiny_regressor(seed=0)
        reg.fit(X_train, y_train)

        held_np, _ = reg.prepare_episode_features(X_hold)
        raw = y_hold.values.astype(float)
        # The loop standardises targets because the vendored engine centres and
        # scales the support targets itself -- the episodes speak that space.
        scaled = ((raw - raw.mean()) / (raw.std() + 1e-8)).astype(np.float32)
        episodes = _fixed_episodes(manager._exaone_episode_tensors, held_np,
                                   scaled, is_reg=True)
        median = reg.median_quantile_index

        def held_out_loss():
            total = 0.0
            with torch.no_grad():
                for xs, ys, xq, yq in episodes:
                    preds = reg.episode_predictions(xs, ys, xq)[..., median]
                    total += float(torch.nn.functional.mse_loss(
                        preds.reshape(-1).float(), yq
                    ))
            return total / len(episodes)

        before = held_out_loss()
        manager()._finetune_exaone_regression_turn_by_turn(
            reg, X_train, y_train, params=dict(FAST_TUNING)
        )
        after = held_out_loss()

        assert after < before, f"held-out loss did not fall: {before:.4f} -> {after:.4f}"
        predictions = reg.predict(X_hold)
        assert predictions.shape == (len(X_hold),) and np.isfinite(predictions).all()

    def test_parameters_actually_moved(self):
        """A loop that ran but changed nothing would still "reduce" a noisy loss."""
        manager = _tuning_manager()
        np.random.seed(1)
        torch.manual_seed(1)
        X_train, y_train, _X, _y = _clf_frames(n=120, holdout=0)
        clf = build_tiny_classifier(seed=1)
        clf.fit(X_train, y_train)
        before = {n: p.detach().clone() for n, p in clf.model_.named_parameters()}
        manager()._finetune_exaone(
            clf, X_train, y_train,
            params=dict(FAST_TUNING, steps_per_epoch=5, support_size=30,
                        query_size=15),
        )
        moved = [n for n, p in clf.model_.named_parameters()
                 if not torch.equal(p.detach(), before[n])]
        assert len(moved) > 0.9 * len(before), (
            f"only {len(moved)}/{len(before)} parameters changed"
        )


# --------------------------------------------------------------------------- #
# Loss wiring details
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestLossWiring:
    def test_logits_are_sliced_to_n_classes_before_cross_entropy(self, monkeypatch):
        """The head is ``class_capacity`` wide; the padding columns are never labels.

        Cross-entropy would happily accept the full-width logits, so nothing
        errors if the slice is dropped -- gradient is just spent pushing down
        columns that can never be a label. The width actually handed to the loss
        is recorded instead.
        """
        manager = _tuning_manager()
        seen = []

        class RecordingCrossEntropy(torch.nn.CrossEntropyLoss):
            def forward(self, input, target):  # noqa: A002 - torch's own name
                seen.append((tuple(input.shape), tuple(target.shape)))
                return super().forward(input, target)

        monkeypatch.setattr(torch.nn, "CrossEntropyLoss", RecordingCrossEntropy)

        np.random.seed(0)
        torch.manual_seed(0)
        X_train, y_train, _X, _y = _clf_frames(n=120, holdout=0)
        clf = build_tiny_classifier()
        clf.fit(X_train, y_train)
        assert clf.n_classes_ == 2 < clf.max_classes == 3

        manager()._finetune_exaone(
            clf, X_train, y_train,
            params=dict(FAST_TUNING, steps_per_epoch=3, support_size=30,
                        query_size=15),
        )
        assert seen, "the loop never reached the loss"
        for logits_shape, target_shape in seen:
            assert logits_shape[-1] == clf.n_classes_ == 2
            assert logits_shape[-1] != clf.max_classes
            assert logits_shape[0] == target_shape[0]

    def test_support_labels_are_never_optimised(self):
        """``y_support`` has no usable gradient and must stay out of the graph.

        The label encoder turns labels into ordinal ranks with a
        comparison-and-count, which is zero almost everywhere -- so a loop that
        put ``y_support`` in the optimiser would be pushing on a dead axis.
        """
        manager = _tuning_manager()
        np.random.seed(0)
        torch.manual_seed(0)
        X_train, y_train, _X, _y = _clf_frames(n=120, holdout=0)
        clf = build_tiny_classifier()
        clf.fit(X_train, y_train)

        captured = []
        original = clf.episode_logits

        def spy(x_support, y_support, x_query, **kwargs):
            captured.append((x_support, y_support))
            return original(x_support, y_support, x_query, **kwargs)

        clf.episode_logits = spy
        manager()._finetune_exaone(
            clf, X_train, y_train,
            params=dict(FAST_TUNING, steps_per_epoch=3, support_size=30,
                        query_size=15),
        )
        assert captured
        for x_support, y_support in captured:
            assert y_support.requires_grad is False
            assert y_support.is_leaf
            assert y_support.grad is None
            # The support FEATURES are equally untracked here -- the loop
            # optimises parameters only -- but they are differentiable in
            # principle, which is what attribution uses.
            assert x_support.requires_grad is False

    def test_regression_loss_reads_the_median_quantile_column(self):
        """Which of the ``quantile_count`` levels the loss is taken on.

        The forward is replaced by a leaf tensor, so the backward marks exactly
        the column the loop consumed -- no other column can pick up gradient.
        """
        manager = _tuning_manager()
        np.random.seed(0)
        torch.manual_seed(0)
        X_train, y_train, _X, _y = _reg_frames(n=120, holdout=0)
        reg = build_tiny_regressor()
        reg.fit(X_train, y_train)
        quantile_count = TINY_REGRESSION["quantile_count"]
        leaves = []

        def fake_predictions(x_support, y_support, x_query, **kwargs):
            leaf = torch.zeros(
                1, int(x_query.shape[1]), quantile_count, requires_grad=True
            )
            leaves.append(leaf)
            return leaf

        reg.episode_predictions = fake_predictions
        manager()._finetune_exaone_regression_turn_by_turn(
            reg, X_train, y_train,
            params=dict(FAST_TUNING, steps_per_epoch=2, support_size=30,
                        query_size=15),
        )
        assert leaves
        median = reg.median_quantile_index
        assert median == quantile_count // 2 == 2
        for leaf in leaves:
            assert leaf.grad is not None
            assert leaf.grad[..., median].abs().sum().item() > 0
            others = [c for c in range(quantile_count) if c != median]
            assert leaf.grad[..., others].abs().sum().item() == 0


# --------------------------------------------------------------------------- #
# TuningManager.tune dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestTuneDispatch:
    def test_classification_inference_calls_plain_fit(self, monkeypatch):
        manager = _tuning_manager()
        clf = build_tiny_classifier()
        calls = []
        monkeypatch.setattr(
            EXAONETabularClassifier, "fit",
            lambda self, X, y: calls.append("fit") or self,
        )
        monkeypatch.setattr(
            manager, "_finetune_exaone",
            lambda *a, **k: pytest.fail("inference must not fine-tune"),
        )
        X_train, y_train, _X, _y = _clf_frames(n=40, holdout=0)
        returned = manager().tune(clf, X_train, y_train, strategy="inference")
        assert calls == ["fit"]
        assert returned is clf

    @pytest.mark.parametrize(
        "strategy,finetune_mode,expected_mode",
        [
            ("finetune", None, "meta-learning"),
            ("finetune", "sft", "sft"),
            ("peft", None, "meta-learning"),
        ],
    )
    def test_classification_finetune_and_peft_route_to_finetune_exaone(
        self, monkeypatch, strategy, finetune_mode, expected_mode
    ):
        manager = _tuning_manager()
        clf = build_tiny_classifier()
        recorded = {}

        def fake(self, model, X, y, params=None, peft_config=None, mode="meta-learning"):
            recorded.update(model=model, mode=mode, peft_config=peft_config)
            return model

        monkeypatch.setattr(manager, "_finetune_exaone", fake)
        params = {"device": "cpu"}
        if finetune_mode is not None:
            params["finetune_mode"] = finetune_mode
        if strategy == "peft":
            params["peft_config"] = {"r": 4}
        X_train, y_train, _X, _y = _clf_frames(n=40, holdout=0)
        manager().tune(clf, X_train, y_train, strategy=strategy, params=params)

        assert recorded["model"] is clf
        assert recorded["mode"] == expected_mode
        if strategy == "peft":
            assert recorded["peft_config"] == {"r": 4}

    def test_regression_inference_calls_plain_fit(self, monkeypatch):
        manager = _tuning_manager()
        reg = build_tiny_regressor()
        calls = []
        monkeypatch.setattr(
            EXAONETabularRegressorWrapper, "fit",
            lambda self, X, y: calls.append("fit") or self,
        )
        X_train, y_train, _X, _y = _reg_frames(n=40, holdout=0)
        returned = manager().tune(reg, X_train, y_train, strategy="inference")
        assert calls == ["fit"]
        assert returned is reg

    def test_regression_finetune_routes_to_the_turn_by_turn_loop(self, monkeypatch):
        manager = _tuning_manager()
        reg = build_tiny_regressor()
        recorded = {}

        def fake(self, model, X, y, params=None):
            recorded.update(model=model, params=params)
            return model

        monkeypatch.setattr(
            manager, "_finetune_exaone_regression_turn_by_turn", fake
        )
        X_train, y_train, _X, _y = _reg_frames(n=40, holdout=0)
        manager().tune(reg, X_train, y_train, strategy="finetune",
                       params={"device": "cpu"})
        assert recorded["model"] is reg
        assert recorded["params"]["device"] == "cpu"

    def test_regression_finetune_mode_is_auto_corrected_with_a_warning(
        self, monkeypatch, caplog
    ):
        manager = _tuning_manager()
        reg = build_tiny_regressor()
        monkeypatch.setattr(
            manager, "_finetune_exaone_regression_turn_by_turn",
            lambda self, model, X, y, params=None: model,
        )
        X_train, y_train, _X, _y = _reg_frames(n=40, holdout=0)
        with caplog.at_level(logging.WARNING, logger="tabtune.TuningManager.tuning"):
            manager().tune(reg, X_train, y_train, strategy="finetune",
                           params={"finetune_mode": "meta-learning"})
        assert any("turn_by_turn" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# PEFT: currently a full fine-tune, and it says so
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestPeftInjectsNothing:
    """``tuning_strategy='peft'`` cannot inject LoRA into this architecture.

    Every attention / feed-forward projection is a raw ``nn.Parameter`` used
    through ``F.linear``, and the LoRA injector wraps ``nn.Linear`` leaves. The
    only ``nn.Linear`` leaves in the whole model are the task head's, which
    ``apply_tabular_lora`` excludes on purpose -- otherwise the
    "no target matched -> adapt every linear layer" fallback would silently turn
    EXAONE PEFT into "LoRA on the output head and nothing else".

    These tests pin the honest current behaviour: the run completes, zero
    adapters are injected, and a warning says so. If someone teaches the injector
    to wrap raw parameters, these tests are the ones to update.
    """

    def test_the_only_linear_leaves_are_the_excluded_task_head(self):
        model = build_tiny_classifier().model_
        linears = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)]
        assert linears, "expected at least the task head to be nn.Linear"
        assert all(name.startswith("transformer.classification_heads")
                   for name in linears)
        # The registered LoRA targets name real projections -- as PARAMETERS.
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS

        targets = MODEL_LORA_TARGETS["EXAONETabular"].target_substrings
        parameter_names = [name for name, _ in model.named_parameters()]
        for substring in targets:
            assert any(substring in name for name in parameter_names), substring
            assert not any(substring in name for name in linears), substring

    def test_apply_tabular_lora_wraps_zero_layers(self):
        from tabtune.TuningManager.peft_utils import LoRALinear, apply_tabular_lora

        model = build_tiny_classifier().model_
        adapted = apply_tabular_lora(
            "EXAONETabular", model, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        )
        assert sum(isinstance(m, LoRALinear) for m in adapted.modules()) == 0

    def test_peft_finetune_completes_and_warns_that_it_wrapped_nothing(self, caplog):
        from tabtune.TuningManager.peft_utils import LoRALinear

        manager = _tuning_manager()
        np.random.seed(0)
        torch.manual_seed(0)
        X_train, y_train, X_hold, _y = _clf_frames(n=120, holdout=20)
        clf = build_tiny_classifier()

        with caplog.at_level(logging.WARNING, logger="tabtune.TuningManager.tuning"):
            manager()._finetune_exaone(
                clf, X_train, y_train,
                params=dict(FAST_TUNING, steps_per_epoch=3, support_size=30,
                            query_size=15),
                peft_config={"r": 4, "lora_alpha": 8},
            )

        assert sum(isinstance(m, LoRALinear) for m in clf.model_.modules()) == 0
        assert any("wrapped ZERO layers" in record.message
                   for record in caplog.records), [r.message for r in caplog.records]
        # The run still completes and the model still predicts -- it is a full
        # fine-tune, not a failure.
        assert clf.predict(X_hold).shape == (len(X_hold),)
        assert all(p.requires_grad for p in clf.model_.parameters())
