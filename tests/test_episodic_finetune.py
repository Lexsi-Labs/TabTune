"""The validated episodic fine-tuning driver, on a toy model (no weights, no network)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from torch import nn

from tabtune.TuningManager.episodic_finetune import (
    EpisodicConfig,
    FinetuneReport,
    ParameterSnapshot,
    _lr_lambda,
    episode_sizes,
    log_loss,
    rmse,
    run_episodic_finetune,
    sample_episode,
    split_validation,
)

pytestmark = [pytest.mark.unit, pytest.mark.episodic_finetune]


# ------------------------------------------------------------------ config


def test_from_params_resolves_aliases_and_ignores_foreign_keys() -> None:
    config = EpisodicConfig.from_params(
        {"lr": 3e-4, "episodes_per_epoch": 7, "context_size": 50, "grad_clip_value": 0.5,
         "random_state": 3, "val_fraction": 0.1, "device": "cpu", "finetune_mode": "meta-learning",
         "peft_config": {"r": 4}, "not_a_setting": 1},
        defaults={"steps_per_epoch": 99, "max_support_size": 128},
    )
    assert config.learning_rate == 3e-4
    assert config.steps_per_epoch == 7  # user alias beats the model default
    assert config.support_size == 50 and config.max_support_size == 128
    assert config.grad_clip == 0.5 and config.seed == 3 and config.validation_fraction == 0.1


def test_canonical_name_wins_over_its_alias() -> None:
    config = EpisodicConfig.from_params({"lr": 1.0, "learning_rate": 2e-5})
    assert config.learning_rate == 2e-5


def test_default_learning_rate_depends_on_strategy() -> None:
    config = EpisodicConfig()
    assert config.resolved_learning_rate(lora=False) == 1e-5
    assert config.resolved_learning_rate(lora=True) == 1e-4
    assert EpisodicConfig(learning_rate=0.3).resolved_learning_rate(lora=True) == 0.3


@pytest.mark.parametrize("bad", [
    {"validation_fraction": 1.0}, {"query_size": 0}, {"support_size": 1}, {"steps_per_epoch": 0},
    {"warmup_fraction": 1.0},
])
def test_invalid_settings_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        EpisodicConfig.from_params(bad)


# ----------------------------------------------------------------- helpers


def test_stratified_split_keeps_every_class_in_the_context() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0] * 60 + [1] * 30 + [2] * 10 + [3])
    train, val = split_validation(len(labels), fraction=0.2, min_rows=8, stratify=labels, rng=rng)
    assert val is not None
    assert set(train) | set(val) == set(range(len(labels))) and not set(train) & set(val)
    assert set(labels[train]) == {0, 1, 2, 3}  # the singleton class stays in the context
    assert abs(len(val) - 20) <= 1 and 3 not in labels[val]


def test_split_is_skipped_when_too_small() -> None:
    rng = np.random.default_rng(0)
    train, val = split_validation(40, fraction=0.2, min_rows=32, stratify=None, rng=rng)
    assert val is None and len(train) == 40
    train, val = split_validation(400, fraction=0.0, min_rows=8, stratify=None, rng=rng)
    assert val is None and len(train) == 400


def test_random_split_for_regression() -> None:
    rng = np.random.default_rng(1)
    train, val = split_validation(200, fraction=0.25, min_rows=8, stratify=None, rng=rng)
    assert len(val) == 50 and len(train) == 150 and not set(train) & set(val)


def test_episode_sizes_match_the_training_part() -> None:
    assert episode_sizes(5000, None, 2048, 256) == (2048, 256)
    assert episode_sizes(300, None, 2048, 256) == (225, 75)   # query capped at a quarter
    assert episode_sizes(5000, 100, 2048, 256) == (100, 256)  # explicit support honoured
    assert episode_sizes(5000, 4096, 1024, 256) == (1024, 256)  # ... but capped


def test_stratified_episode_support_covers_every_class() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), [80, 5, 5, 5, 5])
    train_idx = np.arange(len(labels))
    for _ in range(20):
        support, query = sample_episode(train_idx, n_support=12, n_query=20, stratify=labels, rng=rng)
        assert len(support) == 12 and len(query) == 20
        assert set(labels[support]) == set(range(5))
        assert not set(support) & set(query)


def test_lr_schedule_warms_up_then_decays_to_the_floor() -> None:
    factor = _lr_lambda(total_steps=100, warmup_steps=10, min_fraction=0.1)
    assert factor(0) == pytest.approx(0.1)
    assert factor(9) == pytest.approx(1.0)
    assert factor(10) == pytest.approx(1.0)
    assert factor(55) < factor(10) and factor(99) < factor(55)
    assert factor(99) == pytest.approx(0.1, abs=0.01)


def test_metrics() -> None:
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert log_loss(np.array([0, 1]), proba) == pytest.approx(-np.log(0.9) / 2 - np.log(0.8) / 2)
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))


def test_snapshot_restores_in_place() -> None:
    layer = nn.Linear(3, 1)
    snapshot = ParameterSnapshot(list(layer.parameters()))
    with torch.no_grad():
        layer.weight.add_(1.0)
    snapshot.restore()
    assert torch.allclose(layer.weight, snapshot._saved[0])


# ------------------------------------------------------------------ driver


class _Toy:
    """Linear regression stand-in; the context is unused so validation measures the weights."""

    def __init__(self, n_rows=400, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.X = rng.normal(size=(n_rows, 4)).astype(np.float32)
        self.w = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
        self.y = (self.X @ self.w + noise * rng.normal(size=n_rows)).astype(np.float32)
        torch.manual_seed(seed)
        self.net = nn.Linear(4, 1)
        self.evaluations = 0

    def episode_loss(self, support, query, rng):
        x = torch.as_tensor(self.X[query])
        y = torch.as_tensor(self.y[query])
        return nn.functional.mse_loss(self.net(x).squeeze(-1), y)

    def evaluate(self, context, holdout):
        self.evaluations += 1
        assert not self.net.training, "validation must run in eval mode"
        assert not set(context) & set(holdout)
        with torch.no_grad():
            pred = self.net(torch.as_tensor(self.X[holdout])).squeeze(-1).numpy()
        return rmse(self.y[holdout], pred)

    def run(self, **params) -> FinetuneReport:
        config = EpisodicConfig.from_params(params, defaults={"steps_per_epoch": 10, "query_size": 64})
        return run_episodic_finetune(
            name="Toy", module=self.net, trainable=list(self.net.parameters()), config=config,
            n_rows=len(self.y), stratify=None, episode_loss=self.episode_loss,
            evaluate=self.evaluate, metric_name="rmse", device=torch.device("cpu"), lora=False,
        )


def test_improving_run_keeps_the_best_epoch() -> None:
    toy = _Toy()
    report = toy.run(epochs=8, learning_rate=0.05, min_validation_rows=8, patience=3)
    assert report.baseline is not None and report.best < report.baseline
    assert report.kept == "fine-tuned" and report.improved and report.best_epoch >= 1
    assert report.n_val == 80 and report.n_train == 320
    assert [h["epoch"] for h in report.history] == list(range(1, report.epochs_run + 1))
    assert all("val" in h for h in report.history)
    assert "kept epoch" in report.summary()
    train_idx, val_idx = split_validation(
        400, fraction=0.2, min_rows=8, stratify=None, rng=np.random.default_rng(0)
    )
    toy.net.eval()
    assert toy.evaluate(train_idx, val_idx) == pytest.approx(report.best, rel=1e-6)


def test_diverging_run_falls_back_to_the_pretrained_weights() -> None:
    toy = _Toy()
    before = {k: v.clone() for k, v in toy.net.state_dict().items()}
    report = toy.run(epochs=4, learning_rate=50.0, min_validation_rows=8, patience=2, grad_clip=0.0)
    assert report.kept == "pretrained" and not report.improved and report.best_epoch == 0
    assert report.best == report.baseline
    for key, value in toy.net.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert "pretrained weights were kept" in report.summary()


def test_early_stopping_after_patience_epochs() -> None:
    toy = _Toy()
    report = toy.run(epochs=20, learning_rate=0.0, min_validation_rows=8, patience=2)
    assert report.stopped_early and report.epochs_run == 2
    assert toy.evaluations == 1 + 2  # baseline + one per epoch


def test_no_validation_when_too_few_rows(caplog) -> None:
    toy = _Toy(n_rows=40)
    with caplog.at_level(logging.WARNING, logger="tabtune"):
        report = toy.run(epochs=2, learning_rate=0.05, min_validation_rows=32)
    assert report.n_val == 0 and report.baseline is None and report.kept == "final"
    assert toy.evaluations == 0 and report.epochs_run == 2
    assert "without validation" in report.summary()


def test_validation_can_be_switched_off() -> None:
    toy = _Toy()
    report = toy.run(epochs=2, learning_rate=0.05, validation_fraction=0.0)
    assert report.n_val == 0 and report.kept == "final" and toy.evaluations == 0


def test_keep_best_false_keeps_the_final_weights() -> None:
    toy = _Toy()
    before = {k: v.clone() for k, v in toy.net.state_dict().items()}
    report = toy.run(epochs=2, learning_rate=50.0, min_validation_rows=8, keep_best=False, grad_clip=0.0)
    assert report.kept == "final"
    assert any(not torch.equal(v, before[k]) for k, v in toy.net.state_dict().items())


def test_gradient_accumulation_and_l2sp_run() -> None:
    toy = _Toy()
    report = toy.run(epochs=2, learning_rate=0.05, min_validation_rows=8, episodes_per_step=3, l2sp=1e-3)
    assert report.epochs_run == 2 and all(np.isfinite(h["train_loss"]) for h in report.history)


def test_non_finite_losses_are_skipped() -> None:
    toy = _Toy()
    calls = {"n": 0}
    original = toy.episode_loss

    def flaky(support, query, rng):
        calls["n"] += 1
        if calls["n"] % 2:
            return torch.tensor(float("nan"), requires_grad=True)
        return original(support, query, rng)

    toy.episode_loss = flaky
    report = toy.run(epochs=1, learning_rate=0.05, min_validation_rows=8)
    assert report.history[0]["skipped_episodes"] == 5


def test_report_round_trips_to_dict() -> None:
    toy = _Toy()
    report = toy.run(epochs=1, learning_rate=0.05, min_validation_rows=8)
    data = report.to_dict()
    assert data["model"] == "Toy" and data["config"]["epochs"] == 1 and len(data["history"]) == 1
