"""TabTune-side glue for Causilo: estimator wrappers, fine-tuning and LoRA.

Nothing in the vendored Causilo tree is modified. Everything TabTune needs that
upstream does not provide lives here.

Why fine-tuning needs its own path
----------------------------------
Causilo ships as an inference library: ``Engine.fit`` prepares a context without
touching weights, and every prediction path in ``engine.py`` runs inside
``torch.inference_mode()``. Tensors produced there cannot take part in autograd,
so the public estimator API cannot be fine-tuned through.

The network underneath is ordinary, though. ``ModelRunner.predict(table,
targets)`` maps ``(B, T+Q, features)`` plus ``(B, T)`` training targets to
``(B, Q, outputs)`` and carries gradients when it is not called under inference
mode. This module builds episodes and drives that entry point directly, which is
the same episodic scheme TabTune already uses for the other in-context models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data.dataset import PreparedDataset
from .engine import Engine, resolve_device
from .estimators import CausiloClassifier as _UpstreamClassifier
from .estimators import CausiloRegressor as _UpstreamRegressor
from .execution.runner import ModelRunner

logger = logging.getLogger(__name__)

#: LoRA target substrings, taken from the module names in ``nn/layers``:
#: ``Attention.projection`` is the packed QKV matrix, ``Attention.output`` the
#: attention output, and ``FeedForward`` holds ``value``/``gate``/``output``.
#: The prediction head is deliberately excluded - it is the only layer whose
#: output width is the class capacity, and adapting it interacts badly with the
#: error-correcting output codes used above ten classes.
CAUSILO_LORA_TARGETS: tuple[str, ...] = ("projection", "output", "value", "gate")
CAUSILO_LORA_EXCLUDE: tuple[str, ...] = ("head",)


def torch_module(estimator: Any) -> nn.Module:
    """The trainable network behind a fitted Causilo estimator.

    Raises:
        RuntimeError: If the estimator has not been fitted, since the weights
            are only downloaded on the first fit.
    """
    engine = getattr(estimator, "_engine", None)
    model = getattr(engine, "model", None) if engine is not None else None
    if model is None:
        raise RuntimeError(
            "Causilo loads its weights during fit; call fit() before asking for "
            "the torch module."
        )
    return model


@dataclass
class Episode:
    """One (support, query) split of a prepared table, ready for the network."""

    table: torch.Tensor      # (1, T+Q, features)
    context_targets: torch.Tensor  # (1, T)
    query_targets: torch.Tensor    # (Q,)
    n_context: int


def _prepared_arrays(state) -> tuple[np.ndarray, np.ndarray]:
    """Normalised training features and encoded targets of the first member."""
    dataset: PreparedDataset = state.dataset
    member = dataset.members_by_normalization()[0]
    features = dataset.training_table(member.normalization)[:, member.feature_order]
    return np.ascontiguousarray(features), np.asarray(dataset.targets_for(member))


#: Parameter names this loop accepted before it adopted TabTune's episodic
#: vocabulary (``support_size`` / ``query_size`` / ``steps_per_epoch`` /
#: ``grad_clip`` / ``seed``, as used by the TabFM and TabPFN loops). Kept so
#: existing calls keep working; the canonical name wins if both are given.
_PARAM_ALIASES = {
    "episodes_per_epoch": "steps_per_epoch",
    "grad_clip_value": "grad_clip",
    "random_state": "seed",
}


def _normalise_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Map legacy parameter names onto the canonical ones."""
    if not params:
        return {}
    resolved = {}
    for key, value in params.items():
        canonical = _PARAM_ALIASES.get(key, key)
        if canonical in params and canonical != key:
            continue  # the canonical spelling was given too; it wins
        resolved[canonical] = value
    # context_ratio / max_episode_rows described the same split as
    # support_size / query_size; translate rather than silently ignore.
    ratio = resolved.pop("context_ratio", None)
    rows = resolved.pop("max_episode_rows", None)
    if rows is not None and "support_size" not in resolved and "query_size" not in resolved:
        ratio = 0.7 if ratio is None else float(ratio)
        resolved["support_size"] = max(1, int(int(rows) * ratio))
        resolved["query_size"] = max(1, int(rows) - resolved["support_size"])
    return resolved


def iter_episodes(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    n_episodes: int,
    support_size: int,
    query_size: int,
    rng: np.random.Generator,
) -> Iterator[Episode]:
    """Yield random support/query splits of the same table.

    The support rows are what the model conditions on and the query rows are
    what it is scored against, so each episode is a fresh in-context task drawn
    from one dataset. Sizes are capped at the rows actually available and scaled
    down proportionally when the table is smaller than
    ``support_size + query_size`` - attention over the support set is quadratic,
    and an episode runs once per optimiser step.
    """
    available = len(features)
    wanted = support_size + query_size
    for _ in range(n_episodes):
        if wanted > available:
            n_support = max(1, int(available * support_size / wanted))
            n_query = max(1, available - n_support)
        else:
            n_support, n_query = support_size, query_size
        n_support = min(n_support, available - 1)
        if n_support < 1:
            continue
        n_query = min(n_query, available - n_support)
        rows = rng.permutation(available)[: n_support + n_query]
        table = torch.as_tensor(
            features[rows], dtype=torch.float32, device=device
        ).unsqueeze(0)
        all_targets = torch.as_tensor(targets[rows], dtype=torch.float32, device=device)
        yield Episode(
            table=table,
            context_targets=all_targets[:n_support].unsqueeze(0),
            query_targets=all_targets[n_support:],
            n_context=n_support,
        )


def finetune(
    estimator: Any,
    *,
    task: str,
    params: dict[str, Any] | None = None,
    peft_config: dict[str, Any] | None = None,
) -> Any:
    """Episodic fine-tuning of a fitted Causilo estimator, in place.

    The estimator must already be fitted: that is what downloads the weights and
    builds the preprocessing state this reuses. Afterwards the caller should
    refit so the prediction caches are rebuilt from the updated weights.

    Args:
        estimator: A fitted ``CausiloClassifier`` or ``CausiloRegressor``.
        task: ``"classification"`` or ``"regression"``.
        params: Optimiser and episode settings; see ``defaults`` below.
            Uses TabTune's episodic vocabulary (``support_size``,
            ``query_size``, ``steps_per_epoch``, ``grad_clip``, ``seed``).
        peft_config: When given, LoRA adapters are injected and the base weights
            frozen, so only the adapters train.

    Returns:
        The same estimator, with updated weights.
    """
    defaults: dict[str, Any] = {
        "epochs": 10,
        "steps_per_epoch": 16,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "support_size": 717,
        "query_size": 307,
        "grad_clip": 1.0,
        "seed": 0,
        "log_every": 1,
    }
    config = {**defaults, **_normalise_params(params)}

    model = torch_module(estimator)
    engine: Engine = estimator._engine
    device = engine.device
    state = engine.require_state()
    features, targets = _prepared_arrays(state)

    trainable = _apply_lora(model, peft_config)
    if not trainable:
        raise RuntimeError("No trainable parameters after LoRA injection.")

    optimiser = torch.optim.AdamW(
        trainable, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    rng = np.random.default_rng(int(config["seed"]))
    runner = ModelRunner(model)
    was_training = model.training
    model.train()

    n_outputs = model.config.outputs
    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            total, seen = 0.0, 0
            episodes = iter_episodes(
                features, targets, device=device,
                n_episodes=int(config["steps_per_epoch"]),
                support_size=int(config["support_size"]),
                query_size=int(config["query_size"]),
                rng=rng,
            )
            for episode in episodes:
                # enable_grad is explicit: TabTune may call this from inside an
                # inference context, and Causilo's own API always is one.
                with torch.enable_grad():
                    logits = runner.predict(episode.table, episode.context_targets)[0]
                    loss = _episode_loss(logits, episode.query_targets, task, n_outputs)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(trainable, float(config["grad_clip"]))
                optimiser.step()
                total += float(loss.detach())
                seen += 1
            if seen and epoch % max(1, int(config["log_every"])) == 0:
                logger.info("[Causilo] epoch %d/%d loss=%.4f", epoch, config["epochs"], total / seen)
    finally:
        if not was_training:
            model.eval()

    # The fitted caches were built from the old weights; drop them so the next
    # prediction cannot mix pre- and post-fine-tuning context.
    engine.state = None
    return estimator


def _episode_loss(
    logits: torch.Tensor, query_targets: torch.Tensor, task: str, n_outputs: int
) -> torch.Tensor:
    if task == "classification":
        return torch.nn.functional.cross_entropy(
            logits.float(), query_targets.long().clamp_(0, n_outputs - 1)
        )
    # The regression head emits `outputs` channels that Engine averages after
    # sorting; training against their mean matches how they are consumed.
    return torch.nn.functional.mse_loss(logits.float().mean(dim=-1), query_targets.float())


def _apply_lora(model: nn.Module, peft_config: dict[str, Any] | None) -> list[torch.nn.Parameter]:
    """Inject LoRA adapters when asked, and return the parameters to optimise."""
    if peft_config is None:
        return [p for p in model.parameters() if p.requires_grad]

    from tabtune.TuningManager.peft_utils import apply_tabular_lora

    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_tabular_lora("Causilo", model, peft_config=peft_config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tuned = sum(p.numel() for p in trainable)
    logger.info(
        "[Causilo] LoRA trainable params: %s / %s (%.2f%%)",
        f"{tuned:,}", f"{total:,}", 100.0 * tuned / max(1, total),
    )
    return trainable


class _TabTuneMixin:
    """Accepts TabTune's orchestration kwargs and exposes the torch module."""

    def __init__(self, *args, tuning_strategy: str = "inference", **kwargs):
        kwargs.pop("task_type", None)
        # TabTune passes device="cuda"/"cpu"/"auto"; Causilo validates it itself.
        super().__init__(*args, **kwargs)
        self.tuning_strategy = tuning_strategy

    @property
    def model(self) -> nn.Module:
        """The trainable network, for fine-tuning and PEFT.

        Raises:
            AttributeError: Before the first fit. ``hasattr`` only swallows
                AttributeError, and TabTune probes ``hasattr(model, 'model')``
                to decide whether an estimator exposes a torch module - a
                RuntimeError here would escape that probe as a crash.
        """
        try:
            return torch_module(self)
        except RuntimeError as exc:
            raise AttributeError(str(exc)) from exc

    @staticmethod
    def _to_frame(X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


class CausiloTabTuneClassifier(_TabTuneMixin, _UpstreamClassifier):
    """Causilo classifier with TabTune's constructor conventions."""

    def fit(self, X, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        return super().fit(self._to_frame(X), y)

    def predict(self, X):
        return super().predict(self._to_frame(X))

    def predict_proba(self, X):
        return super().predict_proba(self._to_frame(X))


class CausiloTabTuneRegressor(_TabTuneMixin, _UpstreamRegressor):
    """Causilo regressor with TabTune's constructor conventions."""

    def fit(self, X, y):
        if isinstance(y, (pd.DataFrame,)):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return super().fit(self._to_frame(X), np.asarray(y).ravel().astype(float))

    def predict(self, X, **kwargs):
        return super().predict(self._to_frame(X), **kwargs)


__all__ = [
    "CAUSILO_LORA_EXCLUDE",
    "CAUSILO_LORA_TARGETS",
    "CausiloTabTuneClassifier",
    "CausiloTabTuneRegressor",
    "Episode",
    "finetune",
    "iter_episodes",
    "resolve_device",
    "torch_module",
]
