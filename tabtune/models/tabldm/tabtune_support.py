"""TabTune-side glue for Xiaomi TabLDM: estimator wrappers, fine-tuning and LoRA.

Nothing in the vendored TabLDM tree is modified. Everything TabTune needs that
upstream does not provide lives here.

Fine-tuning
-----------
Unlike most inference-only releases, TabLDM keeps its supervised entry point:
``TabLDM.forward`` branches on ``self.training`` and the training branch
(``_train_forward``) maps ``X (B, T+Q, H)`` plus ``y_train (B, T)`` to
``(B, Q, out_dim)`` with gradients intact. Only the *call sites* in
``_sklearn/`` are wrapped in ``torch.no_grad()``. So fine-tuning here drives the
vendor's own training path rather than a reconstruction of it.

Episodes are drawn from the fitted ``EnsembleGenerator``, which is the same
preprocessing the model sees at prediction time: each episode picks one
(normalisation method, ensemble member) pair, so the weights are adapted across
the same preprocessing distribution they are used with, not just one member's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ._sklearn.classifier import TabLDMClassifier as _UpstreamClassifier
from ._sklearn.regressor import TabLDMRegressor as _UpstreamRegressor

logger = logging.getLogger(__name__)

#: LoRA target substrings, read off the real ``nn.Linear`` leaves of a built
#: model (see ``tests/test_tabldm_integration.py::TestLoraTargets``):
#:
#:   ``linear1`` / ``linear2``   FFN in the column, row and ICL blocks, and both
#:                               the routed and shared MoE experts
#:   ``attn_res_projs``          the AttnRes residual projections
#:   ``in_linear``               the column cell embedder (``SkippableLinear``)
#:   ``ssmax_layer``             the qassmax attention-scaling MLPs
#:   ``decoder``                 the ICL head
#:   ``out_proj``                the attention output projection
TABLDM_LORA_TARGETS: tuple[str, ...] = (
    "linear1",
    "linear2",
    "attn_res_projs",
    "in_linear",
    "ssmax_layer",
    "decoder",
    "out_proj",
)

#: ``y_encoder`` is a ``OneHotAndLinear`` whose rows are class slots. TabLDM's
#: ensemble permutes class ids per member (``class_shuffle_method="shift"``), so
#: those rows are deliberately interchangeable; an adapter fitted under one
#: permutation does not hold under another.
#:
#: ``router`` decides which MoE experts fire. It was trained jointly with expert
#: specialisation, so a LoRA delta on it re-routes tokens to experts the
#: adapters were never fitted against.
TABLDM_LORA_EXCLUDE: tuple[str, ...] = ("y_encoder", "router")

#: Leaves whose weight is consumed as a tensor rather than by calling the
#: module. ``MultiheadAttention.forward`` hands ``self.out_proj.weight`` and
#: ``self.out_proj.bias`` to a functional attention call (``_model/layers.py``),
#: so a plain LoRA wrapper around one would allocate adapters, report them as
#: trainable, and never change a single output. ``MODEL_LORA_TARGETS["TabLDM"]``
#: declares these as ``functional_weight_substrings`` so they get
#: ``FunctionalWeightLoRALinear``, whose ``weight`` carries the delta instead.
TABLDM_FUNCTIONAL_WEIGHT_LEAVES: tuple[str, ...] = ("out_proj",)


def torch_module(estimator: Any) -> nn.Module:
    """The trainable network behind a fitted TabLDM estimator.

    Raises:
        RuntimeError: If the estimator has not been fitted, since the weights
            are only loaded on the first fit.
    """
    model = getattr(estimator, "model_", None)
    if model is None:
        raise RuntimeError(
            "TabLDM loads its weights during fit; call fit() before asking for "
            "the torch module."
        )
    return model


@dataclass
class Episode:
    """One (support, query) split of one preprocessed ensemble member."""

    table: torch.Tensor            # (1, T+Q, features)
    context_targets: torch.Tensor  # (1, T)
    query_targets: torch.Tensor    # (Q,)
    n_context: int


def _ensemble_members(estimator: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every (features, targets) pair the fitted ensemble generator produces.

    Values are exactly what ``_batch_forward`` feeds the network: normalised,
    outlier-clipped, feature-permuted, and - for classification - relabelled by
    that member's class pattern.
    """
    generator = getattr(estimator, "ensemble_generator_", None)
    if generator is None:
        raise RuntimeError(
            "TabLDM builds its ensemble during fit; call fit() before fine-tuning."
        )
    members: list[tuple[np.ndarray, np.ndarray]] = []
    for features, targets in generator.transform(X=None, mode="train").values():
        for index in range(features.shape[0]):
            members.append(
                (np.ascontiguousarray(features[index]), np.asarray(targets[index]))
            )
    if not members:
        raise RuntimeError("The fitted TabLDM ensemble produced no members.")
    return members


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
    members: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    n_episodes: int,
    support_size: int,
    query_size: int,
    rng: np.random.Generator,
) -> Iterator[Episode]:
    """Yield random support/query splits, one per optimiser step.

    A member is drawn per episode so every normalisation method and feature
    permutation in the ensemble is trained against. Sizes are capped at the rows
    actually available and scaled down proportionally when the table is smaller
    than ``support_size + query_size`` - row attention over the support set is
    quadratic, and an episode runs once per optimiser step.
    """
    for _ in range(n_episodes):
        features, targets = members[int(rng.integers(len(members)))]
        available = len(features)
        wanted = support_size + query_size
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
    """Episodic fine-tuning of a fitted TabLDM estimator, in place.

    The estimator must already be fitted: that is what loads the weights and
    builds the ensemble the episodes are drawn from. Afterwards the caller
    should refit so the caches are rebuilt from the updated weights.

    Args:
        estimator: A fitted ``TabLDMClassifier`` or ``TabLDMRegressor``.
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
    device = estimator.device_
    members = _ensemble_members(estimator)

    trainable = apply_lora(model, peft_config)
    if not trainable:
        raise RuntimeError("No trainable parameters after LoRA injection.")

    optimiser = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["seed"]))
    was_training = model.training
    # forward() branches on self.training: eval mode would run the inference
    # path, which chunks, caches and never builds a graph.
    model.train()

    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            total, seen = 0.0, 0
            episodes = iter_episodes(
                members,
                device=device,
                n_episodes=int(config["steps_per_epoch"]),
                support_size=int(config["support_size"]),
                query_size=int(config["query_size"]),
                rng=rng,
            )
            for episode in episodes:
                # enable_grad is explicit: TabTune may call this from inside an
                # inference context, and every TabLDM call site is a no_grad one.
                with torch.enable_grad():
                    output = model(
                        X=episode.table,
                        y_train=episode.context_targets,
                        return_logits=True,
                    )[0]
                    loss = _episode_loss(output, episode.query_targets, task, model)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(trainable, float(config["grad_clip"]))
                optimiser.step()
                total += float(loss.detach())
                seen += 1
            if seen and epoch % max(1, int(config["log_every"])) == 0:
                logger.info(
                    "[TabLDM] epoch %d/%d loss=%.4f",
                    epoch, config["epochs"], total / seen,
                )
    finally:
        if not was_training:
            model.eval()

    # The KV cache was built from the old weights; drop it so a prediction
    # cannot mix pre- and post-fine-tuning context.
    estimator.model_kv_cache_ = None
    return estimator


def _episode_loss(
    output: torch.Tensor,
    query_targets: torch.Tensor,
    task: str,
    model: nn.Module,
) -> torch.Tensor:
    if task == "classification":
        n_outputs = int(model.max_classes)
        return torch.nn.functional.cross_entropy(
            output.float(), query_targets.long().clamp_(0, n_outputs - 1)
        )
    # The regression head emits one value per quantile level, consumed through
    # QuantileToDistribution. Pinball loss at those exact levels is what makes
    # the emitted values quantiles rather than an arbitrary vector.
    levels = model.quantile_dist.alpha_levels.to(
        device=output.device, dtype=torch.float32
    )
    error = query_targets.float().unsqueeze(-1) - output.float()
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def apply_lora(
    model: nn.Module, peft_config: dict[str, Any] | None
) -> list[torch.nn.Parameter]:
    """Inject LoRA adapters when asked, and return the parameters to optimise."""
    if peft_config is None:
        return [p for p in model.parameters() if p.requires_grad]

    from tabtune.TuningManager.peft_utils import apply_tabular_lora

    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_tabular_lora("TabLDM", model, peft_config=peft_config)

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tuned = sum(p.numel() for p in trainable)
    logger.info(
        "[TabLDM] LoRA trainable params: %s / %s (%.2f%%)",
        f"{tuned:,}", f"{total:,}", 100.0 * tuned / max(1, total),
    )
    return trainable


class _TabTuneMixin:
    """Accepts TabTune's orchestration kwargs and exposes the torch module."""

    def __init__(self, *args, tuning_strategy: str = "inference", **kwargs):
        kwargs.pop("task_type", None)
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
    def _densify(X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


class TabLDMTabTuneClassifier(_TabTuneMixin, _UpstreamClassifier):
    """TabLDM classifier with TabTune's constructor conventions."""

    def fit(self, X, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        return super().fit(self._densify(X), y)

    def predict(self, X):
        return super().predict(self._densify(X))

    def predict_proba(self, X):
        return super().predict_proba(self._densify(X))


class TabLDMTabTuneRegressor(_TabTuneMixin, _UpstreamRegressor):
    """TabLDM regressor with TabTune's constructor conventions."""

    def fit(self, X, y, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return super().fit(
            self._densify(X), np.asarray(y).ravel().astype(float), **kwargs
        )

    def predict(self, X, **kwargs):
        return super().predict(self._densify(X), **kwargs)


__all__ = [
    "Episode",
    "TABLDM_FUNCTIONAL_WEIGHT_LEAVES",
    "TABLDM_LORA_EXCLUDE",
    "TABLDM_LORA_TARGETS",
    "TabLDMTabTuneClassifier",
    "TabLDMTabTuneRegressor",
    "apply_lora",
    "finetune",
    "iter_episodes",
    "torch_module",
]
