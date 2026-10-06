"""Fine-tuning loop shared by the adapters that train through TabTune.

An adapter supplies a loss on a batch of ``(context, label)`` windows; this
module samples the windows, runs AdamW with gradient clipping, optionally
evaluates a fixed validation set, keeps the best weights and exports the
tuned tensors (LoRA only, or the full state dict) for persistence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ...logger import get_logger, log_event
from ...registry.errors import ConfigError

logger = get_logger(__name__)

__all__ = [
    "sample_windows",
    "validation_windows",
    "context_scale",
    "pinball",
    "prepare_trainable",
    "run_training",
    "export_state",
    "import_state",
]

MODES = ("full", "lora")


def sample_windows(
    values: Sequence[np.ndarray], context: int, label: int, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``count`` (context, label) windows, weighted by series length.

    Contexts are left-padded and labels right-padded with ``NaN``, which the
    losses ignore.
    """
    lengths = np.array([len(v) for v in values], dtype=float)
    usable = np.flatnonzero(lengths > 1)
    if usable.size == 0:
        raise ConfigError("Fine-tuning needs at least one series with two or more observations.")
    weights = lengths[usable] / lengths[usable].sum()
    ctx = np.full((count, context), np.nan, dtype=np.float32)
    lab = np.full((count, label), np.nan, dtype=np.float32)
    for i in range(count):
        series = np.asarray(values[int(rng.choice(usable, p=weights))], dtype=np.float32)
        cut = int(rng.integers(1, len(series)))
        history = series[max(0, cut - context) : cut]
        future = series[cut : cut + label]
        ctx[i, context - len(history) :] = history
        lab[i, : len(future)] = future
    return ctx, lab


def validation_windows(
    values: Sequence[np.ndarray], context: int, label: int
) -> tuple[np.ndarray, np.ndarray]:
    """The last ``label`` steps of each series, with the context before them."""
    ctx = np.full((len(values), context), np.nan, dtype=np.float32)
    lab = np.full((len(values), label), np.nan, dtype=np.float32)
    for i, series in enumerate(values):
        series = np.asarray(series, dtype=np.float32)
        cut = max(1, len(series) - label)
        history, future = series[max(0, cut - context) : cut], series[cut : cut + label]
        ctx[i, context - len(history) :] = history
        lab[i, : len(future)] = future
    return ctx, lab


def context_scale(context: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-window standard deviation of the observed context (1.0 when undefined).

    Dividing the loss by it weighs every window equally whatever its scale.
    """
    with np.errstate(all="ignore"):
        loc = np.nanmean(context, axis=1, keepdims=True)
        scale = np.sqrt(np.nanmean((context - loc) ** 2, axis=1))
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)
    return scale.astype(np.float32)


def pinball(torch: Any, pred: Any, target: Any, levels: Any, scale: Any) -> Any:
    """Mean pinball loss of ``[batch, quantile, horizon]`` forecasts, ignoring ``NaN`` targets."""
    mask = ~torch.isnan(target)
    y = torch.nan_to_num(target)
    err = (y[:, None, :] - pred) / scale[:, None, None]
    q = levels.to(pred)[None, :, None]
    loss = torch.maximum(q * err, (q - 1) * err)
    weight = mask[:, None, :].to(pred).expand_as(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def prepare_trainable(
    torch: Any,
    module: Any,
    mode: str,
    *,
    lora_targets: Sequence[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: str,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Freeze ``module`` for ``mode``; return the trainable parameters and the LoRA settings."""
    from ..._internal.lora import LoRALinear, inject_lora

    if mode not in MODES:
        raise ConfigError(f"Unknown fine-tuning mode {mode!r}; use one of {MODES}.")
    for p in module.parameters():
        p.requires_grad = mode == "full"
    lora_meta = None
    if mode == "lora":
        existing = [m for m in module.modules() if isinstance(m, LoRALinear)]
        if not existing:
            wrapped = inject_lora(
                module, lora_targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            if not wrapped:
                raise ConfigError(
                    f"No linear layer matched the LoRA targets {list(lora_targets)}; "
                    "pass tuning_params={'peft_config': {'target_modules': [...]}}."
                )
        module.to(device)
        for layer in module.modules():
            if isinstance(layer, LoRALinear):
                for part in (layer.lora_A, layer.lora_B):
                    if part is not None:
                        part.weight.requires_grad = True
        lora_meta = {
            "targets": ",".join(lora_targets),
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
        }
    trainable = [p for p in module.parameters() if p.requires_grad]
    if not trainable:
        raise ConfigError(f"No trainable parameters for fine-tuning mode {mode!r}.")
    return trainable, lora_meta


def run_training(
    torch: Any,
    module: Any,
    trainable: list[Any],
    loss_fn: Callable[[np.ndarray, np.ndarray], Any],
    *,
    train_values: Sequence[np.ndarray],
    validation_values: Sequence[np.ndarray] | None,
    context: int,
    label: int,
    spec: Any,
) -> dict[str, Any]:
    """AdamW loop over sampled windows; restores the best validation weights."""
    optimizer = torch.optim.AdamW(
        trainable, lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    rng = np.random.default_rng(spec.seed)
    if spec.seed is not None:
        torch.manual_seed(spec.seed)
    val = validation_windows(validation_values, context, label) if validation_values else None
    history: list[dict[str, float]] = []
    best_loss, best_state, bad_checks = math.inf, None, 0
    step, last_loss = 0, math.nan
    module.train()
    try:
        for step in range(1, spec.steps + 1):
            loss = loss_fn(*sample_windows(train_values, context, label, spec.batch_size, rng))
            if not torch.isfinite(loss):
                raise ConfigError(
                    f"Fine-tuning loss became non-finite at step {step}; lower the learning rate."
                )
            optimizer.zero_grad()
            loss.backward()
            if spec.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(trainable, spec.gradient_clip_norm)
            optimizer.step()
            last_loss = float(loss.item())
            report_step = step % max(1, spec.validation_every) == 0 or step == spec.steps
            if report_step:
                log_event(logger, "training_step", "Fine-tuning progress",
                          step=step, total_steps=spec.steps, train_loss=last_loss)
            if val is not None and report_step:
                module.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(*val).item())
                module.train()
                history.append({"step": step, "train_loss": last_loss, "validation_loss": val_loss})
                log_event(logger, "validation", "Validation checkpoint", step=step,
                          train_loss=last_loss, validation_loss=val_loss)
                if val_loss < best_loss - 1e-6:
                    best_loss, bad_checks = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                else:
                    bad_checks += 1
                    if spec.patience is not None and bad_checks >= spec.patience:
                        log_event(logger, "early_stopping", "Early stopping", step=step,
                                  best_validation_loss=best_loss, patience=spec.patience)
                        break
    finally:
        module.eval()
    if best_state is not None:
        module.load_state_dict(best_state)
    return {
        "mode": spec.mode,
        "steps_run": step,
        "context_length": context,
        "prediction_length": label,
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "final_train_loss": last_loss,
        "best_validation_loss": None if best_loss == math.inf else best_loss,
        "history": history,
    }


def export_state(
    module: Any, lora_meta: Mapping[str, Any] | None, base_modified: bool
) -> dict[str, Any]:
    """Tensors to persist after fine-tuning, with a ``__meta__`` entry."""
    from ..._internal.lora import lora_state_dict

    if lora_meta is not None and not base_modified:
        state: dict[str, Any] = dict(lora_state_dict(module))
        state["__meta__"] = {"kind": "lora", **lora_meta}
        return state
    state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    meta: dict[str, Any] = {"kind": "full"}
    if lora_meta is not None:
        meta.update({f"lora_{k}": v for k, v in lora_meta.items()})
    state["__meta__"] = meta
    return state


def import_state(
    module: Any, state: Mapping[str, Any], *, default_targets: Sequence[str], device: str
) -> None:
    """Apply a state produced by :func:`export_state` to a freshly loaded ``module``."""
    from ..._internal.lora import inject_lora

    meta = dict(state.get("__meta__", {}))
    tensors = {k: v for k, v in state.items() if k != "__meta__"}

    def targets(value: Any) -> list[str]:
        return [t for t in str(value or "").split(",") if t] or list(default_targets)

    if meta.get("kind") == "lora":
        inject_lora(
            module, targets(meta.get("targets")), r=int(meta["r"]), alpha=int(meta["alpha"])
        )
        module.to(device)
        _, unexpected = module.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ConfigError(f"Saved LoRA tensors do not match the model: {unexpected[:3]}")
    else:
        if meta.get("lora_r"):
            inject_lora(
                module,
                targets(meta.get("lora_targets")),
                r=int(meta["lora_r"]),
                alpha=int(meta["lora_alpha"]),
            )
        module.to(device)
        module.load_state_dict(tensors)
    module.eval()
