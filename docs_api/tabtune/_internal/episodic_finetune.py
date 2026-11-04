"""Validated episodic fine-tuning shared by Kumo Tabular, LimiX-2 and TabAbax.

A model module supplies two callables: the loss of one support/query episode
and a validation score for a held-out index set given a context index set.
The driver holds out a validation split, scores the pretrained weights on it
through the model's inference path, trains on stratified episodes with warmup
and cosine decay, re-scores after every epoch, stops early and restores the
best weights seen (the pretrained ones when nothing beat them).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "EpisodicConfig",
    "FinetuneReport",
    "ParameterSnapshot",
    "log_loss",
    "rmse",
    "run_episodic_finetune",
    "sample_episode",
    "split_validation",
]

_PARAM_ALIASES: dict[str, str] = {
    "lr": "learning_rate",
    "episodes_per_epoch": "steps_per_epoch",
    "n_episodes": "steps_per_epoch",
    "context_size": "support_size",
    "grad_clip_value": "grad_clip",
    "max_grad_norm": "grad_clip",
    "random_state": "seed",
    "val_fraction": "validation_fraction",
    "validation_split": "validation_fraction",
    "early_stopping_patience": "patience",
    "accumulation_steps": "episodes_per_step",
    "gradient_accumulation_steps": "episodes_per_step",
    "l2_sp": "l2sp",
    "warmup_ratio": "warmup_fraction",
}

_IGNORED_PARAMS = {
    "device", "finetune_mode", "peft_config", "show_progress", "verbose",
    "batch_size", "n_estimators", "model_name", "task_type",
}


@dataclass
class EpisodicConfig:
    """Settings of one episodic fine-tuning run.

    Attributes:
        epochs: Maximum passes of the episode loop. Early stopping usually
            ends the run before this.
        steps_per_epoch: Optimiser steps per epoch.
        episodes_per_step: Episodes whose gradients are accumulated into one
            optimiser step. More than one reduces gradient noise at the cost
            of time per step.
        learning_rate: Peak AdamW learning rate. ``None`` picks ``1e-5`` for
            full fine-tuning and ``1e-4`` for LoRA.
        weight_decay: AdamW weight decay.
        support_size: Context rows per episode. ``None`` uses as many rows as
            the training part allows after the query rows are taken, capped
            by ``max_support_size``, so episodes resemble the inference
            context.
        max_support_size: Cap on the support rows per episode (memory bound).
        query_size: Query rows per episode (capped at a quarter of the
            training part).
        grad_clip: Gradient-norm clip; ``0`` disables it.
        seed: Seed for the validation split and episode sampling.
        amp: bfloat16 autocast on CUDA devices that support it.
        log_every: Log interval in epochs.
        warmup_fraction: Fraction of optimiser steps spent on linear warmup.
        min_lr_fraction: Floor of the cosine decay, as a fraction of the peak.
        validation_fraction: Fraction of training rows held out for model
            selection. ``0`` disables validation (not recommended).
        min_validation_rows: Below this many held-out rows the split is too
            noisy to select on; the run then trains without validation.
        patience: Epochs without a validation improvement before stopping.
        min_delta: Relative improvement of the validation metric that counts
            as "better".
        keep_best: Restore the best validation weights at the end. With
            ``False`` the final weights are kept whatever validation said.
        l2sp: Coefficient of an L2 penalty towards the pretrained weights
            (L2-SP). ``0`` disables it. For LoRA this is equivalent to weight
            decay on the adapters.
        eval_every: Validate every this many epochs.
    """

    epochs: int = 10
    steps_per_epoch: int = 16
    episodes_per_step: int = 1
    learning_rate: float | None = None
    weight_decay: float = 0.0
    support_size: int | None = None
    max_support_size: int = 2048
    query_size: int = 256
    grad_clip: float = 1.0
    seed: int = 0
    amp: bool = True
    log_every: int = 1
    warmup_fraction: float = 0.1
    min_lr_fraction: float = 0.1
    validation_fraction: float = 0.2
    min_validation_rows: int = 32
    patience: int = 3
    min_delta: float = 1e-3
    keep_best: bool = True
    l2sp: float = 0.0
    eval_every: int = 1

    @classmethod
    def from_params(
        cls, params: dict[str, Any] | None, defaults: dict[str, Any] | None = None
    ) -> "EpisodicConfig":
        """Build a config from ``tuning_params``, resolving aliases.

        ``defaults`` are applied before ``params``; unknown keys are ignored.
        """
        merged: dict[str, Any] = dict(defaults or {})
        for key, value in (params or {}).items():
            canonical = _PARAM_ALIASES.get(key, key)
            if canonical != key and canonical in (params or {}):
                continue
            merged[canonical] = value
        known = set(cls.__dataclass_fields__)
        unknown = [k for k in merged if k not in known]
        for key in unknown:
            if key not in _IGNORED_PARAMS:
                logger.debug("[episodic] ignoring unknown fine-tuning parameter %r", key)
            merged.pop(key)
        config = cls(**merged)
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.epochs) < 0:
            raise ValueError("epochs must be >= 0")
        if int(self.steps_per_epoch) < 1 or int(self.episodes_per_step) < 1:
            raise ValueError("steps_per_epoch and episodes_per_step must be >= 1")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if int(self.query_size) < 1:
            raise ValueError("query_size must be >= 1")
        if self.support_size is not None and int(self.support_size) < 2:
            raise ValueError("support_size must be >= 2")
        if int(self.max_support_size) < 2:
            raise ValueError("max_support_size must be >= 2")
        if not 0.0 <= float(self.warmup_fraction) < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")

    def resolved_learning_rate(self, lora: bool) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return 1e-4 if lora else 1e-5


@dataclass
class FinetuneReport:
    """Outcome of a run, kept on the estimator as ``finetune_report_``."""

    model: str
    strategy: str
    metric_name: str
    n_rows: int
    n_train: int
    n_val: int
    trainable_params: int
    total_params: int
    baseline: float | None = None
    best: float | None = None
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False
    kept: str = "pretrained"
    seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """Whether the kept weights beat the pretrained ones on validation."""
        return self.kept == "fine-tuned" and self.best_epoch > 0

    @property
    def relative_change(self) -> float | None:
        if self.baseline is None or self.best is None or self.baseline == 0:
            return None
        return (self.best - self.baseline) / abs(self.baseline)

    def summary(self) -> str:
        if self.baseline is None:
            return (
                f"[{self.model}] {self.strategy} fine-tuning ran {self.epochs_run} epoch(s) "
                f"without validation (too few rows); kept the final weights."
            )
        change = self.relative_change
        change_text = "" if change is None else f" ({100 * change:+.2f}%)"
        if self.improved:
            outcome = (
                f"kept epoch {self.best_epoch} weights: validation {self.metric_name} "
                f"{self.baseline:.4f} -> {self.best:.4f}{change_text}"
            )
        elif self.kept == "pretrained":
            outcome = (
                f"no epoch beat the pretrained weights on validation "
                f"({self.metric_name} {self.baseline:.4f}); the pretrained weights were kept"
            )
        else:
            outcome = (
                f"kept the final weights (keep_best=False): validation {self.metric_name} "
                f"{self.baseline:.4f} -> {self.history[-1]['val'] if self.history else float('nan'):.4f}"
            )
        early = ", stopped early" if self.stopped_early else ""
        return (
            f"[{self.model}] {self.strategy} fine-tuning: {self.epochs_run} epoch(s){early}, "
            f"{outcome}; {self.n_train} train / {self.n_val} validation rows, "
            f"{self.trainable_params:,} trainable of {self.total_params:,} parameters, "
            f"{self.seconds:.0f}s."
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ helpers


def log_loss(codes: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of integer class codes under ``proba``."""
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, eps, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    codes = np.asarray(codes, dtype=np.int64).ravel()
    return float(-np.log(proba[np.arange(len(codes)), codes]).mean())


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).ravel()
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def split_validation(
    n_rows: int,
    *,
    fraction: float,
    min_rows: int,
    stratify: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train/validation row indices; ``(all_rows, None)`` when too small to select on.

    Classification splits are stratified and a singleton class stays in the
    training part so the context covers every class.
    """
    all_rows = np.arange(n_rows)
    if fraction <= 0 or n_rows < 2 * min_rows:
        return all_rows, None

    if stratify is None:
        perm = rng.permutation(n_rows)
        n_val = int(round(fraction * n_rows))
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    else:
        labels = np.asarray(stratify).ravel()
        train_parts, val_parts = [], []
        for label in np.unique(labels):
            rows = rng.permutation(np.flatnonzero(labels == label))
            if len(rows) < 2:
                train_parts.append(rows)
                continue
            n_val = int(round(fraction * len(rows)))
            n_val = min(max(n_val, 1), len(rows) - 1)
            val_parts.append(rows[:n_val])
            train_parts.append(rows[n_val:])
        train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
        val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)

    if len(val_idx) < min_rows or len(train_idx) < min_rows:
        return all_rows, None
    if stratify is not None and len(np.unique(np.asarray(stratify)[train_idx])) < 2:
        return all_rows, None
    return train_idx, val_idx


def episode_sizes(
    available: int, support_size: int | None, max_support_size: int, query_size: int
) -> tuple[int, int]:
    """Support/query row counts for a training part of ``available`` rows."""
    n_query = min(int(query_size), max(1, available // 4))
    cap = int(max_support_size) if support_size is None else min(int(support_size), int(max_support_size))
    n_support = min(cap, available - n_query)
    return n_support, n_query


def sample_episode(
    train_idx: np.ndarray,
    *,
    n_support: int,
    n_query: int,
    stratify: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random support/query rows; for classification the support covers every class."""
    perm = rng.permutation(train_idx)
    if stratify is None or n_support < 2:
        return perm[:n_support], perm[n_support:n_support + n_query]
    labels = np.asarray(stratify).ravel()[perm]
    classes, first = np.unique(labels, return_index=True)
    anchors = first[: n_support]
    mask = np.zeros(len(perm), dtype=bool)
    mask[anchors] = True
    rest = perm[~mask]
    support = np.concatenate([perm[anchors], rest[: n_support - len(anchors)]])
    query = rest[n_support - len(anchors): n_support - len(anchors) + n_query]
    return support, query


class ParameterSnapshot:
    """CPU copies of the trainable parameters, restored in place."""

    def __init__(self, parameters: Sequence[torch.Tensor]):
        self._parameters = parameters
        self._saved: list[torch.Tensor] | None = None
        self.capture()

    def capture(self) -> None:
        with torch.no_grad():
            self._saved = [p.detach().to("cpu", copy=True) for p in self._parameters]

    def restore(self) -> None:
        assert self._saved is not None
        with torch.no_grad():
            for parameter, saved in zip(self._parameters, self._saved):
                parameter.copy_(saved.to(parameter.device, non_blocking=False))


def _lr_lambda(total_steps: int, warmup_steps: int, min_fraction: float) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_fraction + (1.0 - min_fraction) * cosine

    return factor


def _amp_dtype(config: EpisodicConfig, device: torch.device) -> torch.dtype | None:
    if not config.amp or device.type != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:  # noqa: BLE001
        return None
    return None


# ------------------------------------------------------------------- driver


def run_episodic_finetune(
    *,
    name: str,
    module: nn.Module,
    trainable: list[nn.Parameter],
    config: EpisodicConfig,
    n_rows: int,
    stratify: np.ndarray | None,
    episode_loss: Callable[[np.ndarray, np.ndarray, np.random.Generator], torch.Tensor],
    evaluate: Callable[[np.ndarray, np.ndarray], float] | None,
    metric_name: str,
    device: torch.device,
    lora: bool,
    snapshot_tensors: Sequence[torch.Tensor] | None = None,
) -> FinetuneReport:
    """Run the validated episodic loop and leave ``module`` on its best weights.

    Args:
        name: Model name for log lines.
        module: The network; switched between train and eval mode around validation.
        trainable: Parameters to optimise (all of them, or the LoRA adapters).
        config: Loop settings.
        n_rows: Training rows the estimator was fitted on.
        stratify: Integer class codes per row for classification, else ``None``.
        episode_loss: ``(support_idx, query_idx, rng) -> scalar loss``, called in
            train mode with gradients enabled; ``None`` skips the episode.
        evaluate: ``(context_idx, holdout_idx) -> metric`` (lower is better) through
            the inference path with only ``context_idx`` as context; ``None``
            disables validation.
        metric_name: Validation metric name for logs and the report.
        device: Device the model lives on.
        lora: Whether ``trainable`` are LoRA adapters (picks the default learning rate).
        snapshot_tensors: Buffers mutated in train mode (MoE routing biases) that are
            captured and restored together with the parameters.
    """
    started = time.time()
    config.validate()
    rng = np.random.default_rng(int(config.seed))
    total_params = sum(p.numel() for p in module.parameters())
    tuned_params = sum(p.numel() for p in trainable)
    report = FinetuneReport(
        model=name,
        strategy="lora" if lora else "full",
        metric_name=metric_name,
        n_rows=int(n_rows),
        n_train=int(n_rows),
        n_val=0,
        trainable_params=int(tuned_params),
        total_params=int(total_params),
        config=asdict(config),
    )
    if not trainable:
        raise RuntimeError("No trainable parameters: nothing to fine-tune.")

    train_idx, val_idx = split_validation(
        n_rows,
        fraction=float(config.validation_fraction),
        min_rows=int(config.min_validation_rows),
        stratify=stratify,
        rng=rng,
    )
    if evaluate is None:
        val_idx = None
    if val_idx is None:
        if config.validation_fraction > 0 and evaluate is not None:
            logger.warning(
                "[%s] %d rows are too few for a %.0f%% validation split of at least %d rows; "
                "fine-tuning runs for %d epoch(s) without model selection. Pass "
                "min_validation_rows= to override.",
                name, n_rows, 100 * config.validation_fraction, config.min_validation_rows,
                config.epochs,
            )
    report.n_train, report.n_val = int(len(train_idx)), 0 if val_idx is None else int(len(val_idx))

    snapshot = None
    if config.keep_best and val_idx is not None:
        snapshot = ParameterSnapshot(list(trainable) + list(snapshot_tensors or ()))
    best: float | None = None
    if val_idx is not None:
        module.eval()
        try:
            with torch.no_grad():
                best = float(evaluate(train_idx, val_idx))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] validation through the inference path failed (%s); "
                "continuing without model selection.", name, exc,
            )
            val_idx, snapshot, best = None, None, None
            report.n_train, report.n_val = int(n_rows), 0
            train_idx = np.arange(n_rows)
        finally:
            module.train()
        if best is not None and not math.isfinite(best):
            logger.warning("[%s] pretrained validation %s is not finite; selecting without a baseline.",
                           name, metric_name)
            best = None
    report.baseline = best
    report.best = best
    if best is not None:
        logger.info("[%s] pretrained weights: validation %s=%.4f on %d held-out rows",
                    name, metric_name, best, len(val_idx))

    learning_rate = config.resolved_learning_rate(lora)
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=float(config.weight_decay))
    total_steps = int(config.epochs) * int(config.steps_per_epoch)
    warmup_steps = int(round(float(config.warmup_fraction) * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, _lr_lambda(total_steps, warmup_steps, float(config.min_lr_fraction))
    )
    amp_dtype = _amp_dtype(config, device)
    reference = None
    if float(config.l2sp) > 0:
        reference = [p.detach().clone() for p in trainable]

    n_support, n_query = episode_sizes(
        len(train_idx), config.support_size, int(config.max_support_size), int(config.query_size)
    )
    if n_support < 2 or n_query < 1:
        raise ValueError(
            f"[{name}] {len(train_idx)} training rows are too few for an episode "
            f"(support {n_support}, query {n_query})."
        )
    logger.info(
        "[%s] %s fine-tuning: %d epoch(s) x %d step(s) x %d episode(s), support %d / query %d rows, "
        "lr %.2e (warmup %d steps, cosine to %.0f%%), %s trainable of %s parameters%s",
        name, "LoRA" if lora else "full", config.epochs, config.steps_per_epoch,
        config.episodes_per_step, n_support, n_query, learning_rate, warmup_steps,
        100 * config.min_lr_fraction, f"{tuned_params:,}", f"{total_params:,}",
        f", validating on {len(val_idx)} rows" if val_idx is not None else "",
    )

    bad_epochs = 0
    module.train()
    try:
        for epoch in range(1, int(config.epochs) + 1):
            total, seen, skipped = 0.0, 0, 0
            for _ in range(int(config.steps_per_epoch)):
                optimiser.zero_grad(set_to_none=True)
                step_losses = []
                for _ in range(int(config.episodes_per_step)):
                    support, query = sample_episode(
                        train_idx, n_support=n_support, n_query=n_query, stratify=stratify, rng=rng
                    )
                    with torch.enable_grad(), torch.autocast(
                        device.type, dtype=amp_dtype, enabled=amp_dtype is not None
                    ):
                        loss = episode_loss(support, query, rng)
                    if loss is None or not torch.isfinite(loss):
                        skipped += 1
                        continue
                    loss = loss / int(config.episodes_per_step)
                    if reference is not None:
                        penalty = sum(((p - r) ** 2).sum() for p, r in zip(trainable, reference))
                        loss = loss + float(config.l2sp) * penalty
                    loss.backward()
                    step_losses.append(float(loss.detach()) * int(config.episodes_per_step))
                if not step_losses:
                    continue
                if config.grad_clip:
                    nn.utils.clip_grad_norm_(trainable, float(config.grad_clip))
                optimiser.step()
                scheduler.step()
                total += float(np.mean(step_losses))
                seen += 1
            train_loss = total / seen if seen else float("nan")
            report.epochs_run = epoch
            entry: dict[str, Any] = {
                "epoch": epoch, "train_loss": train_loss, "lr": float(scheduler.get_last_lr()[0]),
                "skipped_episodes": skipped,
            }

            diverged = not all(bool(torch.isfinite(p).all()) for p in trainable)
            if diverged:
                entry["diverged"] = True
                logger.warning("[%s] epoch %d: parameters became non-finite; stopping.", name, epoch)

            validate_now = val_idx is not None and epoch % max(1, int(config.eval_every)) == 0
            if validate_now:
                module.eval()
                try:
                    with torch.no_grad():
                        metric = float("inf") if diverged else float(evaluate(train_idx, val_idx))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s] epoch %d: validation through the inference path failed (%s); "
                        "counting the epoch as no improvement.", name, epoch, exc,
                    )
                    metric = float("inf")
                finally:
                    module.train()
                entry["val"] = metric
                improved = math.isfinite(metric) and (
                    best is None or metric < best - float(config.min_delta) * abs(best)
                )
                entry["improved"] = bool(improved)
                if improved:
                    best, report.best, report.best_epoch, bad_epochs = metric, metric, epoch, 0
                    if snapshot is not None:
                        snapshot.capture()
                else:
                    bad_epochs += 1
            report.history.append(entry)

            if epoch % max(1, int(config.log_every)) == 0:
                extra = ""
                if "val" in entry:
                    extra = (
                        f" val_{metric_name}={entry['val']:.4f}"
                        f" best={best:.4f}@{report.best_epoch}"
                        + (" *" if entry["improved"] else "")
                    )
                logger.info("[%s] epoch %d/%d loss=%.4f%s lr=%.2e%s", name, epoch, config.epochs,
                            train_loss, extra, entry["lr"],
                            f" (skipped {skipped} episode(s))" if skipped else "")
            if diverged:
                report.stopped_early = True
                break
            if validate_now and bad_epochs >= int(config.patience):
                report.stopped_early = True
                logger.info("[%s] no validation improvement for %d epoch(s); stopping early.",
                            name, bad_epochs)
                break
    finally:
        module.eval()

    if snapshot is not None:
        snapshot.restore()
        report.kept = "fine-tuned" if report.best_epoch > 0 else "pretrained"
    else:
        report.kept = "final"
    report.seconds = time.time() - started
    logger.info(report.summary())
    return report
