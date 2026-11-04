"""TabTune estimators, fine-tuning and LoRA for BaiRong TabAbax.

The vendored tree is TabAbax as published, with its absolute ``tababax.``
imports rewritten to this package. TabTune's additions live here.

TabAbax is a TabICLv2 derivative and kept its supervised entry point:
``TabAbax.forward`` branches on ``self.training``, and ``_train_forward`` maps
``X (B, T+Q, H)`` and ``y_train (B, T)`` to ``(B, Q, out)`` with gradients. Only
the estimator call sites run under ``torch.no_grad()``. Fine-tuning drives that
branch under ``tabtune._internal.episodic_finetune``: each episode takes one
fitted ensemble view (normalisation, feature permutation and class shift), a
stratified support/query split of the training part, and the model's own
objective (cross-entropy over the class slots, or pinball loss at the
regression head's quantile levels). Validation refits the estimator on the
context rows and scores the held-out rows through ``predict_proba`` /
``predict``, so the full ensemble is what gets selected.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from tabtune._internal.episodic_finetune import (
    EpisodicConfig,
    FinetuneReport,
    log_loss,
    rmse,
    run_episodic_finetune,
)

from ._model import InferenceConfig
from ._sklearn.classifier import TabAbaxClassifier
from ._sklearn.regressor import TabAbaxRegressor

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "tababax-ai/TabAbax"
DEFAULT_CLASSIFIER_FILE = "tababax-classifier.ckpt"
DEFAULT_REGRESSOR_FILE = "tababax-regressor.ckpt"

# Feed-forward layers of every transformer block, attention output projections,
# the post-stage projection of the second column interaction, the Fourier cell
# embedder's projection and the ICL decoder.
TABABAX_LORA_TARGETS: tuple[str, ...] = ("linear1", "linear2", "out_proj", "projection", "decoder")
# ``attn.out_proj`` is handed to a functional attention call and
# ``in_linear.projection`` to ``F.linear``; their adapters must live in ``weight``.
TABABAX_FUNCTIONAL_WEIGHT_LEAVES: tuple[str, ...] = ("out_proj", "projection")
# ``y_encoder`` rows are class slots that the ensemble's class shift permutes;
# the QASSMax gates scale attention logits by context length.
TABABAX_LORA_EXCLUDE: tuple[str, ...] = ("y_encoder", "ssmax_layer")

FINETUNE_DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "steps_per_epoch": 30,
    "episodes_per_step": 2,
    "support_size": None,
    "max_support_size": 2048,
    "query_size": 256,
}


def _densify(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    if isinstance(X, pd.DataFrame):
        sparse = [c for c in X.columns if isinstance(X[c].dtype, pd.SparseDtype)]
        if sparse:
            X = X.copy()
            for col in sparse:
                X[col] = X[col].sparse.to_dense()
        return X.reset_index(drop=True)
    return X


def _take_rows(X, rows):
    if isinstance(X, pd.DataFrame):
        return X.iloc[rows].reset_index(drop=True)
    return np.asarray(X)[rows]


class _TabTuneMixin:
    """Keeps the training frame, protects fine-tuned weights across refits and exposes the network."""

    @property
    def model(self) -> nn.Module:
        """The TabAbax network; AttributeError before the first fit."""
        network = getattr(self, "model_", None)
        if network is None:
            raise AttributeError("TabAbax loads its weights during fit; call fit() first.")
        return network

    def _load_model(self) -> None:
        # fit() reloads the checkpoint every time. After fine-tuning, or while
        # validation refits the context, the weights in memory are the ones to keep.
        if getattr(self, "model_", None) is not None and (
            getattr(self, "_tabtune_finetuned", False) or getattr(self, "_tabtune_internal_fit_", False)
        ):
            return
        super()._load_model()

    @contextmanager
    def _internal_refit(self):
        self._tabtune_internal_fit_ = True
        try:
            yield
        finally:
            self._tabtune_internal_fit_ = False

    def __getstate__(self):
        # Upstream drops the network on pickling and reloads the checkpoint on
        # unpickling, which would discard fine-tuned weights and LoRA adapters.
        state = super().__getstate__()
        if getattr(self, "_tabtune_finetuned", False) and getattr(self, "model_", None) is not None:
            state["_tabtune_model_"] = self.model_
        return state

    def __setstate__(self, state):
        model = state.pop("_tabtune_model_", None)
        if model is not None:
            state["model_"] = model
        super().__setstate__(state)

    def _remember(self, X, y) -> None:
        if not getattr(self, "_tabtune_internal_fit_", False):
            self._tabtune_X_ = X
            self._tabtune_y_ = y


class TabAbaxTabTuneClassifier(_TabTuneMixin, TabAbaxClassifier):
    """``TabAbaxClassifier`` with TabTune's constructor conventions.

    Adds ``tuning_strategy`` and ``task_type``. The signature repeats upstream's
    parameters so ``get_params`` and ``clone`` see all of them.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: Optional[int] = 1,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = DEFAULT_CLASSIFIER_FILE,
        checkpoint_revision: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = False,
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
        tuning_strategy: str = "inference",
        task_type: Optional[str] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=outlier_threshold,
            softmax_temperature=softmax_temperature,
            average_logits=average_logits,
            support_many_classes=support_many_classes,
            batch_size=batch_size,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            checkpoint_version=checkpoint_version,
            checkpoint_revision=checkpoint_revision,
            device=device,
            use_amp=use_amp,
            use_fa3=use_fa3,
            offload_mode=offload_mode,
            disk_offload_dir=disk_offload_dir,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            inference_config=inference_config,
        )
        self.tuning_strategy = tuning_strategy
        self.task_type = task_type

    def fit(self, X, y):
        X = _densify(X)
        y = np.asarray(y).ravel()
        self._remember(X, y)
        return super().fit(X, y)

    def predict_proba(self, X):
        return super().predict_proba(_densify(X))

    def predict(self, X):
        return super().predict(_densify(X))


class TabAbaxTabTuneRegressor(_TabTuneMixin, TabAbaxRegressor):
    """``TabAbaxRegressor`` with TabTune's constructor conventions."""

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: Optional[int] = 1,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = DEFAULT_REGRESSOR_FILE,
        checkpoint_revision: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = False,
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
        tuning_strategy: str = "inference",
        task_type: Optional[str] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            outlier_threshold=outlier_threshold,
            batch_size=batch_size,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            checkpoint_version=checkpoint_version,
            checkpoint_revision=checkpoint_revision,
            device=device,
            use_amp=use_amp,
            use_fa3=use_fa3,
            offload_mode=offload_mode,
            disk_offload_dir=disk_offload_dir,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            inference_config=inference_config,
        )
        self.tuning_strategy = tuning_strategy
        self.task_type = task_type

    def fit(self, X, y):
        X = _densify(X)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = np.asarray(y, dtype=float).ravel()
        self._remember(X, y)
        return super().fit(X, y)

    def predict(self, X, **kwargs):
        return super().predict(_densify(X), **kwargs)


# ---------------------------------------------------------------- fine-tuning


def apply_lora(model: nn.Module, peft_config: dict[str, Any] | None) -> list[nn.Parameter]:
    """Inject LoRA when ``peft_config`` is given and return the parameters to optimise."""
    if peft_config is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return list(model.parameters())

    from tabtune.TuningManager.peft_utils import apply_tabular_lora

    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_tabular_lora("TabAbax", model, peft_config=peft_config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tuned = sum(p.numel() for p in trainable)
    logger.info(
        "[TabAbax] LoRA trainable params: %s / %s (%.2f%%)",
        f"{tuned:,}", f"{total:,}", 100.0 * tuned / max(1, total),
    )
    return trainable


def ensemble_views(estimator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every fitted (features, targets) view, rows in training order.

    Values are what the estimator feeds the network: normalised,
    outlier-clipped, feature-permuted and, for classification, class-shifted;
    regression targets are the estimator's standardised targets.
    """
    generator = getattr(estimator, "ensemble_generator_", None)
    if generator is None:
        raise RuntimeError("TabAbax builds its ensemble during fit; call fit() before fine-tuning.")
    views = []
    for features, targets in generator.transform(X=None, mode="train").values():
        for index in range(features.shape[0]):
            views.append((np.ascontiguousarray(features[index], dtype=np.float32), np.asarray(targets[index])))
    if not views:
        raise RuntimeError("The fitted TabAbax ensemble produced no views.")
    return views


def episode_loss(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    task: str,
    device: torch.device,
) -> torch.Tensor:
    """The model's own objective on one support/query split of one ensemble view."""
    rows = np.concatenate([support, query])
    table = torch.as_tensor(features[rows], dtype=torch.float32, device=device).unsqueeze(0)
    y_support = torch.as_tensor(targets[support], dtype=torch.float32, device=device).unsqueeze(0)
    y_query = torch.as_tensor(targets[query], dtype=torch.float32, device=device)
    output = model(X=table, y_train=y_support)[0].float()
    if task == "classification":
        return nn.functional.cross_entropy(output, y_query.long().clamp(0, int(model.max_classes) - 1))
    levels = model.quantile_dist.alpha_levels.to(device=output.device, dtype=torch.float32)
    error = y_query.unsqueeze(-1) - output
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def _align_proba(proba: np.ndarray, context_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    if len(context_classes) == len(all_classes) and np.array_equal(context_classes, all_classes):
        return proba
    full = np.full((proba.shape[0], len(all_classes)), 1e-12, dtype=np.float64)
    full[:, np.searchsorted(all_classes, context_classes)] = proba
    return full


def finetune(
    estimator,
    *,
    task: str,
    params: dict[str, Any] | None = None,
    peft_config: dict[str, Any] | None = None,
):
    """Validated episodic fine-tuning of a fitted TabAbax estimator, in place.

    The outcome is kept on the estimator as ``finetune_report_``. When no epoch
    beats the pretrained weights on the validation split, the pretrained
    weights are kept.
    """
    if getattr(estimator, "model_", None) is None:
        raise RuntimeError("Fit the TabAbax estimator before fine-tuning it.")
    model = estimator.model
    if task == "classification" and int(estimator.n_classes_) > int(model.max_classes):
        raise ValueError(
            f"TabAbax fine-tuning supports up to {model.max_classes} classes (the native head "
            f"width); got {estimator.n_classes_}. Inference handles more through hierarchical "
            "classification, so use tuning_strategy='inference'."
        )

    config = EpisodicConfig.from_params(params, defaults=FINETUNE_DEFAULTS)
    device = estimator.device_
    X_all, y_all = estimator._tabtune_X_, np.asarray(estimator._tabtune_y_)
    views = ensemble_views(estimator)
    trainable = apply_lora(model, peft_config)

    if task == "classification":
        all_classes = np.asarray(estimator.classes_)
        codes = np.asarray(estimator.y_encoder_.transform(y_all), dtype=np.int64)
        stratify = codes
    else:
        y_values = y_all.astype(float)
        y_scale = float(np.std(y_values)) or 1.0
        stratify = None

    def _episode_loss(support: np.ndarray, query: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        features, targets = views[int(rng.integers(len(views)))]
        return episode_loss(model, features, targets, support, query, task, device)

    def _evaluate(context: np.ndarray, holdout: np.ndarray) -> float:
        with estimator._internal_refit():
            estimator.fit(_take_rows(X_all, context), y_all[context])
            if task == "classification":
                proba = np.asarray(estimator.predict_proba(_take_rows(X_all, holdout)), dtype=np.float64)
                proba = _align_proba(proba, np.asarray(estimator.classes_), all_classes)
                return log_loss(codes[holdout], proba)
            prediction = np.asarray(estimator.predict(_take_rows(X_all, holdout)), dtype=float)
            return rmse(y_values[holdout], prediction) / y_scale

    try:
        report: FinetuneReport = run_episodic_finetune(
            name="TabAbax",
            module=model,
            trainable=trainable,
            config=config,
            n_rows=len(y_all),
            stratify=stratify,
            episode_loss=_episode_loss,
            evaluate=_evaluate,
            metric_name="logloss" if task == "classification" else "rmse/std",
            device=device,
            lora=peft_config is not None,
        )
    finally:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        with estimator._internal_refit():
            estimator.fit(X_all, y_all)
    estimator.finetune_report_ = report
    estimator._tabtune_finetuned = True
    return estimator


__all__ = [
    "DEFAULT_CLASSIFIER_FILE",
    "DEFAULT_REGRESSOR_FILE",
    "DEFAULT_REPO_ID",
    "FINETUNE_DEFAULTS",
    "TABABAX_FUNCTIONAL_WEIGHT_LEAVES",
    "TABABAX_LORA_EXCLUDE",
    "TABABAX_LORA_TARGETS",
    "TabAbaxTabTuneClassifier",
    "TabAbaxTabTuneRegressor",
    "apply_lora",
    "ensemble_views",
    "episode_loss",
    "finetune",
]
