"""TabTune estimators, fine-tuning and LoRA for LimiX-2.

``_upstream`` is LimiX ``inference/v2_0`` and ``model/v2_0``; preprocessing and
ensembling run unchanged through ``LimiXPredictor``. Fine-tuning replays that
preprocessing for one ensemble member per episode and trains through the
model's forward pass under ``tabtune._internal.episodic_finetune``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import nn

from tabtune._internal.episodic_finetune import (
    EpisodicConfig,
    FinetuneReport,
    log_loss,
    rmse,
    run_episodic_finetune,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "_upstream" / "config"
DEFAULT_CONFIGS = {
    "classification": CONFIG_DIR / "cls_default_noretrieval_v2.json",
    "regression": CONFIG_DIR / "reg_default_noretrieval_v2.json",
}
DEFAULT_FILENAME = "LimiX-2.ckpt"
# The upstream examples download from the first; the README links the second.
DEFAULT_REPO_IDS = ("stableai-org/LimiX-2", "stable-ai/LimiX-2")
MAX_CLASSES = 10

# Attention and gated-MLP projections in the sample, feature (DSTI) and MLP
# blocks. All are applied as modules, so plain LoRALinear wrappers take effect.
LIMIXV2_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# softmax-scaling MLPs pass scale_linear.weight to a functional op.
LIMIXV2_LORA_EXCLUDE: tuple[str, ...] = ("scale_linear",)

FINETUNE_DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "steps_per_epoch": 16,
    "episodes_per_step": 1,
    "support_size": None,
    "max_support_size": 1024,
    "query_size": 256,
}


def resolve_checkpoint(
    model_path: str | None,
    repo_id: str | None,
    filename: str,
    cache_dir: str | None,
) -> str:
    if model_path:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LimiX-2 checkpoint not found: {path}")
        return str(path)

    from huggingface_hub import hf_hub_download

    candidates = (repo_id,) if repo_id else DEFAULT_REPO_IDS
    errors = []
    for candidate in candidates:
        try:
            return hf_hub_download(repo_id=candidate, filename=filename, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "Could not download the LimiX-2 checkpoint. Pass model_path= to a local "
        f"{filename}, or check network access to Hugging Face.\n" + "\n".join(errors)
    )


def _resolve_device(device: Any) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _densify(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def _take_rows(X, rows):
    if isinstance(X, pd.DataFrame):
        return X.iloc[rows].reset_index(drop=True)
    return np.asarray(X)[rows]


class _LimiXV2Base(BaseEstimator):
    _task = "classification"

    def __init__(
        self,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str = DEFAULT_FILENAME,
        cache_dir: str | None = None,
        inference_config: str | dict | None = None,
        device: Any = None,
        mix_precision: bool = True,
        softmax_temperature: float = 0.9,
        seed: int = 0,
        test_batch_size: int | None = 16384,
        deterministic: bool = False,
        gpu_ids: list[int] | None = None,
        enable_preprocess_parallel: bool = True,
        preprocess_num_jobs: int = 16,
        tuning_strategy: str = "inference",
        task_type: str | None = None,
    ):
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename
        self.cache_dir = cache_dir
        self.inference_config = inference_config
        self.device = device
        self.mix_precision = mix_precision
        self.softmax_temperature = softmax_temperature
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.deterministic = deterministic
        self.gpu_ids = gpu_ids
        self.enable_preprocess_parallel = enable_preprocess_parallel
        self.preprocess_num_jobs = preprocess_num_jobs
        self.tuning_strategy = tuning_strategy
        self.task_type = task_type

    @property
    def _upstream_task(self) -> str:
        return "Classification" if self._task == "classification" else "Regression"

    def _build_predictor(self):
        from ._upstream.inference.predictor import LimiXPredictor

        config = self.inference_config
        if config is None:
            config = str(DEFAULT_CONFIGS[self._task])
        self.device_ = _resolve_device(self.device)
        self.checkpoint_path_ = resolve_checkpoint(
            self.model_path, self.repo_id, self.filename, self.cache_dir
        )
        return LimiXPredictor(
            device=self.device_,
            model_path=self.checkpoint_path_,
            inference_config=config,
            mix_precision=self.mix_precision,
            softmax_temperature=self.softmax_temperature,
            seed=self.seed,
            test_batch_size=self.test_batch_size,
            deterministic=self.deterministic,
            gpu_ids=self.gpu_ids,
            enable_preprocess_parallel=self.enable_preprocess_parallel,
            preprocess_num_jobs=self.preprocess_num_jobs,
        )

    def _fit_context(self, X, y):
        if getattr(self, "predictor_", None) is None:
            self.predictor_ = self._build_predictor()
        X = _densify(X)
        self.X_train_ = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def _predict(self, X):
        if getattr(self, "predictor_", None) is None:
            raise RuntimeError("Call fit() before predicting.")
        X = _densify(X)
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
        return self.predictor_.predict(
            self.X_train_, self.y_train_, X, task_type=self._upstream_task
        )

    @property
    def model(self) -> nn.Module:
        predictor = getattr(self, "predictor_", None)
        if predictor is None:
            # AttributeError, so hasattr(estimator, "model") is False before fit.
            raise AttributeError("LimiX-2 loads its weights during fit; call fit() first.")
        predictor._ensure_model_loaded()
        return predictor.model


class LimiXV2Classifier(ClassifierMixin, _LimiXV2Base):
    _task = "classification"

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if not 2 <= len(self.classes_) <= MAX_CLASSES:
            raise ValueError(
                f"LimiX-2 supports 2 to {MAX_CLASSES} classes, got {len(self.classes_)}."
            )
        self.y_train_ = y
        return self._fit_context(X, y)

    def predict_proba(self, X):
        return np.asarray(self._predict(X), dtype=np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LimiXV2Regressor(RegressorMixin, _LimiXV2Base):
    _task = "regression"

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        self.y_train_ = np.asarray(y, dtype=np.float64).ravel()
        return self._fit_context(X, self.y_train_)

    def predict(self, X):
        return np.asarray(self._predict(X), dtype=np.float64).ravel()


# ---------------------------------------------------------------- fine-tuning


def _classification_member(predictor, x_s, y_s, x_q, pipe_index, rng):
    """One ensemble member's preprocessed episode, as predict() would build it."""
    train_frame, test_frame = predictor._classification_feature_frame_pair(x_s, x_q)
    predictor._configure_adaptive_svd_runtime(
        train_rows=len(y_s), query_rows=len(test_frame), unique_dataset_name=None
    )
    frame = pd.concat([train_frame, test_frame], ignore_index=True)
    values, categorical = predictor._member_numeric_inputs(
        frame, train_size=len(y_s), cast_float32=True
    )[pipe_index]
    classes, y_encoded = np.unique(y_s, return_inverse=True)
    permutation = rng.permutation(len(classes))
    predictor.class_permutations = [permutation] * predictor.n_estimators
    x_, y_, _ = predictor.get_xy_cls(
        values.copy(), pipe_index, y_s.copy(), y_encoded.copy(),
        predictor.preprocess_pipelines[pipe_index], list(categorical), "Classification",
    )
    return np.asarray(x_), np.asarray(y_), classes, permutation


def _regression_member(predictor, x_s, y_s, x_q, y_q, pipe_index):
    member_inputs, y_standard = predictor._prepare_reg_context(x_s, y_s, x_q)
    values, categorical = member_inputs[pipe_index]
    x_, y_, transforms = predictor.get_xy_reg(
        values.copy(), pipe_index, y_standard.copy(),
        predictor.preprocess_pipelines[pipe_index], list(categorical), "Regression",
    )
    query = (np.asarray(y_q, dtype=np.float64) - predictor.y_mean) / predictor.y_std
    for step in transforms:
        if step is not None:
            query = np.asarray(step.transform(query.reshape(-1, 1))).ravel()
    return np.asarray(x_), np.asarray(y_), query


def _forward(model, x, y, eval_pos, task, device, amp_dtype):
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        return model(x=x_t, y=y_t, eval_pos=eval_pos, task_type=task)


def _classification_loss(output, model_config, n_classes, permutation, targets):
    logits = output["cls_output"].squeeze(0).float()
    if model_config.get("cls_random_mapping", False):
        mapping = output["cls_process_config"]["cls_head_mapping"]
        logits = torch.gather(logits, -1, mapping.expand(logits.shape[0], -1))
    logits = logits[:, :n_classes][:, torch.as_tensor(permutation, device=logits.device)]
    return nn.functional.cross_entropy(logits, targets)


def _regression_loss(output, model, targets):
    logits = output["reg_output"][0].squeeze(0).float()
    borders = model._reg_borders.to(device=logits.device, dtype=torch.float32)
    buckets = torch.searchsorted(borders, targets.contiguous(), right=True) - 1
    buckets = buckets.clamp_(0, logits.shape[-1] - 1)
    return nn.functional.cross_entropy(logits, buckets)


def apply_lora(model: nn.Module, peft_config: dict[str, Any] | None) -> list[nn.Parameter]:
    if peft_config is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return list(model.parameters())

    from tabtune.TuningManager.peft_utils import apply_tabular_lora

    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_tabular_lora("LimiXv2", model, peft_config=peft_config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tuned = sum(p.numel() for p in trainable)
    logger.info(
        "[LimiXv2] LoRA trainable params: %s / %s (%.2f%%)",
        f"{tuned:,}", f"{total:,}", 100.0 * tuned / max(1, total),
    )
    return trainable


def _align_proba(proba: np.ndarray, context_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    """Expand probabilities to ``all_classes`` columns."""
    if len(context_classes) == len(all_classes) and np.array_equal(context_classes, all_classes):
        return proba
    full = np.full((proba.shape[0], len(all_classes)), 1e-12, dtype=np.float64)
    positions = np.searchsorted(all_classes, context_classes)
    full[:, positions] = proba
    return full


def finetune(
    estimator: _LimiXV2Base,
    *,
    task: str,
    params: dict[str, Any] | None = None,
    peft_config: dict[str, Any] | None = None,
) -> _LimiXV2Base:
    """Validated episodic fine-tuning of a fitted estimator, in place.

    Each episode preprocesses a support/query split through one ensemble
    pipeline, as ``predict`` does, and minimises cross-entropy on the query
    rows (over the bucket grid for regression). The outcome is kept on the
    estimator as ``finetune_report_``.
    """
    from ._upstream.model.autobatch import AutobatchConfig

    predictor = getattr(estimator, "predictor_", None)
    if predictor is None:
        raise RuntimeError("Fit the LimiX-2 estimator before fine-tuning it.")

    config = EpisodicConfig.from_params(params, defaults=FINETUNE_DEFAULTS)
    device = estimator.device_

    # Frozen-model reuse and GPU pool workers reload the checkpoint.
    predictor.reuse_frozen_model = False
    if predictor._use_pipeline_gpu_pool():
        warnings.warn(
            "LimiX-2 multi-GPU pipeline workers load weights from the checkpoint file; "
            "running the fine-tuned model on a single device instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        predictor.close()
        predictor.gpu_ids = None

    model = estimator.model.to(device)
    trainable = apply_lora(model, peft_config)

    X, y = estimator.X_train_, np.asarray(estimator.y_train_)
    n_rows = len(y)
    upstream_task = estimator._upstream_task
    pipelines = list(predictor._active_pipeline_indices())
    amp_dtype = None
    if config.amp and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    if task == "classification":
        all_classes = estimator.classes_
        codes = np.searchsorted(all_classes, y).astype(np.int64)
        y_scale = 1.0
    else:
        all_classes = None
        codes = None
        y_scale = float(np.std(y)) or 1.0

    def _episode_loss(support: np.ndarray, query: np.ndarray, rng: np.random.Generator):
        x_s, x_q = _take_rows(X, support), _take_rows(X, query)
        pipe_index = pipelines[int(rng.integers(len(pipelines)))]
        autobatch, AutobatchConfig.ENABLE_AUTOBATCH = AutobatchConfig.ENABLE_AUTOBATCH, False
        try:
            if task == "classification":
                if len(np.unique(y[support])) < 2:
                    return None
                x_, y_, classes, permutation = _classification_member(
                    predictor, x_s, y[support], x_q, pipe_index, rng
                )
                known = np.isin(y[query], classes)
                if not known.any():
                    return None
                targets = torch.as_tensor(np.searchsorted(classes, y[query]), device=device)
                output = _forward(model, x_, y_, len(support), upstream_task, device, amp_dtype)
                mask = torch.as_tensor(known, device=device)
                logits_output = {**output, "cls_output": output["cls_output"][:, mask]}
                return _classification_loss(
                    logits_output, predictor.model_config, len(classes), permutation, targets[mask]
                )
            x_, y_, y_query = _regression_member(
                predictor, x_s, y[support], x_q, y[query], pipe_index
            )
            targets = torch.as_tensor(y_query, dtype=torch.float32, device=device)
            output = _forward(model, x_, y_, len(support), upstream_task, device, amp_dtype)
            return _regression_loss(output, model, targets)
        finally:
            AutobatchConfig.ENABLE_AUTOBATCH = autobatch

    def _evaluate(context: np.ndarray, holdout: np.ndarray) -> float:
        saved = (estimator.X_train_, estimator.y_train_)
        estimator.X_train_, estimator.y_train_ = _take_rows(X, context), y[context]
        try:
            x_h = _take_rows(X, holdout)
            if task == "classification":
                proba = np.asarray(estimator._predict(x_h), dtype=np.float64)
                proba = _align_proba(proba, np.unique(y[context]), all_classes)
                return log_loss(codes[holdout], proba)
            prediction = np.asarray(estimator._predict(x_h), dtype=np.float64).ravel()
            return rmse(y[holdout], prediction) / y_scale
        finally:
            estimator.X_train_, estimator.y_train_ = saved

    report: FinetuneReport = run_episodic_finetune(
        name="LimiXv2",
        module=model,
        trainable=trainable,
        config=config,
        n_rows=n_rows,
        stratify=codes,
        episode_loss=_episode_loss,
        evaluate=_evaluate,
        metric_name="logloss" if task == "classification" else "rmse/std",
        device=device,
        lora=peft_config is not None,
    )
    model.eval()
    estimator.finetune_report_ = report
    estimator._tabtune_finetuned = True
    return estimator


__all__ = [
    "DEFAULT_CONFIGS",
    "DEFAULT_REPO_IDS",
    "FINETUNE_DEFAULTS",
    "LIMIXV2_LORA_EXCLUDE",
    "LIMIXV2_LORA_TARGETS",
    "LimiXV2Classifier",
    "LimiXV2Regressor",
    "apply_lora",
    "finetune",
    "resolve_checkpoint",
]
