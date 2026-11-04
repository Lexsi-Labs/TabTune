"""TabTune estimators, fine-tuning and LoRA for NVIDIA Kumo Tabular.

``_sdm`` is the part of NVIDIA structured-data-models that KumoTabular runs on.
Inference goes through ``ICLModel.forward`` unchanged; fine-tuning runs
support/query episodes through the same forward pass under
``tabtune._internal.episodic_finetune``.
"""

from __future__ import annotations

import logging
import math
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

DEFAULT_REPO_ID = "nvidia/Kumo-Tabular"
DEFAULT_REVISION = "v1.0.0"
SIZES = ("small", "medium", "large")
# Estimator counts used for NVIDIA's reported benchmark results.
DEFAULT_NUM_ESTIMATORS = {"small": 8, "medium": 8, "large": 16}
NUM_QUANTILES = 999
TARGET = "__target__"
MAX_FINETUNE_CLASSES = 10

# Attention and MLP projections of the column, row and ICL blocks.
KUMO_LORA_TARGETS: tuple[str, ...] = ("qkv_lin", "out_lin", "mlp")
# qkv_lin weights are sliced and applied with F.linear for cross-attention.
KUMO_FUNCTIONAL_WEIGHT_LEAVES: tuple[str, ...] = ("qkv_lin",)
# Cell embeddings are read as tensors, label projections are class-slot
# specific, and the attention-temperature gates are tiny scalar MLPs.
KUMO_LORA_EXCLUDE: tuple[str, ...] = ("cell_embedding", "y_lin", "query_scaling")

FINETUNE_DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "steps_per_epoch": 30,
    "episodes_per_step": 2,
    "support_size": None,
    "max_support_size": 2048,
    "query_size": 256,
}


def _sdm():
    from . import _sdm

    return _sdm


def resolve_checkpoint(
    task: str,
    size: str,
    model_path: str | None,
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
) -> str:
    """Local file, local snapshot directory, or Hugging Face download."""
    filename = f"{size}/{'classifier' if task == 'classification' else 'regressor'}.pt"
    if model_path:
        path = Path(model_path).expanduser()
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Kumo Tabular checkpoint not found: {path}")
        return str(path)

    from ._sdm.models._huggingface import download_checkpoint

    return download_checkpoint(
        repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
    )


def load_model(task: str, size: str, checkpoint: str, device: torch.device):
    from ._sdm.models import KumoTabular

    model = KumoTabular(task=task, size=size, pretrained=False, device="meta")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.models[task].load_state_dict(state, assign=True)
    return model.to(device).eval()


def _resolve_device(device: Any) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_frame(X) -> pd.DataFrame:
    if hasattr(X, "toarray"):
        X = X.toarray()
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    frame = frame.reset_index(drop=True)
    frame.columns = [str(c) for c in frame.columns]
    for column in frame.columns:
        values = frame[column]
        if isinstance(values.dtype, pd.SparseDtype):
            frame[column] = values.sparse.to_dense()
        elif pd.api.types.is_datetime64_any_dtype(values):
            # Kumo embeds numerical and categorical cells only.
            frame[column] = values.astype("int64").where(values.notna()).astype("float64")
        elif values.dtype == object:
            frame[column] = values.astype("string")
    return frame


class _KumoBase(BaseEstimator):
    _task = "classification"

    def __init__(
        self,
        size: str = "large",
        num_estimators: int | None = None,
        estimator_batch_size: int | str | None = "auto",
        query_batch_size: int | None = 8192,
        max_context_size: int | None = None,
        model_path: str | None = None,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str | None = DEFAULT_REVISION,
        cache_dir: str | None = None,
        device: Any = None,
        autocast: bool = True,
        low_cardinality: str = "infer",
        seed: int = 0,
        tuning_strategy: str = "inference",
        task_type: str | None = None,
    ):
        self.size = size
        self.num_estimators = num_estimators
        self.estimator_batch_size = estimator_batch_size
        self.query_batch_size = query_batch_size
        self.max_context_size = max_context_size
        self.model_path = model_path
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.device = device
        self.autocast = autocast
        self.low_cardinality = low_cardinality
        self.seed = seed
        self.tuning_strategy = tuning_strategy
        self.task_type = task_type

    @property
    def model(self) -> nn.Module:
        kumo = getattr(self, "kumo_", None)
        if kumo is None:
            raise AttributeError("Kumo Tabular loads its weights during fit; call fit() first.")
        return kumo.models[self._task]

    @property
    def n_estimators_(self) -> int:
        return self.num_estimators or DEFAULT_NUM_ESTIMATORS.get(self.size, 16)

    def _ensure_model(self):
        if getattr(self, "kumo_", None) is None:
            if self.size not in SIZES and not self.model_path:
                raise ValueError(f"size must be one of {SIZES}, got {self.size!r}")
            self.device_ = _resolve_device(self.device)
            self.checkpoint_path_ = resolve_checkpoint(
                self._task, self.size, self.model_path, self.repo_id, self.revision, self.cache_dir
            )
            self.kumo_ = load_model(self._task, self.size, self.checkpoint_path_, self.device_)
        return self.kumo_

    def _feature_table(self, X):
        sdm = _sdm()
        frame = _as_frame(X)
        if getattr(self, "stypes_", None) is None:
            stypes = sdm.infer_stypes(frame, _low_cardinality=self.low_cardinality, unsupported="drop")
            self.stypes_ = {
                name: stype for name, stype in stypes.items()
                if sdm.Stype(stype) in (sdm.Stype.numerical, sdm.Stype.categorical)
            }
            self.feature_names_in_ = list(frame.columns)
        else:
            if not set(self.feature_names_in_).issubset(frame.columns):
                if frame.shape[1] != len(self.feature_names_in_):
                    raise ValueError(
                        f"X has {frame.shape[1]} features, but the estimator was fitted "
                        f"with {len(self.feature_names_in_)}."
                    )
                frame.columns = self.feature_names_in_
            frame = frame[self.feature_names_in_]
        return sdm.TableTensor.from_pandas(frame, stypes=self.stypes_, device=self.device_)

    def _fit_context(self, X, y_frame: pd.DataFrame, target_stype: str):
        sdm = _sdm()
        self._ensure_model()
        self.stypes_ = None
        self.x_context_ = self._feature_table(X)
        self.y_context_ = sdm.TableTensor.from_pandas(
            y_frame, stypes={TARGET: target_stype}, device=self.device_
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def _context(self):
        x, y = self.x_context_, self.y_context_
        estimators = self.n_estimators_
        limit = self.max_context_size
        if limit is None or x.size(0) <= limit:
            return x, y, estimators, False
        # One random context subsample per estimator, as upstream's benchmark does.
        generator = torch.Generator(self.device_).manual_seed(self.seed)
        repeats = math.ceil(estimators * limit / x.size(0))
        perm = torch.cat([
            torch.randperm(x.size(0), generator=generator, device=self.device_)
            for _ in range(repeats)
        ])[: estimators * limit]
        shape = (estimators, limit)
        return x[perm].unflatten(0, shape), y[perm].unflatten(0, shape), None, True

    def _batch_size(self, x_context, n_query: int):
        if self.estimator_batch_size != "auto":
            return self.estimator_batch_size
        if self.device_.type != "cuda" or self.max_context_size is not None:
            return 1
        rows, cols = x_context.size(-2) + n_query, x_context.size(-1)
        return 1 if rows > 3_000 or rows * cols > 50_000 else None

    def _forward_tables(self, x_query_all):
        """Ensemble outputs for an already-built query table, in query chunks."""
        kumo = self.kumo_
        was_training = kumo.training
        kumo.eval()
        try:
            x_context, y_context, estimators, expand = self._context()
            step = self.query_batch_size or x_query_all.size(0)
            outputs = []
            for start in range(0, x_query_all.size(0), step):
                x_query = x_query_all[start:start + step]
                if expand:
                    x_query = x_query.expand(self.n_estimators_, *x_query.size())
                # Reseeded per chunk, so every chunk sees the same members.
                generator = torch.Generator(self.device_).manual_seed(self.seed)
                with torch.amp.autocast(
                    self.device_.type, dtype=torch.float16,
                    enabled=self.autocast and self.device_.type == "cuda",
                ):
                    outputs.append(kumo(
                        x_context=x_context,
                        y_context=y_context,
                        x_query=x_query,
                        num_estimators=estimators,
                        estimator_batch_size=self._batch_size(x_context, x_query.size(-2)),
                        generator=generator,
                    ))
            return outputs
        finally:
            if was_training:
                kumo.train()

    def _forward(self, X):
        if getattr(self, "kumo_", None) is None:
            raise RuntimeError("Call fit() before predicting.")
        return self._forward_tables(self._feature_table(X))


class KumoTabularClassifier(ClassifierMixin, _KumoBase):
    _task = "classification"

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.classes_, codes = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Kumo Tabular needs at least two classes.")
        self.y_train_ = y
        self.y_codes_ = codes.astype(np.int64)
        return self._fit_context(X, pd.DataFrame({TARGET: self.y_codes_}), "categorical")

    def _proba_from_outputs(self, outputs) -> np.ndarray:
        sdm = _sdm()
        parts = []
        for out in outputs:
            columns = list(out.columns[sdm.Stype.numerical])
            order = [columns.index(str(i)) for i in range(len(self.classes_))]
            parts.append(out.numerical[..., order].float().cpu().numpy())
        proba = np.concatenate(parts, axis=0).astype(np.float64)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._proba_from_outputs(self._forward(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KumoTabularRegressor(RegressorMixin, _KumoBase):
    _task = "regression"

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        self.y_train_ = np.asarray(y, dtype=np.float64).ravel()
        return self._fit_context(X, pd.DataFrame({TARGET: self.y_train_}), "numerical")

    @staticmethod
    def _quantiles_from_outputs(outputs) -> np.ndarray:
        parts = [out.numerical.float().cpu().numpy() for out in outputs]
        return np.concatenate(parts, axis=0).astype(np.float64)

    def predict_quantiles(self, X) -> np.ndarray:
        """The 999 predicted quantiles (levels 0.001 ... 0.999) per row."""
        return self._quantiles_from_outputs(self._forward(X))

    def predict(self, X):
        return self.predict_quantiles(X).mean(axis=1)


# ---------------------------------------------------------------- fine-tuning


def train_recipe(kumo):
    """Default recipe without target sign flipping or output post-processing."""
    sp = _sdm().processing
    recipe = kumo.default_recipe()
    recipe.target = sp.StypeDispatch(
        categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")],
        numerical=sp.Standardize(),
    )
    recipe.output = sp.Identity()
    return recipe


def apply_lora(model: nn.Module, peft_config: dict[str, Any] | None) -> list[nn.Parameter]:
    if peft_config is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return list(model.parameters())

    from tabtune.TuningManager.peft_utils import apply_tabular_lora

    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_tabular_lora("KumoTabular", model, peft_config=peft_config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tuned = sum(p.numel() for p in trainable)
    logger.info(
        "[KumoTabular] LoRA trainable params: %s / %s (%.2f%%)",
        f"{tuned:,}", f"{total:,}", 100.0 * tuned / max(1, total),
    )
    return trainable


def episode_loss(kumo, task: str, recipe, x_context, y_context, x_query, y_query, y_scale: float,
                 generator: torch.Generator | None = None):
    from ._sdm.evaluation import to_class_indices

    out = kumo(
        x_context=x_context, y_context=y_context, x_query=x_query, recipe=recipe, generator=generator
    )[0]
    if task == "classification":
        logits, target = to_class_indices(out, y_query, missing_score=-torch.inf)
        return nn.functional.cross_entropy(logits.float(), target)
    # Pinball loss over the model's fixed quantile levels, in target-std units.
    diff = (y_query.numerical - out.numerical).float() / y_scale
    level = torch.linspace(0.001, 0.999, NUM_QUANTILES, device=diff.device)
    return torch.maximum(level * diff, (level - 1) * diff).mean()


def finetune(
    estimator: _KumoBase,
    *,
    task: str,
    params: dict[str, Any] | None = None,
    peft_config: dict[str, Any] | None = None,
) -> _KumoBase:
    """Validated episodic fine-tuning of a fitted estimator, in place.

    ``params`` takes the vocabulary of ``EpisodicConfig``; the outcome is kept
    on the estimator as ``finetune_report_``.
    """
    kumo = getattr(estimator, "kumo_", None)
    if kumo is None:
        raise RuntimeError("Fit the Kumo Tabular estimator before fine-tuning it.")
    if task == "classification" and len(estimator.classes_) > MAX_FINETUNE_CLASSES:
        raise ValueError(
            f"Kumo Tabular fine-tuning supports up to {MAX_FINETUNE_CLASSES} classes "
            f"(got {len(estimator.classes_)}); inference handles more through "
            "error-correcting output codes, so use tuning_strategy='inference'."
        )

    config = EpisodicConfig.from_params(params, defaults=FINETUNE_DEFAULTS)
    device = estimator.device_
    network = kumo.models[task]
    trainable = apply_lora(network, peft_config)
    recipe = train_recipe(kumo)
    x_all, y_all = estimator.x_context_, estimator.y_context_
    n_rows = int(x_all.size(0))
    y_scale = 1.0
    if task == "regression":
        y_scale = float(y_all.numerical.float().std()) or 1.0
    generator = torch.Generator(device).manual_seed(int(config.seed))
    codes = estimator.y_codes_ if task == "classification" else None
    y_values = None if task == "classification" else np.asarray(estimator.y_train_, dtype=np.float64)

    def _index(rows: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)

    def _episode_loss(support: np.ndarray, query: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        s, q = _index(support), _index(query)
        return episode_loss(
            kumo, task, recipe, x_all[s], y_all[s], x_all[q], y_all[q], y_scale, generator=generator
        )

    def _evaluate(context: np.ndarray, holdout: np.ndarray) -> float:
        c, h = _index(context), _index(holdout)
        saved = (estimator.x_context_, estimator.y_context_)
        estimator.x_context_, estimator.y_context_ = x_all[c], y_all[c]
        try:
            outputs = estimator._forward_tables(x_all[h])
            if task == "classification":
                return log_loss(codes[holdout], estimator._proba_from_outputs(outputs))
            prediction = estimator._quantiles_from_outputs(outputs).mean(axis=1)
            return rmse(y_values[holdout], prediction) / y_scale
        finally:
            estimator.x_context_, estimator.y_context_ = saved

    report: FinetuneReport = run_episodic_finetune(
        name="KumoTabular",
        module=kumo,
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
    kumo.eval()
    estimator.finetune_report_ = report
    estimator._tabtune_finetuned = True
    return estimator


__all__ = [
    "DEFAULT_NUM_ESTIMATORS",
    "DEFAULT_REPO_ID",
    "FINETUNE_DEFAULTS",
    "KUMO_FUNCTIONAL_WEIGHT_LEAVES",
    "KUMO_LORA_EXCLUDE",
    "KUMO_LORA_TARGETS",
    "MAX_FINETUNE_CLASSES",
    "KumoTabularClassifier",
    "KumoTabularRegressor",
    "apply_lora",
    "episode_loss",
    "finetune",
    "load_model",
    "resolve_checkpoint",
    "train_recipe",
]
