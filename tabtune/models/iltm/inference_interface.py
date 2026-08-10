from __future__ import annotations

import os
import gc
import math
import logging
from typing import List, Tuple, Optional, Dict, Any
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .iltm_model import iLTM

from .tree_embedding import TreeEmbedding
from .realmlp_td_s_preprocessing import get_realmlp_td_s_pipeline_separated
from .utils import (
    seed_everything,
    full_main_forward,
    fine_tune_main_network,
    standardize_column_dtypes,
    check_stratification,
    compute_permutation_feature_importance,
    is_cuda_oom,
    clear_cuda_cache,
    compute_feature_target_correlations,
    select_top_correlated_features,
    detect_object_string_columns,
)
from .model_checkpoints import resolve_model_checkpoint

torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True

class PermutationImportanceMixin:
    """
    Adds feature_importance to iLTM estimators using permutation importance on ORIGINAL features.
    Supports single features or grouped sets like ('group_name', ['f1','f2']).
    """
    def feature_importance(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        features: list | None = None,
        groups: list[tuple[str, list[str]]] | None = None,
        n_repeats: int = 5,
        subsample_size: int | None = 5000,
        metric: str = "auto",
        random_state: int | None = 0,
        silent: bool = False,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute permutation feature importance on original columns.

        Parameters
        ----------
        X, y : held-out data. y must be on the original target scale for regression.
        features : list of column names or indices. May also include tuples ('group_name', [cols...]).
        groups : list of tuples ('group_name', [cols...]).
        n_repeats : number of shuffles per feature or group.
        subsample_size : cap rows used for speed. None means use all rows.
        metric : 'auto' or a specific metric.
                 regression: 'r2' (default), 'rmse', 'mse', 'mae'
                 classification: 'roc_auc' (default), 'log_loss', 'accuracy'
        random_state : RNG seed.
        silent : reduce logging.
        feature_names : names to apply if X is a numpy array.

        Returns
        -------
        DataFrame indexed by feature with columns: importance, stddev, p_value, n
        """
        check_is_fitted(self)

        # allow integer feature list for numpy X
        if isinstance(X, np.ndarray) and features is not None:
            # remap int indices to names
            feats = []
            for f in features:
                if isinstance(f, tuple):
                    name, cols = f
                    cols2 = [str(c) if isinstance(c, str) else f"f{int(c)}" for c in cols]
                    feats.append((name, cols2))
                else:
                    feats.append(f"f{int(f)}" if not isinstance(f, str) else f)
            features = feats
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X.shape[1])]

        # build predict functions
        def _predict_fn(df: pd.DataFrame) -> np.ndarray:
            return self.predict(df)

        def _predict_proba_fn(df: pd.DataFrame) -> np.ndarray:
            return self.predict_proba(df)  # type: ignore[attr-defined]

        # classification vs regression
        task = getattr(self, "task_type", "regression")
        proba_fn = _predict_proba_fn if task == "classification" and hasattr(self, "predict_proba") else None

        return compute_permutation_feature_importance(
            X=X,
            y=y,
            predict_fn=_predict_fn,
            predict_proba_fn=proba_fn,
            task_type=task,
            metric=metric,
            features=features,
            groups=groups,
            n_repeats=n_repeats,
            subsample_size=subsample_size,
            feature_names=feature_names,
            random_state=random_state,
            silent=silent,
        )


class _iLTMBase(BaseEstimator):
    """
    Shared base for iLTMClassifier and iLTMRegressor.
    """

    _model_cache: Dict[tuple, iLTM] = {}

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        n_ensemble: int = 16,
        batch_size: int = 4096,
        seed: int = 3,
        finetuning: bool = True,
        finetuning_optimizer: str = 'adamw',
        finetuning_data: str = 'entire_dataset',
        finetuning_val_frac: float = 0.1,
        finetuning_batch_size: int | None = 256,
        finetuning_dropout: float = 0.15,
        finetuning_max_steps: int = 4096,
        finetuning_lr: float = 0.00021136819819225127,
        finetuning_classification_val_metric: str = 'auto',  # options: 'auto', 'logloss', 'auc' ('auto': 'auc' binary, 'logloss' multiclass)
        initial_transformations_finetuning: bool = True,
        early_stopping_mode: str = "auto",
        patience_epochs: int = 50,
        patience_checks: int | None = None,
        val_checks_per_epoch_target: int = 4,
        val_check_interval_batches: int | None = None,
        max_train_batches_per_epoch: int | None = 500,
        finetuning_subset_frac: float | None = None,
        finetuning_subset_max_samples: int | None = 500_000,
        val_max_samples: int | None = 100_000,
        min_epochs: int = 0,
        cooldown_checks: int = 2,
        gradient_clip_norm: float = 0.9968664343329384,
        scheduler_min_lr: float = 0.00025787005984493423,
        checkpoint: str | None = "xgbrconcat",
        stratify_sampling: bool = False,
        feature_bagging: bool = False,
        feature_bagging_size: int = 3000,
        feature_bagging_type: str = 'uniform',
        cat_features: List[int] | None = None,
        task_type: str = None,  # overridden by subclasses
        preprocessing: str = 'realmlp_td_s_v0',
        dim_exp_type: str = 'rf',
        rf_size: int = 32768,
        n_dims: int = 512,
        pca_fit: str = 'reduced',
        pca_sampling: str = 'zeropad',
        pca_svd_driver: str | None = None,
        clip_data_value: float = 1000000,
        n_classes_limit: int = 100,
        hn_n_layers: int = 4,
        hn_hidden_size: int = 1024,
        hyper_dropout: float = 0.0,
        main_n_layers: int = 3,
        bottleneck_size: int = 0,
        tree_embedding: bool = False,
        tree_model: str = 'XGBoost_hist',
        tree_n_estimators: int = 200,
        tree_lr: float | None = 0.05848722909622601,
        tree_max_depth: int | None = 6,
        tree_min_samples_leaf: int | None = 90,
        tree_subsample: float | None = 0.5074398421774727,
        tree_feature_fraction: float | None = 0.8132119890393672,
        tree_data_split: str = 'dynamic',
        tree_for_each_predictor: bool = True,
        tree_use_default_params: bool = False,
        tree_select_best_model: bool = True,
        concat_tree_with_orig_features: bool = False,
        tree_max_leaves: int | None = None,
        tree_gamma: float | None = 1.4598703125721042,
        tree_l2_leaf_reg: float | None = 1.007789530064673,
        tree_bagging_temperature: float | None = 0.12093655368094158,
        onehot_max_features: bool = True,
        do_retrieval: bool = False,
        retrieval_alpha: float = 0.7024108748899226,
        retrieval_temperature: float = 1.921535071554998,
        retrieval_distance: str = 'euclidean',
        retrieval_alpha_finetuning: bool = False,
        retrieval_temperature_finetuning: bool = False,
        logging_level: int | str = logging.WARNING,
        clip_predictions: bool = True,
        normalize_predictions: bool = False,
        use_amp_inference: bool = False,
        use_amp_finetuning: bool = False,
        adaptive_memory: bool = True,
        gpu_mem_low_mb: int = 2048,
        gpu_mem_very_low_mb: int = 1024,
        min_finetuning_batch_size: int = 64,
        gpu_oom_retries: int = 2,
        auto_disable_retrieval_on_low_mem: bool = False,
        auto_amp_on_low_mem: bool = False,
        auto_stop_on_low_cpu_memory: bool = True,
        cpu_memory_limit_gb: float | None = None,
        cpu_memory_safety_margin_gb: float = 2.0,
        corr_select_k: int = 300,
        inference_storage_dtype: str | None = "float16",
    ) -> None:

        # Logging
        # NOTE (TabTune vendoring): upstream called setup_logging() here, which
        # resets the ROOT logger handlers and would clobber TabTune's own
        # logging configuration. We only adjust this module's logger level.
        self.logging_level = logging_level
        global logger
        logger = logging.getLogger(__name__)
        try:
            logger.setLevel(self.logging_level)
        except (TypeError, ValueError):
            pass

        # Resolve model checkpoint config if provided
        if checkpoint is not None:
            try:
                checkpoint_config = resolve_model_checkpoint(checkpoint)
                # Update parameters based on checkpoint configuration
                for key, value in checkpoint_config.items():
                    if key == 'checkpoint':
                        checkpoint = value
                    elif key == 'preprocessing':
                        preprocessing = value
                    elif key == 'tree_embedding':
                        tree_embedding = value
                    elif key == 'tree_model':
                        tree_model = value
                    elif key == 'do_retrieval':
                        do_retrieval = value
                    elif key == 'bottleneck_size':
                        bottleneck_size = value
                    elif key == 'concat_tree_with_orig_features':
                        concat_tree_with_orig_features = value
                    else:
                        logger.warning(f"Unknown parameter from checkpoint config: {key}")
            except ValueError as e:
                logger.error(f"Failed to resolve model checkpoint: {e}")
                raise

        # Store config
        # Store device as string in __dict__ for sklearn clone compatibility
        # but also store as torch.device for internal use
        if isinstance(device, str):
            device_str = device
        elif isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        # Store string version in __dict__ so sklearn's get_params sees it
        self.__dict__['device'] = device_str
        # Store torch.device for internal use (but don't let sklearn see it)
        self._device_torch = torch.device(device_str)
        self.n_ensemble = int(n_ensemble)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.finetuning = bool(finetuning)
        self.finetuning_optimizer = finetuning_optimizer
        self.finetuning_data = finetuning_data
        self.finetuning_val_frac = float(finetuning_val_frac)
        self.finetuning_batch_size = int(batch_size) if finetuning_batch_size is None else int(finetuning_batch_size)
        self.finetuning_dropout = float(finetuning_dropout)
        self.finetuning_max_steps = int(finetuning_max_steps)
        self.finetuning_lr = float(finetuning_lr)
        self.finetuning_classification_val_metric = finetuning_classification_val_metric
        self.initial_transformations_finetuning = bool(initial_transformations_finetuning)
        self.early_stopping_mode = early_stopping_mode
        self.patience_epochs = int(patience_epochs)
        self.patience_checks = None if patience_checks is None else int(patience_checks)
        self.val_checks_per_epoch_target = int(val_checks_per_epoch_target)
        self.val_check_interval_batches = val_check_interval_batches
        self.max_train_batches_per_epoch = None if max_train_batches_per_epoch is None else int(max_train_batches_per_epoch)
        self.finetuning_subset_frac = finetuning_subset_frac
        self.finetuning_subset_max_samples = finetuning_subset_max_samples
        self.val_max_samples = val_max_samples
        self.min_epochs = int(min_epochs)
        self.cooldown_checks = int(cooldown_checks)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.scheduler_min_lr = float(scheduler_min_lr)

        self.checkpoint = checkpoint

        self.stratify_sampling = bool(stratify_sampling)
        self.feature_bagging = bool(feature_bagging)
        self.feature_bagging_size = int(feature_bagging_size)
        self.feature_bagging_type = feature_bagging_type

        if cat_features is not None and not (isinstance(cat_features, list) and all(isinstance(i, int) for i in cat_features)):
            raise ValueError("cat_features must be a list of integer indices or None.")
        self.cat_features = cat_features if cat_features is not None else []
        self.task_type = task_type
        self.preprocessing = preprocessing
        self.dim_exp_type = dim_exp_type
        self.rf_size = int(rf_size)
        self.n_dims = int(n_dims)
        self.pca_fit = pca_fit
        self.pca_sampling = pca_sampling
        self.pca_svd_driver = pca_svd_driver
        self.clip_data_value = float(clip_data_value)
        self.n_classes_limit = int(n_classes_limit)
        self.hn_n_layers = int(hn_n_layers)
        self.hn_hidden_size = int(hn_hidden_size)
        self.hyper_dropout = float(hyper_dropout)
        self.main_n_layers = int(main_n_layers)
        self.bottleneck_size = int(bottleneck_size)

        self.tree_embedding = bool(tree_embedding)
        self.tree_model = tree_model
        self.tree_n_estimators = int(tree_n_estimators)
        self.tree_lr = tree_lr
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.tree_subsample = tree_subsample
        self.tree_feature_fraction = tree_feature_fraction
        self.tree_data_split = tree_data_split
        self.tree_for_each_predictor = bool(tree_for_each_predictor)
        self.tree_use_default_params = bool(tree_use_default_params)
        self.tree_select_best_model = bool(tree_select_best_model)
        self.concat_tree_with_orig_features = bool(concat_tree_with_orig_features)
        self.tree_max_leaves = tree_max_leaves
        self.tree_gamma = tree_gamma
        self.tree_l2_leaf_reg = tree_l2_leaf_reg
        self.tree_bagging_temperature = tree_bagging_temperature
        self.onehot_max_features = bool(onehot_max_features)

        if self.concat_tree_with_orig_features:
            if self.tree_max_depth is not None and self.tree_max_depth > 8:
                logger.warning(
                    f"concat_tree_with_orig_features=True, limiting tree_max_depth "
                    f"from {self.tree_max_depth} to 8 to avoid OOM errors."
                )
                self.tree_max_depth = 8

        # Retrieval
        self.do_retrieval = bool(do_retrieval)
        self.retrieval_alpha = float(retrieval_alpha)
        self.retrieval_temperature = float(retrieval_temperature)
        self.retrieval_distance = retrieval_distance
        self.retrieval_alpha_finetuning = bool(retrieval_alpha_finetuning)
        self.retrieval_temperature_finetuning = bool(retrieval_temperature_finetuning)

        # These flags will be respected only for regression
        self.clip_predictions = bool(clip_predictions) if self.task_type == 'regression' else False
        self.normalize_predictions = bool(normalize_predictions) if self.task_type == 'regression' else False
        # Disable AMP if tree embedding is used, to match previous behavior
        self.use_amp_inference = bool(use_amp_inference) if not self.tree_embedding else False
        self.use_amp_finetuning = bool(use_amp_finetuning) if not self.tree_embedding else False

        # Memory controls
        self.inference_chunk_rows = 10000
        self.retrieval_context_max_rows = 8192

        self.adaptive_memory = bool(adaptive_memory)
        self.gpu_mem_low_mb = int(gpu_mem_low_mb)
        self.gpu_mem_very_low_mb = int(gpu_mem_very_low_mb)
        self.min_finetuning_batch_size = int(min_finetuning_batch_size)
        self.gpu_oom_retries = int(gpu_oom_retries)
        self.auto_disable_retrieval_on_low_mem = bool(auto_disable_retrieval_on_low_mem)
        self.auto_amp_on_low_mem = bool(auto_amp_on_low_mem)
        self.auto_stop_on_low_cpu_memory = bool(auto_stop_on_low_cpu_memory)
        self.cpu_memory_limit_gb = None if cpu_memory_limit_gb is None else float(cpu_memory_limit_gb)
        self.cpu_memory_safety_margin_gb = float(cpu_memory_safety_margin_gb)

        # Correlation-based feature selection
        self.corr_select_k = int(corr_select_k)
        self.inference_storage_dtype = inference_storage_dtype
        self._inference_storage_torch_dtype = self._resolve_inference_storage_dtype(inference_storage_dtype)

        # Placeholders
        self.tr_: TreeEmbedding | List[TreeEmbedding] | None = None
        seed_everything(self.seed)

        self.model_path = self.checkpoint
        self._model: iLTM | None = None

        self.predictors_: List[dict] = []
        self.preprocessors_: List[dict] = []

    @staticmethod
    def _resolve_inference_storage_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(
                    "inference_storage_dtype must be one of None, 'float32', "
                    "'float16', or 'bfloat16'."
                )
            return None if dtype == torch.float32 else dtype

        normalized = str(dtype).lower()
        aliases = {
            "none": None,
            "float32": None,
            "fp32": None,
            "torch.float32": None,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "torch.float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "torch.bfloat16": torch.bfloat16,
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported inference_storage_dtype={dtype!r}. Expected one of "
                "None, 'float32', 'float16', or 'bfloat16'."
            )
        return aliases[normalized]

    def _module_to_device_and_dtype(
        self,
        module: nn.Module,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> nn.Module:
        if dtype is None:
            return module.to(device)
        return module.to(device=device, dtype=dtype)

    @staticmethod
    def _read_int_file(path: str) -> int | None:
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            return None
        if raw == "" or raw.lower() == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        if value <= 0 or value >= 1 << 60:
            return None
        return value

    @classmethod
    def _get_cgroup_memory_limit_bytes(cls) -> int | None:
        candidates = [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]
        values = [value for path in candidates if (value := cls._read_int_file(path)) is not None]
        return min(values) if values else None

    @classmethod
    def _get_cgroup_memory_usage_bytes(cls) -> int | None:
        candidates = [
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ]
        values = [value for path in candidates if (value := cls._read_int_file(path)) is not None]
        return min(values) if values else None

    @staticmethod
    def _get_autogluon_memory_limit_bytes() -> int | None:
        raw = os.environ.get("AG_MEMORY_LIMIT_IN_GB")
        if raw is None:
            return None
        try:
            value_gb = float(raw)
        except ValueError:
            return None
        if value_gb <= 0:
            return None
        return int(value_gb * (1024 ** 3))

    @staticmethod
    def _get_slurm_memory_limit_bytes() -> int | None:
        mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node:
            try:
                value_mb = int(mem_per_node)
            except ValueError:
                value_mb = 0
            if value_mb > 0:
                return value_mb * 1024 * 1024

        mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
        if mem_per_cpu:
            try:
                value_mb = int(mem_per_cpu)
            except ValueError:
                value_mb = 0
            cpus_raw = (
                os.environ.get("SLURM_CPUS_PER_TASK")
                or os.environ.get("SLURM_CPUS_ON_NODE")
                or os.environ.get("SLURM_JOB_CPUS_PER_NODE", "").split("(", 1)[0]
            )
            try:
                cpus = int(cpus_raw)
            except (TypeError, ValueError):
                cpus = 1
            if value_mb > 0 and cpus > 0:
                return value_mb * cpus * 1024 * 1024
        return None

    def _get_effective_cpu_memory_limit_bytes(self) -> int | None:
        limits: list[int] = []
        if self.cpu_memory_limit_gb is not None and self.cpu_memory_limit_gb > 0:
            limits.append(int(self.cpu_memory_limit_gb * (1024 ** 3)))
        ag_limit = self._get_autogluon_memory_limit_bytes()
        if ag_limit is not None:
            limits.append(ag_limit)
        cgroup_limit = self._get_cgroup_memory_limit_bytes()
        if cgroup_limit is not None:
            limits.append(cgroup_limit)
        slurm_limit = self._get_slurm_memory_limit_bytes()
        if slurm_limit is not None:
            limits.append(slurm_limit)
        return min(limits) if limits else None

    @classmethod
    def _get_process_rss_bytes(cls) -> int | None:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
        return None

    def _get_cpu_memory_usage_bytes(self, limit_bytes: int | None = None) -> int | None:
        cgroup_limit = self._get_cgroup_memory_limit_bytes()
        if cgroup_limit is not None and limit_bytes is not None and cgroup_limit <= int(limit_bytes * 1.05):
            return self._get_cgroup_memory_usage_bytes() or self._get_process_rss_bytes()
        return self._get_process_rss_bytes() or self._get_cgroup_memory_usage_bytes()

    @classmethod
    def _estimate_object_storage_bytes(cls, obj: Any, seen: set[int] | None = None) -> int:
        if seen is None:
            seen = set()
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return 0

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, nn.Module):
            total = 0
            for param in obj.parameters(recurse=True):
                total += param.numel() * param.element_size()
            for buffer in obj.buffers(recurse=True):
                total += buffer.numel() * buffer.element_size()
            return total
        if isinstance(obj, dict):
            return sum(cls._estimate_object_storage_bytes(value, seen) for value in obj.values())
        if isinstance(obj, (list, tuple, set)):
            return sum(cls._estimate_object_storage_bytes(value, seen) for value in obj)
        return 0

    def _estimate_predictors_storage_bytes(self) -> int:
        return sum(self._estimate_object_storage_bytes(predictor) for predictor in self.predictors_)

    def _should_stop_for_cpu_memory_before_predictor(self, next_predictor_index: int) -> tuple[bool, dict[str, float | int] | None]:
        if not self.auto_stop_on_low_cpu_memory or not self.predictors_:
            return False, None

        limit_bytes = self._get_effective_cpu_memory_limit_bytes()
        usage_bytes = self._get_cpu_memory_usage_bytes(limit_bytes=limit_bytes)
        if limit_bytes is None or usage_bytes is None:
            return False, None

        current_model_bytes = self._estimate_predictors_storage_bytes()
        if current_model_bytes <= 0:
            return False, None

        avg_predictor_bytes = current_model_bytes / len(self.predictors_)
        projected_model_bytes = current_model_bytes + avg_predictor_bytes
        margin_bytes = max(0, int(self.cpu_memory_safety_margin_gb * (1024 ** 3)))
        # During fold return or serialization, a serialized copy can coexist
        # with the live fitted model. Project that copy without materializing it.
        projected_peak_bytes = usage_bytes + avg_predictor_bytes + projected_model_bytes + margin_bytes

        if projected_peak_bytes <= limit_bytes:
            return False, None

        return True, {
            "next_predictor_index": next_predictor_index,
            "limit_gb": limit_bytes / (1024 ** 3),
            "usage_gb": usage_bytes / (1024 ** 3),
            "available_gb": (limit_bytes - usage_bytes) / (1024 ** 3),
            "avg_predictor_gb": avg_predictor_bytes / (1024 ** 3),
            "projected_model_gb": projected_model_bytes / (1024 ** 3),
            "projected_peak_gb": projected_peak_bytes / (1024 ** 3),
            "margin_gb": margin_bytes / (1024 ** 3),
            "predictors_fit": len(self.predictors_),
            "n_ensemble": self.n_ensemble,
        }

    @classmethod
    def __sklearn_tags__(cls, estimator=None):
        try:
            from sklearn.utils._tags import _DEFAULT_TAGS
            base = dict(_DEFAULT_TAGS)
        except Exception:
            base = {}
        base.update({
            "requires_fit": True,
            "allow_nan": True,
            "non_deterministic": False,
            "poor_score": False,
        })
        return SimpleNamespace(**base)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # Ensure device is returned as string for sklearn clone compatibility
        # Get the string version from __dict__ if available
        if 'device' in self.__dict__:
            params['device'] = self.__dict__['device']
        elif 'device' in params and isinstance(params['device'], torch.device):
            params['device'] = str(params['device'])
        return params
    
    @property
    def device(self):
        """Return the device as a torch.device object for internal use."""
        return self._device_torch
    
    @device.setter
    def device(self, value):
        """Set the device, updating both string and torch.device versions."""
        if isinstance(value, str):
            device_str = value
        elif isinstance(value, torch.device):
            device_str = str(value)
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__dict__['device'] = device_str
        self._device_torch = torch.device(device_str)

    # -----------------------------
    # Model IO
    # -----------------------------
    def _initialize_model(self) -> iLTM:
        arch_params = (
            self.n_dims, self.hn_n_layers, self.hn_hidden_size, self.clip_data_value,
            self.rf_size, self.n_classes_limit, self.dim_exp_type, self.bottleneck_size,
            self.main_n_layers, self.pca_fit, self.pca_svd_driver,
            self.hyper_dropout, self.pca_sampling
        )
        cache_key = (self.model_path, str(self.device)) + arch_params
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = iLTM(
            n_dims=self.n_dims,
            hn_n_layers=self.hn_n_layers,
            hn_hidden_size=self.hn_hidden_size,
            clip_data_value=self.clip_data_value,
            rf_size=self.rf_size,
            n_classes_limit=self.n_classes_limit,
            dim_exp_type=self.dim_exp_type,
            bottleneck_size=self.bottleneck_size,
            main_n_layers=self.main_n_layers,
            pca_fit=self.pca_fit,
            pca_svd_driver=self.pca_svd_driver,
            hyper_dropout=self.hyper_dropout,
            pca_sampling=self.pca_sampling
        ).to(self.device)

        state_dict: Dict[str, Any] = {}

        path_exists = bool(self.model_path) and os.path.exists(self.model_path)  # type: ignore[arg-type]
        if path_exists:
            logger.info(f"Loading model from {self.model_path} on {self.device} device...")
            try:
                state_dict = torch.load(self.model_path, map_location=torch.device(self.device), weights_only=True)  # type: ignore[arg-type]
            except TypeError:
                state_dict = torch.load(self.model_path, map_location=torch.device(self.device))  # type: ignore[arg-type]
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}, cannot proceed without a valid checkpoint.")

        # Remap keys for backward compatibility
        modified_state_dict: Dict[str, Any] = {}
        for key in state_dict:
            if ('hypernetworks' in key or 'hn_emb_to_weights' in key) and 'hypernetwork_block' not in key:
                new_key = 'hypernetwork_block.' + key
            elif 'norm' in key and 'initial_transformation_block' not in key:
                new_key = 'initial_transformation_block.' + key
            elif '.rf' in key:
                continue
            elif 'initial_transformation_block.norm' in key:
                continue
            else:
                new_key = key
            modified_state_dict[new_key] = state_dict[key]

        if len(modified_state_dict) > 0:
            try:
                model.load_state_dict(modified_state_dict)
                logger.info(f"Model loaded from {self.model_path} on {self.device} device.")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model file not found at {self.model_path}") from e
            except RuntimeError as e:
                raise RuntimeError(f"Failed to load the model from {self.model_path}") from e

        model.eval()
        self._model_cache[cache_key] = model
        return model

    def _get_tags(self) -> dict:
        tags = super()._get_tags()
        tags["allow_nan"] = True
        return tags
    

    def _auto_tune_for_memory(self):
        from .utils import get_gpu_memory_info
        info = get_gpu_memory_info(self.device)
        if info is None or not self.adaptive_memory:
            return

        free = info["free_mb"]
        try:
            # Pre-initialization adjustments that affect model shape
            if free < self.gpu_mem_very_low_mb:
                # Reduce expansion size and latent dims when VRAM is very tight
                if self.rf_size > 2**14:
                    logger.warning("Low VRAM: reducing rf_size from %d to %d", self.rf_size, 2**14)
                    self.rf_size = 2**14

            if free < self.gpu_mem_low_mb:
                # Reduce finetuning batch size, enable AMP, and disable retrieval
                if self.finetuning_batch_size > self.min_finetuning_batch_size:
                    old = self.finetuning_batch_size
                    self.finetuning_batch_size = max(self.min_finetuning_batch_size, old // 2)
                    logger.warning("Low VRAM: finetuning_batch_size %d -> %d", old, self.finetuning_batch_size)

                if self.auto_amp_on_low_mem and not self.use_amp_finetuning and not self.tree_embedding:
                    self.use_amp_finetuning = True
                    logger.warning("Low VRAM: enabling AMP for finetuning")

                if self.auto_disable_retrieval_on_low_mem and self.do_retrieval:
                    self.do_retrieval = False
                    logger.warning("Low VRAM: disabling retrieval during finetuning")

            # Tree embedding estimator count can be very costly on GPU
            if self.tree_embedding and free < 4096:
                if self.tree_n_estimators > 200:
                    logger.warning("Low VRAM: reducing tree_n_estimators %d -> %d", self.tree_n_estimators, 200)
                    self.tree_n_estimators = 200
        except Exception:
            pass


    # -----------------------------
    # Preprocessing
    # -----------------------------
    def _preprocess_fitting_data(
        self,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        is_classification: bool
    ) -> Tuple[np.ndarray, np.ndarray, dict]:

        logger.debug(f"Preprocessing x {np.shape(x)} with type: {self.preprocessing}")
        preprocessing_objects: Dict[str, Any] = {}

        if self.preprocessing == 'minimal':
            logger.debug("Preprocessing with minimal")
            if not isinstance(x, (np.ndarray, pd.DataFrame)) and not isinstance(y, (np.ndarray, pd.Series)):
                x, y = check_X_y(x, y)
            if not isinstance(x, (np.ndarray, pd.DataFrame)):
                x = check_array(x)
            if not isinstance(y, (np.ndarray, pd.Series)):
                y = np.array(y)
            x = np.array(x)
            y = np.array(y)

            if not self.tree_embedding:
                preprocessing_objects['cat_features'] = self.cat_features if self.cat_features is not None else []
            else:
                preprocessing_objects['cat_features'] = []

            preprocessing_objects['num_imputer'] = SimpleImputer(missing_values=np.nan, strategy="mean")
            if len(x.shape) == 2:
                self._all_feature_idxs = np.arange(x.shape[1])
            else:
                raise ValueError("Reshape your data")

            preprocessing_objects['numerical_feature_idxs'] = np.setdiff1d(self._all_feature_idxs, preprocessing_objects['cat_features'])
            if len(preprocessing_objects['numerical_feature_idxs']) > 0:
                preprocessing_objects['num_imputer'].fit(x[:, preprocessing_objects['numerical_feature_idxs']])
                x[:, preprocessing_objects['numerical_feature_idxs']] = preprocessing_objects['num_imputer'].transform(
                    x[:, preprocessing_objects['numerical_feature_idxs']]
                )
                logger.debug(f"Len numerical features: {len(preprocessing_objects['numerical_feature_idxs'])}")

            if len(preprocessing_objects['cat_features']) > 0:
                preprocessing_objects['cat_imputer'] = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                preprocessing_objects['cat_imputer'].fit(x[:, preprocessing_objects['cat_features']])
                x[:, preprocessing_objects['cat_features']] = preprocessing_objects['cat_imputer'].transform(
                    x[:, preprocessing_objects['cat_features']]
                )

                x_df = pd.DataFrame(x)
                preprocessing_objects['one_hot_encoder'] = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), preprocessing_objects['cat_features']),
                    ],
                    remainder="passthrough",
                )
                preprocessing_objects['one_hot_encoder'].fit(x_df)
                x = preprocessing_objects['one_hot_encoder'].transform(x_df)

            x, y = check_X_y(x, y)
            preprocessing_objects['scaler'] = StandardScaler()
            preprocessing_objects['scaler'].fit(x)
            x = preprocessing_objects['scaler'].transform(x)

        elif self.preprocessing == 'realmlp_td_s_v0':
            logger.debug("Preprocessing with realmlp_td_s_v0")
            preprocessing_objects['pipeline'] = get_realmlp_td_s_pipeline_separated(cat_features=self.cat_features)

            if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
                pass
            else:
                x = pd.DataFrame(x)
                x.columns = range(x.shape[1])
                x = standardize_column_dtypes(x)

            if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
                x = x.astype(np.float32, copy=False)

            preprocessing_objects['pipeline'].fit(x)
            x_num, x_cat = preprocessing_objects['pipeline'].transform(x)

            # nan checks numeric
            preprocessing_objects['all_nan_num_columns'] = np.isnan(x_num).all(axis=0)
            if np.all(preprocessing_objects['all_nan_num_columns']) and x_num.shape[1] > 0:
                raise ValueError("All numeric columns are NaN when preprocessing with realmlp_td_s_v0")
            if np.any(preprocessing_objects['all_nan_num_columns']) and x_num.shape[1] > 0:
                logger.warning(f"Removing all-NaN numeric columns: {np.where(preprocessing_objects['all_nan_num_columns'])[0]}")
                x_num = x_num[:, ~preprocessing_objects['all_nan_num_columns']]

            # nan checks categorical
            preprocessing_objects['all_nan_cat_columns'] = np.isnan(x_cat).all(axis=0)
            if np.all(preprocessing_objects['all_nan_cat_columns']) and x_cat.shape[1] > 0:
                raise ValueError("All categorical columns are NaN when preprocessing with realmlp_td_s_v0")
            if np.any(preprocessing_objects['all_nan_cat_columns']) and x_cat.shape[1] > 0:
                logger.warning(f"Removing all-NaN categorical columns: {np.where(preprocessing_objects['all_nan_cat_columns'])[0]}")
                x_cat = x_cat[:, ~preprocessing_objects['all_nan_cat_columns']]

            if np.isnan(x_num).any() or np.isnan(x_cat).any():
                logger.warning("There are still NaN values in the dataset, encoding as 0.")
                x_num[np.isnan(x_num)] = 0
                x_cat[np.isnan(x_cat)] = 0

            x = np.concatenate([x_num, x_cat], axis=1)

        elif self.preprocessing == 'none' or self.preprocessing is None:
            logger.debug("Not preprocessing data")
            x, y = check_X_y(x, y)
            x = np.array(x)
            y = np.array(y)
        else:
            raise ValueError(f"Invalid preprocessing type: {self.preprocessing}")

        # y handling differs by task
        if is_classification:
            check_classification_targets(y)
            y = column_or_1d(y, warn=True)
        else:
            y = np.array(y, dtype=np.float32)

        # Correlation-based feature selection (posneg_topk strategy)
        if self.corr_select_k > 0 and x.shape[1] > self.corr_select_k:
            logger.debug(f"Applying correlation-based feature selection: selecting up to {self.corr_select_k} features using posneg_topk strategy (only non-zero correlations)")
            r = compute_feature_target_correlations(x, y)
            selected_indices = select_top_correlated_features(r, self.corr_select_k)
            x = x[:, selected_indices]
            preprocessing_objects['corr_selected_indices'] = selected_indices
        else:
            preprocessing_objects['corr_selected_indices'] = None

        return x, y, preprocessing_objects

    def _preprocess_test_data(self, x_test: np.ndarray | pd.DataFrame, preprocessing_objects: dict) -> Tensor:
        logger.debug(f"Preprocessing test data x_test {np.shape(x_test)} with type: {self.preprocessing}")
        if not isinstance(x_test, (np.ndarray, pd.DataFrame)):
            x_test = check_array(x_test)
        x_test = np.array(x_test)

        if self.preprocessing == 'minimal':
            if len(x_test.shape) == 1:
                raise ValueError("Reshape your data")
            if 'numerical_feature_idxs' in preprocessing_objects and len(preprocessing_objects['numerical_feature_idxs']) > 0:
                x_test[:, preprocessing_objects['numerical_feature_idxs']] = preprocessing_objects['num_imputer'].transform(
                    x_test[:, preprocessing_objects['numerical_feature_idxs']]
                )

            if 'cat_imputer' in preprocessing_objects:
                x_test[:, preprocessing_objects['cat_features']] = preprocessing_objects['cat_imputer'].transform(
                    x_test[:, preprocessing_objects['cat_features']]
                )
                x_test_df = pd.DataFrame(x_test)
                x_test = preprocessing_objects['one_hot_encoder'].transform(x_test_df)

            x_test = check_array(x_test)
            x_test = preprocessing_objects['scaler'].transform(x_test)

        elif self.preprocessing == 'realmlp_td_s_v0':
            x_df = pd.DataFrame(x_test)
            x_df.columns = range(x_df.shape[1])
            x_df = standardize_column_dtypes(x_df)
            x_num, x_cat = preprocessing_objects['pipeline'].transform(x_df)

            if np.any(preprocessing_objects['all_nan_num_columns']):
                x_num = x_num[:, ~preprocessing_objects['all_nan_num_columns']]
            if np.any(preprocessing_objects['all_nan_cat_columns']):
                x_cat = x_cat[:, ~preprocessing_objects['all_nan_cat_columns']]

            if np.isnan(x_num).any() or np.isnan(x_cat).any():
                x_num[np.isnan(x_num)] = 0
                x_cat[np.isnan(x_cat)] = 0

            x_test = np.concatenate([x_num, x_cat], axis=1)

        elif self.preprocessing == 'none' or self.preprocessing is None:
            x_test = np.array(x_test)
            x_test = check_array(x_test)
        else:
            raise ValueError(f"Invalid preprocessing type: {self.preprocessing}")

        # Apply correlation-based feature selection if used during training
        if preprocessing_objects.get('corr_selected_indices') is not None:
            selected_indices = preprocessing_objects['corr_selected_indices']
            x_test = x_test[:, selected_indices]

        if isinstance(x_test, dict):
            return {
                'x_num': torch.as_tensor(x_test['x_num'], dtype=torch.float),
                'x_cat': torch.as_tensor(x_test['x_cat'], dtype=torch.float)
            }
        else:
            return torch.as_tensor(x_test, dtype=torch.float)

    # -----------------------------
    # Sampling and device helpers
    # -----------------------------
    def _sample_data(
        self,
        X: Tensor | Dict[str, Tensor],
        y: Tensor,
        pca_sampling: str = 'repeat'
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Selects a batch of samples (X_pred, y_pred) from X,y
        Returns: (X_pred, y_pred, feature_idxs)
        """
        n_features = X['x_num'].shape[1] + X['x_cat'].shape[1] if isinstance(X, dict) else X.shape[1]

        feature_idxs = None
        if self.feature_bagging and n_features >= self.feature_bagging_size:
            logger.debug("Performing feature bagging")
            if self.feature_bagging_type == 'uniform':
                feature_idxs = torch.randperm(n_features)[: self.feature_bagging_size]
            elif self.feature_bagging_type == 'std_multinomial':
                stds = torch.std(X if not isinstance(X, dict) else torch.cat([X['x_num'], X['x_cat']], dim=1), dim=0)  # type: ignore[arg-type]
                feature_idxs = torch.multinomial(stds, self.feature_bagging_size, replacement=False)
            else:
                raise ValueError(f"Invalid feature bagging type: {self.feature_bagging_type}")
            if not isinstance(X, dict):
                X = X[:, feature_idxs]

        if self.stratify_sampling and self.task_type != 'regression':
            logger.debug("Using stratified sampling")
            classes, _ = torch.unique(y, return_counts=True)
            samples_per_class = max(1, self.batch_size // len(classes))
            sampled_indices = []
            for cls in classes:
                cls_indices = (y == cls).nonzero(as_tuple=True)[0]
                n_samples = min(samples_per_class, len(cls_indices))
                cls_sampled_indices = cls_indices[torch.randperm(len(cls_indices))[:n_samples]]
                sampled_indices.append(cls_sampled_indices)
            sampled_indices = torch.cat(sampled_indices)
            sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
        else:
            len_X = X['x_num'].shape[0] if isinstance(X, dict) else X.shape[0]
            sampled_indices = torch.randperm(len_X)[: self.batch_size]

        if isinstance(X, dict):
            y_pred = y[sampled_indices]
            X_pred = {
                'x_num': X['x_num'][sampled_indices],
                'x_cat': X['x_cat'][sampled_indices]
            }
        else:
            X_pred, y_pred = X[sampled_indices].flatten(start_dim=1), y[sampled_indices]

        if y_pred.shape[0] < self.n_dims:
            if pca_sampling == 'repeat':
                n_repeats = math.ceil(self.n_dims / y_pred.shape[0])
                if isinstance(X_pred, dict):
                    X_pred['x_num'] = X_pred['x_num'].repeat((n_repeats, 1))
                    X_pred['x_cat'] = X_pred['x_cat'].repeat((n_repeats, 1))
                else:
                    X_pred = X_pred.repeat((n_repeats, 1))
                y_pred = y_pred.repeat(n_repeats)
            elif pca_sampling == 'bootstrap':
                additional_samples_needed = self.n_dims - y_pred.shape[0]
                bootstrap_indices = torch.randint(0, len_X, (additional_samples_needed,))
                if isinstance(X_pred, dict):
                    X_pred['x_num'] = torch.cat([X_pred['x_num'], X['x_num'][bootstrap_indices]])
                    X_pred['x_cat'] = torch.cat([X_pred['x_cat'], X['x_cat'][bootstrap_indices]])
                else:
                    X_pred = torch.cat([X_pred, X[bootstrap_indices]])
                y_pred = torch.cat([y_pred, y[bootstrap_indices]])
            elif pca_sampling == 'zeropad':
                pass
            else:
                raise ValueError(f"Invalid PCA sampling strategy: {pca_sampling}")

        return X_pred, y_pred, feature_idxs

    def _move_predictor_to_device(self, predictor: dict, device: str | torch.device | None = None) -> dict:
        if device is None:
            device = self.device
        target = torch.device(device)
        for key in predictor:
            if key in ['rf', 'pca', 'norm']:
                if predictor[key] is not None:
                    dtype = torch.float32 if key == "rf" else None
                    predictor[key] = self._module_to_device_and_dtype(predictor[key], device=target, dtype=dtype)
            elif key == 'main_network':
                for i, layer in enumerate(predictor[key]):
                    predictor[key][i] = layer.to(target)
            elif isinstance(predictor[key], (nn.Module, nn.Sequential)):
                predictor[key] = predictor[key].to(target)
        gc.collect()
        return predictor

    def _move_predictor_to_cpu(self, predictor: dict) -> dict:
        storage_dtype = self._inference_storage_torch_dtype
        for key in predictor:
            if key in ['rf', 'pca', 'norm']:
                if predictor[key] is not None:
                    dtype = storage_dtype if key == "rf" else None
                    predictor[key] = self._module_to_device_and_dtype(predictor[key], device="cpu", dtype=dtype)
            elif key == 'main_network':
                for i, layer in enumerate(predictor[key]):
                    predictor[key][i] = layer.to("cpu")
            elif isinstance(predictor[key], (nn.Module, nn.Sequential)):
                predictor[key] = predictor[key].to("cpu")
        gc.collect()
        return predictor

    # -----------------------------
    # Predictor generation and forward
    # -----------------------------
    def _generate_predictor(
        self,
        X: Tensor | Dict[str, Tensor],
        y: Tensor,
        *,
        n_outputs: int,
        X_val: Tensor | Dict[str, Tensor] | None = None,
        y_val: Tensor | None = None,
        fit_deadline: float | None = None,
        fit_time_cushion_frac: float = 0.001,    # (0.1% headroom)
    ) -> dict | None:
        X_pred, y_pred, feature_bagging_idxs = self._sample_data(X, y, pca_sampling=self.pca_sampling)
        X_pred, y_pred = (
            ({'x_num': X_pred['x_num'].to(self.device), 'x_cat': X_pred['x_cat'].to(self.device)} if isinstance(X_pred, dict) else X_pred.to(self.device)),
            y_pred.to(self.device)
        )

        with torch.no_grad():
            rf, pca, main_network, norm = self._model(X_pred, y_pred, n_outputs)  # type: ignore[operator]

        retrieval_parameters = {
            "do_retrieval": self.do_retrieval,
            "retrieval_alpha": self.retrieval_alpha,
            "retrieval_temperature": self.retrieval_temperature,
            "retrieval_distance": self.retrieval_distance
        }

        if self.finetuning:
            # Feature bagging on the full set, if any
            if self.feature_bagging and feature_bagging_idxs is not None and not isinstance(X, dict):
                X_ = X[:, feature_bagging_idxs]
            else:
                X_ = X

            if X_val is not None:
                if self.feature_bagging and feature_bagging_idxs is not None and not isinstance(X_val, dict):
                    X_val = X_val[:, feature_bagging_idxs]
                if isinstance(X_val, dict):
                    X_val = {'x_num': X_val['x_num'].to(self.device), 'x_cat': X_val['x_cat'].to(self.device)}
                else:
                    X_val = X_val.to(self.device)
                if y_val is not None:
                    y_val = y_val.to(self.device)

            attempts = 0
            cur_bs = int(self.finetuning_batch_size)
            cur_amp = bool(self.use_amp_finetuning)
            cur_retrieval = bool(self.do_retrieval)
            cur_init_tf_finetune = bool(self.initial_transformations_finetuning)

            while True:
                try:
                    fine_tuned_parts = fine_tune_main_network(
                        cfg=vars(self._model), X=X_, y=y, n_classes=n_outputs,
                        rf=rf, pca=pca, main_network=main_network, norm=norm,
                        device=self.device, max_epochs=self.finetuning_max_steps, batch_size=cur_bs,
                        finetuning_optimizer=self.finetuning_optimizer, finetuning_lr=self.finetuning_lr, finetuning_data=self.finetuning_data,
                        finetuning_dropout=self.finetuning_dropout, X_val=X_val, y_val=y_val,
                        val_check_interval_batches=self.val_check_interval_batches,
                        do_retrieval=cur_retrieval, retrieval_alpha=self.retrieval_alpha,
                        retrieval_temperature=self.retrieval_temperature, retrieval_distance=self.retrieval_distance,
                        retrieval_alpha_finetuning=self.retrieval_alpha_finetuning,
                        retrieval_temperature_finetuning=self.retrieval_temperature_finetuning,
                        initial_transformations_finetuning=cur_init_tf_finetune,
                        gradient_clip_norm=self.gradient_clip_norm, scheduler_min_lr=self.scheduler_min_lr,
                        use_amp_finetuning=cur_amp, early_stopping_mode=self.early_stopping_mode,
                        patience_epochs=self.patience_epochs, patience_checks=self.patience_checks,
                        val_checks_per_epoch_target=self.val_checks_per_epoch_target,
                        max_train_batches_per_epoch=self.max_train_batches_per_epoch,
                        finetuning_subset_frac=self.finetuning_subset_frac,
                        finetuning_subset_max_samples=self.finetuning_subset_max_samples,
                        val_max_samples=self.val_max_samples,
                        min_epochs=self.min_epochs, cooldown_checks=self.cooldown_checks,
                        classification_val_metric=self.finetuning_classification_val_metric,
                        finetuning_val_frac=self.finetuning_val_frac,
                        fit_deadline=fit_deadline,
                        fit_time_cushion_frac=fit_time_cushion_frac,
                    )
                    break  # success

                except Exception as e:
                    logger.error(f"Error during finetuning: {e}")
                    # Treat any error as OOM only if it looks like OOM
                    if not is_cuda_oom(e) or attempts >= self.gpu_oom_retries:
                        raise
                    else:
                        logger.debug("CUDA OOM detected, retrying with reduced resources")
                    attempts += 1
                    clear_cuda_cache()
                    # Backoff policy: halve batch, enable AMP, disable retrieval, disable init transforms finetuning
                    new_bs = max(self.min_finetuning_batch_size, cur_bs // 2)
                    if new_bs < cur_bs:
                        logger.warning("CUDA OOM during finetuning. Reducing batch size %d -> %d", cur_bs, new_bs)
                        cur_bs = new_bs
                        continue
                    if not cur_amp and not self.tree_embedding:
                        logger.warning("CUDA OOM during finetuning. Enabling AMP.")
                        cur_amp = True
                        continue
                    if cur_retrieval:
                        logger.warning("CUDA OOM during finetuning. Disabling retrieval.")
                        cur_retrieval = False
                        continue
                    if cur_init_tf_finetune:
                        logger.warning("CUDA OOM during finetuning. Freezing initial transformations.")
                        cur_init_tf_finetune = False
                        continue
                    # If we reach here, nothing else to relax
                    raise

            rf = fine_tuned_parts.get("rf", rf)
            pca = fine_tuned_parts.get("pca", pca)
            main_network = fine_tuned_parts.get("main_network", main_network)
            norm = fine_tuned_parts.get("norm", norm)
            retrieval_parameters["retrieval_alpha"] = fine_tuned_parts.get("retrieval_alpha", self.retrieval_alpha)
            retrieval_parameters["retrieval_temperature"] = fine_tuned_parts.get("retrieval_temperature", self.retrieval_temperature)

        predictor = {
            "feature_bagging_idxs": feature_bagging_idxs,
            "rf": rf,
            "pca": pca,
            "norm": norm,
            "main_network": main_network,
            "X_ctxt_superset": None,
            "y_ctxt_superset": None,
            "retrieval_parameters": retrieval_parameters,
            "timed_out": bool(fine_tuned_parts.get("timed_out", False)) if self.finetuning else False
        }

        # Retrieval context cap
        if self.do_retrieval:
            if isinstance(X, dict):
                num_rows = X['x_num'].shape[0]
            else:
                num_rows = X.shape[0]
            max_ctxt = int(self.retrieval_context_max_rows)
            sample_size = min(max_ctxt, int(num_rows))
            if sample_size < num_rows:
                idx = torch.randperm(num_rows)[:sample_size]
            else:
                idx = torch.arange(num_rows)
            if isinstance(X, dict):
                predictor["X_ctxt_superset"] = {
                    'x_num': X['x_num'][idx].cpu(),
                    'x_cat': X['x_cat'][idx].cpu()
                }
            else:
                predictor["X_ctxt_superset"] = X[idx].cpu()
            predictor["y_ctxt_superset"] = y[idx].cpu()

        return self._move_predictor_to_cpu(predictor)

    def _forward_pass_predictor(self, predictor: dict, X: Tensor | Dict[str, Tensor], n_outputs: int) -> Tensor:
        predictor = self._move_predictor_to_device(predictor, device=self.device)

        feature_bagging_idxs = predictor['feature_bagging_idxs']
        rf = predictor['rf']
        pca = predictor['pca']
        norm = predictor['norm']
        main_network = predictor['main_network']
        X_ctxt_superset = predictor['X_ctxt_superset']
        y_ctxt_superset = predictor['y_ctxt_superset']
        do_retrieval = predictor['retrieval_parameters']['do_retrieval']
        retrieval_alpha = predictor['retrieval_parameters']['retrieval_alpha']
        retrieval_temperature = predictor['retrieval_parameters']['retrieval_temperature']
        retrieval_distance = predictor['retrieval_parameters']['retrieval_distance']

        if isinstance(X, dict):
            x_num = X['x_num']
            x_cat = X['x_cat']
            x_num_features = x_num.shape[1]
            X_concat = torch.cat([x_num, x_cat], dim=1)
            if X_ctxt_superset is not None:
                assert isinstance(X_ctxt_superset, dict)
                x_ctxt_num = X_ctxt_superset['x_num']
                x_ctxt_cat = X_ctxt_superset['x_cat']
                assert x_ctxt_num.shape[1] == x_num_features
                X_ctxt_superset_concat = torch.cat([x_ctxt_num, x_ctxt_cat], dim=1)
            else:
                X_ctxt_superset_concat = None
        else:
            X_concat = X
            X_ctxt_superset_concat = X_ctxt_superset

        if self.feature_bagging and feature_bagging_idxs is not None:
            X_concat = X_concat[:, feature_bagging_idxs]
            if X_ctxt_superset_concat is not None:
                X_ctxt_superset_concat = X_ctxt_superset_concat[:, feature_bagging_idxs]

        ds = torch.utils.data.TensorDataset(X_concat)
        bs = int(self.batch_size)
        outs = []

        while True:
            try:
                loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
                for (Xb,) in loader:
                    Xb = Xb.to(self.device)
                    model_cfg = vars(self._model)
                    out = full_main_forward(
                        Xb,
                        n_outputs,
                        bs,
                        model_cfg,
                        rf, pca, norm, main_network, self.device, self.use_amp_inference,
                        do_retrieval, X_ctxt_superset_concat, y_ctxt_superset,
                        retrieval_alpha, retrieval_temperature, retrieval_distance,
                    )
                    outs.append(out)
                break  # success
            except RuntimeError as e:
                from .utils import is_cuda_oom, clear_cuda_cache
                if not is_cuda_oom(e) or bs <= 128:
                    raise
                clear_cuda_cache()
                new_bs = max(128, bs // 2)
                logger.warning("CUDA OOM during inference forward. Reducing batch size %d -> %d", bs, new_bs)
                bs = new_bs

        predictor = self._move_predictor_to_cpu(predictor)
        return torch.cat(outs, dim=0)


    # -----------------------------
    # Tree split helper
    # -----------------------------
    def _split_data_tree_embedding(
        self,
        X_original: np.ndarray | pd.DataFrame,
        y_original: np.ndarray | pd.Series,
        *,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stratify_by = check_stratification(y_original, self.stratify_sampling, self.task_type)

        if self.tree_data_split == 'half':
            X_tree, X, y_tree, y = train_test_split(
                X_original, y_original, test_size=0.5, random_state=random_state, stratify=stratify_by
            )
            logger.debug("Split strategy 'half': Half of the data for tree embedding, half for the main network.")

        elif self.tree_data_split == 'all':
            X_tree, y_tree = X_original, y_original
            X, y = X_original, y_original
            logger.debug("Split strategy 'all': Using all data for tree embedding and the main network.")

        elif self.tree_data_split == 'dynamic':
            n_samples = len(X_original)
            if n_samples < 2000:
                X_tree, y_tree = X_original, y_original
                X, y = X_original, y_original
                logger.debug("Dynamic split: <2k samples, using all data for both tree and embedding.")
            elif n_samples > 200000:
                X_tree, X, y_tree, y = train_test_split(
                    X_original, y_original,
                    test_size=(n_samples - 100000) / n_samples,
                    random_state=random_state,
                    stratify=stratify_by
                )
                logger.debug("Dynamic split: >200k samples, using 100k for tree and the rest for embedding.")
            else:
                X_tree, X, y_tree, y = train_test_split(
                    X_original, y_original, test_size=0.5, random_state=random_state, stratify=stratify_by
                )
                logger.debug("Dynamic split: 2k <= samples <= 200k, using 50-50 split.")
        else:
            raise ValueError(f"Unsupported tree_data_split strategy: {self.tree_data_split}")

        return X_tree, y_tree, X, y

    # -----------------------------
    # Shared fit body
    # -----------------------------
    def _fit_common(
        self,
        X_original: np.ndarray | pd.DataFrame,
        y_proc: np.ndarray | pd.Series,
        *,
        eval_set: Optional[tuple] = None,
        n_outputs: int,
        fit_deadline: float | None = None,
        fit_time_cushion_frac: float = 0.001,    # (0.1% headroom)
        return_partial_on_timeout: bool = True,
    ):
        # Reset state
        self.predictors_ = []
        self.preprocessors_ = []
        seed_everything(self.seed)
        self._auto_tune_for_memory()
        
        # Detect object columns with string content and add them to cat_features
        self.cat_features = detect_object_string_columns(X_original, self.cat_features)
        
        # helper to normalize 1D numpy targets
        def _as_numpy_1d(y, *, dtype=None) -> np.ndarray:
            arr = np.asarray(y)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            return arr

        # Derived flags
        if self.task_type == 'regression':
            self.clip_predictions_ = self.clip_predictions
            self.normalize_predictions_ = self.normalize_predictions
            if not self.finetuning:
                self.normalize_predictions_ = True
            elif self.normalize_predictions_ and self.finetuning:
                logger.warning("both normalize_predictions and finetuning are set to True.")
        else:
            self.clip_predictions_ = False
            self.normalize_predictions_ = False

        # Model required
        if self.checkpoint is None:
            raise ValueError("checkpoint must be provided.")
        self.model_path = self.checkpoint or self.model_path
        self._model = self._initialize_model()

        # TreeEmbedding creation
        self.tr_ = None
        if self.tree_embedding:
            if not self.tree_for_each_predictor:
                self.tr_ = TreeEmbedding(
                    tree_model=self.tree_model,
                    cat_features=self.cat_features,
                    seed=self.seed,
                    task_type=self.task_type,
                    n_estimators=self.tree_n_estimators,
                    lr=self.tree_lr,
                    max_depth=self.tree_max_depth,
                    min_samples_leaf=self.tree_min_samples_leaf,
                    subsample=self.tree_subsample,
                    feature_fraction=self.tree_feature_fraction,
                    device=str(self.device) if hasattr(self.device, '__str__') else self.device,
                    use_default_params=self.tree_use_default_params,
                    select_best_model=self.tree_select_best_model,
                    max_leaves=self.tree_max_leaves,
                    gamma=self.tree_gamma,
                    l2_leaf_reg=self.tree_l2_leaf_reg,
                    bagging_temperature=self.tree_bagging_temperature,
                    onehot_max_features=self.onehot_max_features
                )
            else:
                self.tr_ = [
                    TreeEmbedding(
                        tree_model=self.tree_model,
                        cat_features=self.cat_features,
                        seed=self.seed + i,
                        task_type=self.task_type,
                        n_estimators=self.tree_n_estimators,
                        lr=self.tree_lr,
                        max_depth=self.tree_max_depth,
                        min_samples_leaf=self.tree_min_samples_leaf,
                        subsample=self.tree_subsample,
                        feature_fraction=self.tree_feature_fraction,
                        device=str(self.device) if hasattr(self.device, '__str__') else self.device,
                        use_default_params=self.tree_use_default_params,
                        select_best_model=self.tree_select_best_model,
                        max_leaves=self.tree_max_leaves,
                        gamma=self.tree_gamma,
                        l2_leaf_reg=self.tree_l2_leaf_reg,
                        bagging_temperature=self.tree_bagging_temperature,
                        onehot_max_features=self.onehot_max_features
                    ) for i in range(self.n_ensemble)
                ]

        # Eval set unpack and normalize target once
        if eval_set is not None:
            X_val_original, y_val_proc = eval_set
            # choose dtype by task
            y_val_np = _as_numpy_1d(
                y_val_proc,
                dtype=(np.float32 if n_outputs == 1 else np.int64),
            )
            y_val_tensor_base = torch.from_numpy(y_val_np)
        else:
            X_val_original, y_val_proc = None, None
            y_val_tensor_base = None

        # Single tree path
        if self.tree_embedding and not self.tree_for_each_predictor:
            X_tree, y_tree, X_for_nn, y_for_nn = self._split_data_tree_embedding(
                X_original, y_proc, random_state=self.seed
            )
            self.tr_.fit_tree(X_tree, y_tree, eval_set=eval_set, concat_with_orig_features=self.concat_tree_with_orig_features)  # type: ignore[union-attr]

            X_emb_train = self.tr_.transform(X_for_nn)  # type: ignore[union-attr]
            X_emb_val = None
            if X_val_original is not None:
                X_emb_val = self.tr_.transform(X_val_original)  # type: ignore[union-attr]

            if self.concat_tree_with_orig_features:
                if self.tr_.n_orig_features_to_keep_ is not None:  # type: ignore[union-attr]
                    if isinstance(X_for_nn, pd.DataFrame):
                        X_for_nn = X_for_nn.iloc[:, :self.tr_.n_orig_features_to_keep_]  # type: ignore[union-attr]
                    else:
                        X_for_nn = X_for_nn[:, :self.tr_.n_orig_features_to_keep_]  # type: ignore[union-attr]
                X_work = np.concatenate([X_for_nn, X_emb_train], axis=1)
                if X_val_original is not None and X_emb_val is not None:
                    X_val_work = X_val_original
                    if self.tr_.n_orig_features_to_keep_ is not None:  # type: ignore[union-attr]
                        if isinstance(X_val_work, pd.DataFrame):
                            X_val_work = X_val_work.iloc[:, :self.tr_.n_orig_features_to_keep_]  # type: ignore[union-attr]
                        else:
                            X_val_work = X_val_work[:, :self.tr_.n_orig_features_to_keep_]  # type: ignore[union-attr]
                    X_val_work = np.concatenate([X_val_work, X_emb_val], axis=1)
                else:
                    X_val_work = None
            else:
                X_work = X_emb_train
                X_val_work = X_emb_val
            y_work = y_for_nn
        else:
            X_work, y_work = X_original, y_proc
            X_val_work = X_val_original if X_val_original is not None else None

        # Preprocess once if not per-predictor-tree
        if not (self.tree_embedding and self.tree_for_each_predictor):
            X_np, y_np, preproc = self._preprocess_fitting_data(X_work, y_work, is_classification=self.task_type == 'classification')
            self.preprocessors_.append(preproc)
            if X_val_work is not None:
                X_val_tensor = self._preprocess_test_data(X_val_work, preproc)
            else:
                X_val_tensor = None

            if isinstance(X_np, dict):
                y_tensor = torch.from_numpy(y_np)
                X_tensor = {'x_num': torch.from_numpy(X_np['x_num']), 'x_cat': torch.from_numpy(X_np['x_cat'])}
            else:
                X_tensor, y_tensor = torch.from_numpy(X_np), torch.from_numpy(y_np)

        predictor_times: list[float] = []

        # Generate predictors
        for i in range(self.n_ensemble):
            current_time = time.time()
            # Always gate the start of a new predictor on the remaining time budget
            if fit_deadline is not None:
                remaining = fit_deadline - current_time
                logger.debug(f"_fit_common: Before predictor {i+1}/{self.n_ensemble}, remaining={remaining:.2f}s, deadline={fit_deadline:.2f}, current={current_time:.2f}")
                if remaining <= 0:
                    logger.debug(f"_fit_common: Time budget exhausted (remaining={remaining:.2f}s <= 0)")
                    logger.warning(
                        "Early return: time budget exhausted before starting predictor %d. "
                        "Stopping at %d/%d predictors.",
                        i + 1, len(self.predictors_), self.n_ensemble
                    )
                    break
                if predictor_times:
                    avg_pred_time = sum(predictor_times) / len(predictor_times)
                    needed = avg_pred_time * (1.0 + fit_time_cushion_frac)
                    logger.debug(f"_fit_common: avg_pred_time={avg_pred_time:.2f}s, needed={needed:.2f}s, remaining={remaining:.2f}s")
                    if remaining <= max(1e-3, needed):
                        logger.debug(f"_fit_common: Time budget nearly exhausted (remaining={remaining:.2f}s <= needed={needed:.2f}s)")
                        logger.warning(
                            "Early return: time budget nearly exhausted (remaining=%.2fs < needed=%.2fs; "
                            "avg_pred=%.2fs, cushion=+%.0f%%). "
                            "Stopping at %d/%d predictors.",
                            remaining, needed, avg_pred_time, 100*fit_time_cushion_frac,
                            len(self.predictors_), self.n_ensemble
                        )
                        break

            stop_for_memory, memory_details = self._should_stop_for_cpu_memory_before_predictor(i + 1)
            if stop_for_memory and memory_details is not None:
                logger.warning(
                    "Early return: CPU memory budget nearly exhausted before predictor %d. "
                    "Stopping at %d/%d predictors. "
                    "usage=%.2fGB, available=%.2fGB, avg_predictor=%.2fGB, "
                    "projected_model=%.2fGB, projected_peak=%.2fGB, limit=%.2fGB, margin=%.2fGB.",
                    memory_details["next_predictor_index"],
                    memory_details["predictors_fit"],
                    memory_details["n_ensemble"],
                    memory_details["usage_gb"],
                    memory_details["available_gb"],
                    memory_details["avg_predictor_gb"],
                    memory_details["projected_model_gb"],
                    memory_details["projected_peak_gb"],
                    memory_details["limit_gb"],
                    memory_details["margin_gb"],
                )
                break

            logger.info(f"Generating predictor {i + 1} of {self.n_ensemble}...")
            t_pred_start = time.time()
            if self.tree_embedding and self.tree_for_each_predictor:
                # Per-predictor tree path
                X_tree, y_tree, X_for_nn, y_for_nn = self._split_data_tree_embedding(
                    X_original, y_proc, random_state=self.seed + i
                )
                self.tr_[i].fit_tree(X_tree, y_tree, eval_set=eval_set, concat_with_orig_features=self.concat_tree_with_orig_features)

                X_emb_tr = self.tr_[i].transform(X_for_nn)
                X_emb_val = self.tr_[i].transform(X_val_original) if X_val_original is not None else None

                if self.concat_tree_with_orig_features:
                    if self.tr_[i].n_orig_features_to_keep_ is not None:
                        if isinstance(X_for_nn, pd.DataFrame):
                            X_for_nn = X_for_nn.iloc[:, :self.tr_[i].n_orig_features_to_keep_]
                        else:
                            X_for_nn = X_for_nn[:, :self.tr_[i].n_orig_features_to_keep_]
                    X_fit = np.concatenate([X_for_nn, X_emb_tr], axis=1)
                    if X_val_original is not None and X_emb_val is not None:
                        X_val_work = X_val_original
                        if self.tr_[i].n_orig_features_to_keep_ is not None:
                            if isinstance(X_val_work, pd.DataFrame):
                                X_val_work = X_val_work.iloc[:, :self.tr_[i].n_orig_features_to_keep_]
                            else:
                                X_val_work = X_val_work[:, :self.tr_[i].n_orig_features_to_keep_]
                        X_val_fit = np.concatenate([X_val_work, X_emb_val], axis=1)
                    else:
                        X_val_fit = None
                else:
                    X_fit = X_emb_tr
                    X_val_fit = X_emb_val

                X_np, y_np, preproc = self._preprocess_fitting_data(X_fit, y_for_nn, is_classification=self.task_type == 'classification')
                self.preprocessors_.append(preproc)
                X_val_tensor = None
                if X_val_fit is not None:
                    X_val_tensor = self._preprocess_test_data(X_val_fit, preproc)

                if isinstance(X_np, dict):
                    y_tensor = torch.from_numpy(y_np)
                    X_tensor = {'x_num': torch.from_numpy(X_np['x_num']), 'x_cat': torch.from_numpy(X_np['x_cat'])}
                else:
                    X_tensor, y_tensor = torch.from_numpy(X_np), torch.from_numpy(y_np)

            # Logging shapes
            def _shape(x):
                if isinstance(x, dict):
                    return {k: tuple(v.shape) for k, v in x.items()}
                return tuple(x.shape) if hasattr(x, 'shape') else None

            # choose eval y tensor only if we actually built an X_val tensor for this predictor
            y_val_tensor = y_val_tensor_base if (eval_set is not None and X_val_tensor is not None) else None
            logger.debug(
                f"[iLTM.fit] _generate_predictor inputs: "
                f"X={_shape(X_tensor)}, y={_shape(y_tensor)}, "
                f"X_val={_shape(X_val_tensor)}, y_val={_shape(y_val_tensor)}"
            )
            
            pred = self._generate_predictor(
                X_tensor, y_tensor,
                n_outputs=n_outputs,
                X_val=X_val_tensor,
                y_val=y_val_tensor,
                fit_deadline=fit_deadline,
                fit_time_cushion_frac=fit_time_cushion_frac,
            )
            gc.collect()

            pred_duration = time.time() - t_pred_start
            predictor_times.append(pred_duration)
            remaining_after_pred = (fit_deadline - time.time()) if fit_deadline else None
            if remaining_after_pred is not None:
                logger.debug(f"_fit_common: Predictor {i+1} completed in {pred_duration:.2f}s, timed_out={pred['timed_out']}, remaining={remaining_after_pred:.2f}s")
            else:
                logger.debug(f"_fit_common: Predictor {i+1} completed in {pred_duration:.2f}s, timed_out={pred['timed_out']}")
            # Always keep the predictor if either it didn't time out, or we allow partial on timeout,
            # or it's the very first predictor (to avoid returning an empty model).
            if (not pred["timed_out"]) or return_partial_on_timeout or (pred["timed_out"] and len(self.predictors_) == 0):
                if pred["timed_out"] and len(self.predictors_) == 0 and not return_partial_on_timeout:
                    logger.warning("return_partial_on_timeout is False but the first predictor timed out. Returning this predictor anyway.")
                self.predictors_.append(pred)
                logger.debug(f"_fit_common: Added predictor {i+1} to ensemble (total={len(self.predictors_)})")
                # If this predictor timed out, don't start another one. Stop here and use the partial ensemble.
                if pred["timed_out"]:
                    logger.debug(f"_fit_common: Predictor {i+1} timed out, stopping ensemble generation")
                    logger.warning(
                        "Stopping ensemble generation after predictor %d due to time budget hit (partial ensemble used).",
                        i + 1,
                    )
                    break
            else:
                logger.debug(f"_fit_common: Predictor {i+1} timed out and return_partial_on_timeout=False")
                logger.warning(
                    "Early return: time budget hit during finetuning of the current ensemble member. "
                    f"Stopped at {len(self.predictors_)} / {self.n_ensemble} predictors."
                )
                break

        fit_end_time = time.time()
        fit_start_time_for_duration = getattr(self, '_fit_start_time', None)
        fit_duration = (fit_end_time - fit_start_time_for_duration) if fit_start_time_for_duration else None
        logger.debug(f"_fit_common END: {len(self.predictors_)}/{self.n_ensemble} predictors generated")
        if fit_deadline:
            remaining = fit_deadline - fit_end_time
            logger.debug(f"_fit_common END: remaining time={remaining:.2f}s, fit_duration={fit_duration:.2f}s" if fit_duration else f"_fit_common END: remaining time={remaining:.2f}s")
        logger.info(f"{len(self.predictors_)} predictors generated. Model fitted and ready for inference.")
        return self

    # -----------------------------
    # Shared predict helpers
    # -----------------------------
    def _preprocess_for_predict_once(self, X_original: np.ndarray | pd.DataFrame) -> Tensor:
        # Shared path when not per-predictor tree
        if self.tree_embedding and not self.tree_for_each_predictor:
            X_emb = self.tr_.transform(X_original)  # type: ignore[union-attr]
            if self.concat_tree_with_orig_features:
                if self.tr_.n_orig_features_to_keep_ is not None:  # type: ignore[union-attr]
                    if isinstance(X_original, pd.DataFrame):
                        X_original = X_original.iloc[:, :self.tr_.n_orig_features_to_keep_]  # type: ignore[union-attr]
                    else:
                        X_original = X_original[:, :self.tr_.n_orig_features_to_keep_]  # type: ignore[union-attr]
                X_work = np.concatenate([X_original, X_emb], axis=1)
            else:
                X_work = X_emb
        else:
            X_work = X_original

        return self._preprocess_test_data(X_work, self.preprocessors_[0])

    def _predict_ensemble(
        self,
        X_original: np.ndarray | pd.DataFrame,
        *,
        n_outputs: int,
        softmax_per_predictor: bool = False
    ) -> torch.Tensor:
        check_is_fitted(self)

        # If we can preprocess once, do it
        preprocessed_once: Optional[Tensor] = None
        if not (self.tree_embedding and self.tree_for_each_predictor):
            preprocessed_once = self._preprocess_for_predict_once(X_original)

        yhats: List[Tensor] = []

        for i, predictor in enumerate(self.predictors_):
            if self.tree_embedding and self.tree_for_each_predictor:
                # Chunked per predictor path
                n_samples = X_original.shape[0] if isinstance(X_original, np.ndarray) else len(X_original)
                try:
                    chunk_rows = int(self.inference_chunk_rows)
                except Exception:
                    chunk_rows = 10000

                outs_chunks: List[Tensor] = []
                for start in range(0, n_samples, chunk_rows):
                    end = min(start + chunk_rows, n_samples)
                    X_batch_orig = X_original.iloc[start:end] if isinstance(X_original, pd.DataFrame) else X_original[start:end]
                    X_emb = self.tr_[i].transform(X_batch_orig)
                    if self.concat_tree_with_orig_features:
                        X_batch_base = X_batch_orig
                        if self.tr_[i].n_orig_features_to_keep_ is not None:
                            if isinstance(X_batch_base, pd.DataFrame):
                                X_batch_base = X_batch_base.iloc[:, :self.tr_[i].n_orig_features_to_keep_]
                            else:
                                X_batch_base = X_batch_base[:, :self.tr_[i].n_orig_features_to_keep_]
                        X_batch = np.concatenate([X_batch_base, X_emb], axis=1)
                    else:
                        X_batch = X_emb

                    X_tensor = self._preprocess_test_data(X_batch, self.preprocessors_[i])
                    out = self._forward_pass_predictor(predictor, X_tensor, n_outputs=n_outputs)
                    if softmax_per_predictor:
                        out = F.softmax(out, dim=1)
                    outs_chunks.append(out)

                outputs = torch.cat(outs_chunks, dim=0)
            else:
                outputs = self._forward_pass_predictor(predictor, preprocessed_once, n_outputs=n_outputs)  # type: ignore[arg-type]
                if softmax_per_predictor:
                    outputs = F.softmax(outputs, dim=1)

            yhats.append(outputs)

        yhats_stacked = torch.stack(yhats)
        return torch.mean(yhats_stacked, dim=0)


# =====================================================================
#                           REGRESSOR
# =====================================================================
class iLTMRegressor(_iLTMBase, RegressorMixin, PermutationImportanceMixin):
    """
    Scikit-learn like regressor interface for iLTM.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        n_ensemble: int = 16,
        batch_size: int = 4096,
        seed: int = 3,
        finetuning: bool = True,
        finetuning_optimizer: str = 'adamw',
        finetuning_data: str = 'entire_dataset',
        finetuning_val_frac: float = 0.1,
        finetuning_batch_size: int | None = 256,
        finetuning_dropout: float = 0.15,
        finetuning_max_steps: int = 4096,
        finetuning_lr: float = 0.00021136819819225127,
        initial_transformations_finetuning: bool = True,
        early_stopping_mode: str = "auto",
        patience_epochs: int = 50,
        patience_checks: int | None = None,
        val_checks_per_epoch_target: int = 4,
        val_check_interval_batches: int | None = None,
        max_train_batches_per_epoch: int | None = 500,
        finetuning_subset_frac: float | None = None,
        finetuning_subset_max_samples: int | None = 500_000,
        val_max_samples: int | None = 100_000,
        min_epochs: int = 0,
        cooldown_checks: int = 2,
        gradient_clip_norm: float = 0.9968664343329384,
        scheduler_min_lr: float = 0.00025787005984493423,
        checkpoint: str | None = "xgbrconcat",
        stratify_sampling: bool = False,
        feature_bagging: bool = False,
        feature_bagging_size: int = 3000,
        feature_bagging_type: str = 'uniform',
        cat_features: List[int] | None = None,
        task_type: str = 'regression',  # overridden by subclasses
        preprocessing: str = 'realmlp_td_s_v0',
        dim_exp_type: str = 'rf',
        rf_size: int = 32768,
        n_dims: int = 512,
        pca_fit: str = 'reduced',
        pca_sampling: str = 'zeropad',
        pca_svd_driver: str | None = None,
        clip_data_value: float = 1000000,
        n_classes_limit: int = 100,
        hn_n_layers: int = 4,
        hn_hidden_size: int = 1024,
        hyper_dropout: float = 0.0,
        main_n_layers: int = 3,
        bottleneck_size: int = 0,
        tree_embedding: bool = False,
        tree_model: str = 'XGBoost_hist',
        tree_n_estimators: int = 200,
        tree_lr: float | None = 0.05848722909622601,
        tree_max_depth: int | None = 6,
        tree_min_samples_leaf: int | None = 90,
        tree_subsample: float | None = 0.5074398421774727,
        tree_feature_fraction: float | None = 0.8132119890393672,
        tree_data_split: str = 'dynamic',
        tree_for_each_predictor: bool = True,
        tree_use_default_params: bool = False,
        tree_select_best_model: bool = True,
        concat_tree_with_orig_features: bool = False,
        tree_max_leaves: int | None = None,
        tree_gamma: float | None = 1.4598703125721042,
        tree_l2_leaf_reg: float | None = 1.007789530064673,
        tree_bagging_temperature: float | None = 0.12093655368094158,
        onehot_max_features: bool = True,
        do_retrieval: bool = False,
        retrieval_alpha: float = 0.7024108748899226,
        retrieval_temperature: float = 1.921535071554998,
        retrieval_distance: str = 'euclidean',
        retrieval_alpha_finetuning: bool = False,
        retrieval_temperature_finetuning: bool = False,
        logging_level: int | str = logging.WARNING,
        clip_predictions: bool = True,
        normalize_predictions: bool = False,
        use_amp_inference: bool = False,
        use_amp_finetuning: bool = False,
        adaptive_memory: bool = True,
        gpu_mem_low_mb: int = 3072,
        gpu_mem_very_low_mb: int = 1024,
        min_finetuning_batch_size: int = 256,
        gpu_oom_retries: int = 2,
        auto_disable_retrieval_on_low_mem: bool = False,
        auto_amp_on_low_mem: bool = False,
        auto_stop_on_low_cpu_memory: bool = True,
        cpu_memory_limit_gb: float | None = None,
        cpu_memory_safety_margin_gb: float = 2.0,
        corr_select_k: int = 300,
        inference_storage_dtype: str | None = "float16",
    ) -> None:
        params = locals().copy()
        params.pop("self")
        params.pop("__class__")
        super().__init__(**params)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: tuple | None = None,
        *,
        fit_max_time: float | None = None,
        fit_time_cushion_frac: float = 0.001,    # (0.1% headroom)
        fit_time_margin_frac: float = 0.05,      # (5% reserved for wrapper overhead)
        fit_time_margin_min_seconds: float = 10.0,
        return_partial_on_timeout: bool = True,
    ) -> iLTMRegressor:
        """
        Fit the iLTM regressor model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix for training.
        y : np.ndarray or pd.Series
            Target values for training.
        eval_set : tuple or None
            Optional validation set as a tuple (X_val, y_val).
        fit_max_time : float or None
            Maximum time in seconds to allow for fitting. If None, no time limit is applied.
        fit_time_cushion_frac : float
            Fractional time cushion to add when checking for time budget during ensemble generation.
            Used to estimate if there's enough time for the next predictor.
        fit_time_margin_frac : float
            Fractional time margin to reserve from fit_max_time for post-fit operations
            (e.g., prediction, scoring). The effective training time becomes
            fit_max_time * (1 - fit_time_margin_frac).
        fit_time_margin_min_seconds : float
            Minimum time to reserve from fit_max_time for post-fit operations
            (e.g., prediction, scoring).
        return_partial_on_timeout : bool
            If True, if the time limit triggers during creation of an ensemble member,
            include that member using the best weights at the moment of timeout.
        """
        fit_start_time = time.time()
        if fit_max_time:
            reserved_time = max(float(fit_max_time) * fit_time_margin_frac, float(fit_time_margin_min_seconds))
            effective_fit_time = max(0.0, float(fit_max_time) - reserved_time)
            fit_deadline = fit_start_time + effective_fit_time
            logger.debug(f"iLTMRegressor.fit START: fit_max_time={fit_max_time:.2f}s, reserved_margin={reserved_time:.2f}s (frac={fit_time_margin_frac:.3f}, min={fit_time_margin_min_seconds:.2f}), effective_time={effective_fit_time:.2f}s")
        else:
            fit_deadline = None
            logger.debug(f"iLTMRegressor.fit START: fit_max_time=None (no time limit)")
        if fit_deadline:
            logger.debug(f"iLTMRegressor.fit: Effective deadline in {fit_deadline - fit_start_time:.2f}s")
        self._fit_start_time = fit_start_time  # Store for use in _fit_common
        
        # Normalize y for the hypernetwork
        y_arr = np.array(y, dtype=np.float32)
        self._y_mean = float(np.mean(y_arr))
        self._y_std  = float(np.std(y_arr)) + 1e-6
        y_proc = ((y_arr - self._y_mean) / self._y_std).astype(np.float32)

        if eval_set is not None:
            X_val_original, y_val_raw = eval_set
            y_val_proc = ((np.array(y_val_raw, dtype=np.float32) - self._y_mean) / self._y_std).astype(np.float32)
            eval_set_proc = (X_val_original, y_val_proc)
        else:
            eval_set_proc = None

        # Record training target range in normalized space for clipping
        self._train_min = float(np.min(y_proc))
        self._train_max = float(np.max(y_proc))
        if self.clip_predictions:
            logger.debug(f"Training target range: min={self._train_min:.4f}, max={self._train_max:.4f}")

        # Regression has one output
        self.n_outputs_ = 1
        
        result = self._fit_common(
            X, y_proc,
            eval_set=eval_set_proc,
            n_outputs=self.n_outputs_,
            fit_deadline=fit_deadline,
            fit_time_cushion_frac=fit_time_cushion_frac,
            return_partial_on_timeout=return_partial_on_timeout,
        )
        fit_total_end = time.time()
        fit_total_duration = fit_total_end - fit_start_time
        logger.debug(f"iLTMRegressor.fit END: Total fit duration={fit_total_duration:.2f}s")
        return result

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        yhats = self._predict_ensemble(
            X, n_outputs=self.n_outputs_, softmax_per_predictor=False
        )

        if self.normalize_predictions_:
            yhats = (yhats - yhats.mean()) / (yhats.std() + 1e-6)

        if self.clip_predictions_:
            pred_min, pred_max = float(yhats.min()), float(yhats.max())
            logger.debug(f"Predictions range before clipping: min={pred_min:.4f}, max={pred_max:.4f}")
            yhats = torch.clamp(yhats, min=self._train_min, max=self._train_max)
            new_min, new_max = float(yhats.min()), float(yhats.max())
            logger.debug(f"Predictions range after clipping: min={new_min:.4f}, max={new_max:.4f}")

        preds = yhats.cpu().numpy() * self._y_std + self._y_mean
        return preds


# =====================================================================
#                           CLASSIFIER
# =====================================================================
class iLTMClassifier(_iLTMBase, ClassifierMixin, PermutationImportanceMixin):
    """
    Scikit-learn like classifier interface for iLTM.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        n_ensemble: int = 16,
        batch_size: int = 4096,
        seed: int = 3,
        finetuning: bool = True,
        finetuning_optimizer: str = 'adamw',
        finetuning_data: str = 'entire_dataset',
        finetuning_val_frac: float = 0.1,
        finetuning_batch_size: int | None = 256,
        finetuning_dropout: float = 0.15,
        finetuning_max_steps: int = 4096,
        finetuning_lr: float = 0.00021136819819225127,
        finetuning_classification_val_metric: str = 'auto',  # options: 'auto', 'logloss', 'auc' ('auto': 'auc' binary, 'logloss' multiclass)
        initial_transformations_finetuning: bool = True,
        early_stopping_mode: str = "auto",
        patience_epochs: int = 50,
        patience_checks: int | None = None,
        val_checks_per_epoch_target: int = 4,
        val_check_interval_batches: int | None = None,
        max_train_batches_per_epoch: int | None = 500,
        finetuning_subset_frac: float | None = None,
        finetuning_subset_max_samples: int | None = 500_000,
        val_max_samples: int | None = 100_000,
        min_epochs: int = 0,
        cooldown_checks: int = 2,
        gradient_clip_norm: float = 0.9968664343329384,
        scheduler_min_lr: float = 0.00025787005984493423,
        checkpoint: str | None = "xgbrconcat",
        stratify_sampling: bool = False,
        feature_bagging: bool = False,
        feature_bagging_size: int = 3000,
        feature_bagging_type: str = 'uniform',
        cat_features: List[int] | None = None,
        task_type: str = 'classification',  # overridden by subclasses
        preprocessing: str = 'realmlp_td_s_v0',
        dim_exp_type: str = 'rf',
        rf_size: int = 32768,
        n_dims: int = 512,
        pca_fit: str = 'reduced',
        pca_sampling: str = 'zeropad',
        pca_svd_driver: str | None = None,
        clip_data_value: float = 1000000,
        n_classes_limit: int = 100,
        hn_n_layers: int = 4,
        hn_hidden_size: int = 1024,
        hyper_dropout: float = 0.0,
        main_n_layers: int = 3,
        bottleneck_size: int = 0,
        tree_embedding: bool = False,
        tree_model: str = 'XGBoost_hist',
        tree_n_estimators: int = 200,
        tree_lr: float | None = 0.05848722909622601,
        tree_max_depth: int | None = 6,
        tree_min_samples_leaf: int | None = 90,
        tree_subsample: float | None = 0.5074398421774727,
        tree_feature_fraction: float | None = 0.8132119890393672,
        tree_data_split: str = 'dynamic',
        tree_for_each_predictor: bool = True,
        tree_use_default_params: bool = False,
        tree_select_best_model: bool = True,
        concat_tree_with_orig_features: bool = False,
        tree_max_leaves: int | None = None,
        tree_gamma: float | None = 1.4598703125721042,
        tree_l2_leaf_reg: float | None = 1.007789530064673,
        tree_bagging_temperature: float | None = 0.12093655368094158,
        onehot_max_features: bool = True,
        do_retrieval: bool = False,
        retrieval_alpha: float = 0.7024108748899226,
        retrieval_temperature: float = 1.921535071554998,
        retrieval_distance: str = 'euclidean',
        retrieval_alpha_finetuning: bool = False,
        retrieval_temperature_finetuning: bool = False,
        logging_level: int | str = logging.WARNING,
        clip_predictions: bool = True,          # ignored for classification
        normalize_predictions: bool = False,     # ignored for classification
        use_amp_inference: bool = False,
        use_amp_finetuning: bool = False,
        voting: str = 'soft',
        adaptive_memory: bool = True,
        gpu_mem_low_mb: int = 2048,
        gpu_mem_very_low_mb: int = 1024,
        min_finetuning_batch_size: int = 64,
        gpu_oom_retries: int = 2,
        auto_disable_retrieval_on_low_mem: bool = False,
        auto_amp_on_low_mem: bool = False,
        auto_stop_on_low_cpu_memory: bool = True,
        cpu_memory_limit_gb: float | None = None,
        cpu_memory_safety_margin_gb: float = 2.0,
        corr_select_k: int = 300,
        inference_storage_dtype: str | None = "float16",
    ) -> None:
        params = locals().copy()
        params.pop("self")
        params.pop("__class__")
        params.pop("voting")
        super().__init__(**params)
        self.voting = voting

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: tuple | None = None,
        *,
        fit_max_time: float | None = None,
        fit_time_cushion_frac: float = 0.001,  # (0.1% headroom)
        fit_time_margin_frac: float = 0.05,    # (5% reserved for wrapper overhead)
        fit_time_margin_min_seconds: float = 10.0,
        return_partial_on_timeout: bool = True,
    ) -> iLTMClassifier:
        """
        Fit the iLTM classifier model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix for training.
        y : np.ndarray or pd.Series
            Target values for training.
        eval_set : tuple or None
            Optional validation set as a tuple (X_val, y_val).
        fit_max_time : float or None
            Maximum time in seconds to allow for fitting. If None, no time limit is applied.
        fit_time_cushion_frac : float
            Fractional time cushion to add when checking for time budget during ensemble generation.
            Used to estimate if there's enough time for the next predictor.
        fit_time_margin_frac : float
            Fractional time margin to reserve from fit_max_time for post-fit operations
            (e.g., prediction, scoring). The effective training time becomes
            fit_max_time * (1 - fit_time_margin_frac).
        fit_time_margin_min_seconds : float
            Minimum time to reserve from fit_max_time for post-fit operations
            (e.g., prediction, scoring).
        return_partial_on_timeout : bool
            If True, if the time limit triggers during creation of an ensemble member,
            include that member using the best weights at the moment of timeout.
        """
        fit_start_time = time.time()
        if fit_max_time:
            reserved_time = max(float(fit_max_time) * fit_time_margin_frac, float(fit_time_margin_min_seconds))
            effective_fit_time = max(0.0, float(fit_max_time) - reserved_time)
            fit_deadline = fit_start_time + effective_fit_time
            logger.debug(f"iLTMClassifier.fit START: fit_max_time={fit_max_time:.2f}s, reserved_margin={reserved_time:.2f}s (frac={fit_time_margin_frac:.3f}, min={fit_time_margin_min_seconds:.2f}), effective_time={effective_fit_time:.2f}s")
        else:
            fit_deadline = None
            logger.debug(f"iLTMClassifier.fit START: fit_max_time=None (no time limit)")
        if fit_deadline:
            logger.debug(f"iLTMClassifier.fit: Effective deadline in {fit_deadline - fit_start_time:.2f}s")
        self._fit_start_time = fit_start_time  # Store for use in _fit_common
        # Encode labels
        self.classes_, y_proc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_

        # Sanity check and possibly align eval_set labels
        if eval_set is not None:
            eval_X, eval_y = eval_set
            classes_in_eval, eval_y_proc = np.unique(eval_y, return_inverse=True)

            # Check if some classes in the evaluation set are not present in the original dataset
            if not set(classes_in_eval).issubset(set(self.classes_)):
                logger.debug(f"Classes in original dataset: {self.classes_}, classes in evaluation set: {classes_in_eval}")
                logger.debug("Removing samples with unseen classes from the evaluation set.")
                mask = np.isin(eval_y, self.classes_)
                eval_X, eval_y = eval_X[mask], eval_y[mask]

            eval_set = eval_X, eval_y_proc

        result = self._fit_common(
            X, y_proc,
            eval_set=eval_set,
            n_outputs=self.n_outputs_,
            fit_deadline=fit_deadline,
            fit_time_cushion_frac=fit_time_cushion_frac,
            return_partial_on_timeout=return_partial_on_timeout,
        )
        fit_total_end = time.time()
        fit_total_duration = fit_total_end - fit_start_time
        logger.debug(f"iLTMClassifier.fit END: Total fit duration={fit_total_duration:.2f}s")
        return result

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        probs = self._predict_ensemble(
            X, n_outputs=self.n_outputs_, softmax_per_predictor=True
        )
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.voting == 'soft':
            outputs = self.predict_proba(X)
            return self.classes_[np.argmax(outputs, axis=1)]
        elif self.voting == 'hard':
            check_is_fitted(self)

            # Preprocess once if possible
            preprocessed_once: Optional[Tensor] = None
            if not (self.tree_embedding and self.tree_for_each_predictor):
                preprocessed_once = self._preprocess_for_predict_once(X)

            votes: List[Tensor] = []
            for i, predictor in enumerate(self.predictors_):
                if self.tree_embedding and self.tree_for_each_predictor:
                    n_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
                    try:
                        chunk_rows = int(self.inference_chunk_rows)
                    except Exception:
                        chunk_rows = 10000
                    pred_labels_chunks: List[Tensor] = []
                    for start in range(0, n_samples, chunk_rows):
                        end = min(start + chunk_rows, n_samples)
                        X_batch_orig = X.iloc[start:end] if isinstance(X, pd.DataFrame) else X[start:end]
                        X_emb = self.tr_[i].transform(X_batch_orig)
                        if self.concat_tree_with_orig_features:
                            X_base = X_batch_orig
                            if self.tr_[i].n_orig_features_to_keep_ is not None:
                                if isinstance(X_base, pd.DataFrame):
                                    X_base = X_base.iloc[:, :self.tr_[i].n_orig_features_to_keep_]
                                else:
                                    X_base = X_base[:, :self.tr_[i].n_orig_features_to_keep_]
                            X_batch = np.concatenate([X_base, X_emb], axis=1)
                        else:
                            X_batch = X_emb
                        X_tensor = self._preprocess_test_data(X_batch, self.preprocessors_[i])
                        logits = self._forward_pass_predictor(predictor, X_tensor, n_outputs=self.n_outputs_)
                        pred_labels_chunks.append(torch.argmax(logits, dim=1))
                    pred_labels = torch.cat(pred_labels_chunks, dim=0)
                else:
                    logits = self._forward_pass_predictor(predictor, preprocessed_once, n_outputs=self.n_outputs_)  # type: ignore[arg-type]
                    pred_labels = torch.argmax(logits, dim=1)
                votes.append(pred_labels.unsqueeze(0))

            votes_tensor = torch.cat(votes, dim=0)
            majority, _ = votes_tensor.mode(dim=0)
            return self.classes_[majority.cpu().numpy()]
        else:
            raise ValueError("Invalid voting method specified. Choose 'soft' or 'hard'.")
