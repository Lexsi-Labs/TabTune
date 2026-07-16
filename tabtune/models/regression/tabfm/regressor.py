"""
TabFM regression wrapper for the TabTune pipeline.

Drives the **vendored** TabFM regression engine (real architecture + HF loader +
preprocessing/ensembling under ``tabtune/models/tabfm/``) behind the uniform
TabTune contract.  Inference is delegated to the vendored ``TabFMRegressor``;
the underlying ``torch.nn.Module`` is exposed as ``self.model_`` so the
``TuningManager`` can run episodic turn-by-turn fine-tuning, LoRA/PEFT and
checkpointing on the real modules.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ...tabfm import backbone as _bk

logger = logging.getLogger(__name__)

_VENDORED_REG_KWARGS = {
    "n_estimators", "norm_methods", "feat_shuffle_method", "permute_categorical",
    "outlier_threshold", "max_num_features", "max_num_rows", "use_amp", "batch_size",
    "random_state", "verbose", "cat_encoder_mode", "num_folds_for_cv",
    "n_feature_crosses", "n_svd_features", "total_svd_pool", "enable_nnls", "nnls_beta",
}


class TabFMRegressorWrapper(BaseEstimator, RegressorMixin):
    """TabFM tabular in-context regressor with the TabTune contract."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype=None,
        checkpoint_path: Optional[str] = None,
        tuning_strategy: str = "inference",
        ignore_pretraining_limits: bool = True,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError("TabFM regression supports tuning_strategy in {'inference','finetune'}.")
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.tuning_strategy = tuning_strategy
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k != "task_type"}

        self.backbone_ = None
        self.estimator_ = None
        self.model_ = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self._is_fitted = False

    def _vendored_kwargs(self):
        kw = {}
        if self.random_state is not None:
            kw["random_state"] = self.random_state
        for k, v in self._extra_kwargs.items():
            if k in _VENDORED_REG_KWARGS:
                kw[k] = v
            else:
                logger.debug("[TabFM] ignoring unknown regression model_param %r", k)
        return kw

    def _initialize_model_variables(self):
        self._load_model()

    def _load_model(self):
        if self.estimator_ is not None and self.model_ is not None:
            return
        self.backbone_ = _bk.load_backbone(
            model_type="regression", device=self.device,
            dtype=self.dtype, checkpoint_path=self.checkpoint_path,
        )
        self.model_ = self.backbone_
        self.estimator_ = _bk.build_estimator("regression", self.backbone_, **self._vendored_kwargs())

    def fit(self, X, y):
        self._load_model()
        y_arr = np.asarray(y, dtype=float).ravel()
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr) + 1e-8)
        logger.info("[TabFM] Fitting regression context (n=%d)", len(y_arr))
        self.estimator_.fit(_as_frame(X), y_arr)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("TabFMRegressorWrapper must be fitted before predict().")
        return np.asarray(self.estimator_.predict(_as_frame(X)), dtype=float).ravel()

    # -- fine-tuning helpers --------------------------------------------------
    def prepare_episode_features(self, X_raw):
        return _bk.split_vendored_preprocessing(self.estimator_, _as_frame(X_raw))

    def icl_predict(self, x, y, train_size, cat_mask=None, d=None):
        """Real differentiable support/query forward; returns query scalars."""
        if self.model_ is None:
            raise RuntimeError("TabFM regression backbone not loaded.")
        out = _bk.real_icl_logits(self.model_, x, y, train_size, cat_mask=cat_mask, d=d)
        if out.dim() == 3 and out.size(-1) == 1:
            out = out.squeeze(-1)  # [B, T]
        return out

    def to(self, device):
        if self.model_ is not None:
            self.model_.to(device)
        return self

    def eval(self):
        if self.model_ is not None:
            self.model_.eval()
        return self

    def train(self, mode: bool = True):
        if self.model_ is not None:
            self.model_.train(mode)
        return self

    def parameters(self):
        return iter(()) if self.model_ is None else self.model_.parameters()


def _as_frame(X):
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))
