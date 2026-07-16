"""
scikit-learn compatible TabFM classifier wrapper for TabTune.

This is **not** an API-call wrapper: TabTune vendors the entire TabFM PyTorch
stack (architecture + HF loader + preprocessing/ensembling engine) under
``tabtune/models/tabfm/``.  This class gives that vendored engine the uniform
TabTune contract so the ``TabularPipeline`` / ``TuningManager`` / leaderboard /
ensemble / distillation machinery treats it like every other model:

* ``fit`` / ``predict`` / ``predict_proba`` -- real in-context inference,
  delegated to the vendored ``TabFMClassifier`` (full preprocessing, 32-member
  ensembling and calibration);
* ``model_`` -- the real ``torch.nn.Module`` (vendored ``TabFM``) used by the
  ``TuningManager`` for episodic fine-tuning, LoRA/PEFT injection and checkpoints;
* ``y_encoder_`` / ``classes_`` / ``n_classes_`` -- label bookkeeping for the
  pipeline's evaluation and probability-alignment code;
* ``icl_logits`` -- the model's **real** support/query forward for fine-tuning;
* ``prepare_episode_features`` -- vendored preprocessing for FT episodes.

All heavy imports (torch + the vendored engine) happen lazily inside
``_load_model`` so ``import tabtune`` works without the optional TabFM stack.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from . import backbone as _bk

logger = logging.getLogger(__name__)

# Constructor arguments understood by the vendored TabFMClassifier -- everything
# else in model_params is ignored (with a debug log) so unknown pipeline keys
# never crash construction.
_VENDORED_CLF_KWARGS = {
    "n_estimators", "norm_methods", "feat_shuffle_method", "class_shift",
    "permute_categorical", "outlier_threshold", "max_num_features", "max_num_rows",
    "softmax_temperature", "average_logits", "use_amp", "batch_size", "random_state",
    "verbose", "cat_encoder_mode", "binary_calibration_method",
    "multiclass_calibration_method", "num_folds_for_cv", "n_feature_crosses",
    "n_svd_features", "total_svd_pool", "enable_nnls", "nnls_beta",
    "calibration_lambda", "min_rows_for_single_val_split",
}


class TabFMClassifier(ClassifierMixin, BaseEstimator):
    """TabFM (Google) tabular in-context classifier with the TabTune contract."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype=None,
        checkpoint_path: Optional[str] = None,
        n_estimators: Optional[int] = None,
        softmax_temperature: Optional[float] = None,
        ignore_pretraining_limits: bool = True,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state
        self._extra_kwargs = dict(kwargs)

        self.backbone_ = None
        self.estimator_ = None
        self.model_ = None
        self.y_encoder_: Optional[LabelEncoder] = None
        self.classes_ = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self._is_fitted = False

    # -- sklearn plumbing -----------------------------------------------------
    def _more_tags(self):
        return {"non_deterministic": True, "allow_nan": True}

    def _vendored_kwargs(self):
        kw = {}
        if self.n_estimators is not None:
            kw["n_estimators"] = self.n_estimators
        if self.softmax_temperature is not None:
            kw["softmax_temperature"] = self.softmax_temperature
        if self.random_state is not None:
            kw["random_state"] = self.random_state
        for k, v in self._extra_kwargs.items():
            if k in _VENDORED_CLF_KWARGS:
                kw[k] = v
            else:
                logger.debug("[TabFM] ignoring unknown model_param %r", k)
        return kw

    def _load_model(self):
        """Load the pretrained backbone + build the vendored estimator (idempotent)."""
        if self.estimator_ is not None and self.model_ is not None:
            return
        self.backbone_ = _bk.load_backbone(
            model_type="classification", device=self.device,
            dtype=self.dtype, checkpoint_path=self.checkpoint_path,
        )
        self.model_ = self.backbone_  # the real torch.nn.Module (TabFM)
        self.estimator_ = _bk.build_estimator("classification", self.backbone_, **self._vendored_kwargs())

    # -- core API -------------------------------------------------------------
    def fit(self, X, y):
        self._load_model()
        self.y_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        logger.info("[TabFM] Fitting in-context context (%s classes)", self.n_classes_)
        self.estimator_.fit(_as_frame(X), _as_1d(y))
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("TabFMClassifier must be fitted before predict_proba().")
        proba = np.asarray(self.estimator_.predict_proba(_as_frame(X)), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("TabFMClassifier must be fitted before predict().")
        return np.asarray(self.estimator_.predict(_as_frame(X)))

    # -- fine-tuning helpers --------------------------------------------------
    @property
    def max_classes(self) -> int:
        return int(getattr(self.model_, "max_classes", self.n_classes_ or 2))

    def prepare_episode_features(self, X_raw):
        """Vendored-normalised numeric features for FT episodes: (X[N,H], cat_mask)."""
        return _bk.split_vendored_preprocessing(self.estimator_, _as_frame(X_raw))

    def icl_logits(self, x, y, train_size, cat_mask=None, d=None):
        """Real differentiable support/query forward (see backbone.real_icl_logits)."""
        if self.model_ is None:
            raise RuntimeError("TabFM backbone not loaded; call _load_model() first.")
        return _bk.real_icl_logits(self.model_, x, y, train_size, cat_mask=cat_mask, d=d)

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


# -- small input coercers -----------------------------------------------------
def _as_frame(X):
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


def _as_1d(y):
    arr = np.asarray(y)
    return arr.ravel() if arr.ndim > 1 else arr
