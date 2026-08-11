"""
scikit-learn compatible xRFM classifier wrapper for TabTune.

This is **not** a pip-dependency wrapper: TabTune vendors the entire xRFM
engine (Recursive Feature Machines + the tree-splittable xRFM estimator,
MIT licensed, see ``tabtune/models/xrfm/LICENSE``) under
``tabtune/models/xrfm/``.  This class gives that vendored engine the uniform
TabTune contract so the ``TabularPipeline`` / ``TuningManager`` / leaderboard
machinery treats it like every other model:

* ``fit`` / ``predict`` / ``predict_proba`` -- real xRFM training + inference
  on raw mixed-type frames (internal one-hot/standardise encoding per the
  upstream README);
* ``model_`` -- the fitted vendored ``xRFM`` estimator (a kernel machine, NOT a
  ``torch.nn.Module``; checkpoints are saved via joblib by the TuningManager);
* ``y_encoder_`` / ``classes_`` / ``n_classes_`` -- label bookkeeping for the
  pipeline's evaluation and probability-alignment code;
* ``leaf_models`` / ``transform_features`` / ``numeric_targets`` -- hooks the
  ``TuningManager`` uses for RFM refinement ('finetune') and low-rank
  M-matrix adaptation ('peft').

Note on tuning strategies: xRFM has **no pretrained checkpoint and no gradient
-trained weights**.  Its learned state is the per-leaf Mahalanobis matrix ``M``
(from AGOP iterations) plus kernel ridge weights.  TabTune therefore maps
``tuning_strategy='finetune'`` to full RFM (re)training with user-controlled
hyperparameters, and ``'peft'`` to a frozen-base low-rank update of ``M``
(see ``TuningManager._finetune_xrfm`` / ``_peft_xrfm``).

All heavy imports (torch + the vendored engine) happen lazily inside
``_load_model`` so ``import tabtune`` works without the optional stack.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .preprocessing import XRFMFeatureEncoder

logger = logging.getLogger(__name__)

# Constructor arguments understood by the vendored xRFM estimator -- everything
# else in model_params is ignored (with a debug log) so unknown pipeline keys
# never crash construction.
_ENGINE_KWARGS = {
    "max_leaf_size", "min_subset_size", "number_of_splits", "n_trees",
    "n_tree_iters", "split_method", "categorical_info", "classification_mode",
    "time_limit_s", "n_threads", "refill_size", "split_temperature",
    "overlap_fraction", "use_temperature_tuning", "keep_weight_frac_in_predict",
    "max_leaf_count_in_ensemble", "temp_tuning_space",
}

# Keys of the per-leaf RFM 'model' config exposed as first-class wrapper args.
_RFM_MODEL_KEYS = ("kernel", "bandwidth", "exponent", "diag", "bandwidth_mode")
_RFM_FIT_KEYS = ("reg", "iters")


class XRFMClassifier(ClassifierMixin, BaseEstimator):
    """xRFM (Recursive Feature Machine) tabular classifier with the TabTune contract."""

    def __init__(
        self,
        device: Optional[str] = None,
        tuning_strategy: str = "inference",
        rfm_params: Optional[dict] = None,
        kernel: str = "l2",
        bandwidth: float = 10.0,
        exponent: float = 1.0,
        diag: bool = False,
        bandwidth_mode: str = "constant",
        reg: float = 1e-3,
        iters: int = 4,
        n_trees: int = 1,
        tuning_metric: str = "brier",
        categorical_encoding: str = "onehot",
        val_size: float = 0.2,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        **kwargs,
    ):
        if tuning_strategy not in ("inference", "finetune", "peft"):
            raise ValueError(
                "XRFM classification supports tuning_strategy in {'inference','finetune','peft'}."
            )
        self.device = device
        self.tuning_strategy = tuning_strategy
        self.rfm_params = rfm_params
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.exponent = exponent
        self.diag = diag
        self.bandwidth_mode = bandwidth_mode
        self.reg = reg
        self.iters = iters
        self.n_trees = n_trees
        self.tuning_metric = tuning_metric
        self.categorical_encoding = categorical_encoding
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in ("task_type", "tuning_strategy")}

        self.model_ = None  # the fitted vendored xRFM estimator (kernel machine)
        self.feature_encoder_: Optional[XRFMFeatureEncoder] = None
        self.y_encoder_: Optional[LabelEncoder] = None
        self.classes_ = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self._is_fitted = False

    # -- sklearn plumbing -----------------------------------------------------
    def _more_tags(self):
        return {"non_deterministic": True, "allow_nan": True}

    def _resolved_device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _engine_kwargs(self):
        kw = {}
        for k, v in self._extra_kwargs.items():
            if k in _ENGINE_KWARGS:
                kw[k] = v
            else:
                logger.debug("[XRFM] ignoring unknown model_param %r", k)
        return kw

    def _resolved_rfm_params(self):
        """Assemble the nested rfm_params dict the vendored engine expects."""
        if self.rfm_params is not None:
            return self.rfm_params
        return {
            "model": {
                "kernel": self.kernel,
                "bandwidth": self.bandwidth,
                "exponent": self.exponent,
                "diag": self.diag,
                "bandwidth_mode": self.bandwidth_mode,
            },
            "fit": {
                "reg": self.reg,
                "iters": self.iters,
                "verbose": self.verbose,
                "early_stop_rfm": True,
                "return_best_params": True,
            },
        }

    def _build_engine(self):
        """Construct a fresh vendored xRFM estimator from the current hyperparams."""
        from .xrfm import xRFM  # lazy: pulls torch + the full engine

        return xRFM(
            rfm_params=self._resolved_rfm_params(),
            device=self._resolved_device(),
            n_trees=self.n_trees,
            tuning_metric=self.tuning_metric,
            random_state=self.random_state,
            verbose=self.verbose,
            **self._engine_kwargs(),
        )

    def _load_model(self):
        """Build the engine if absent (idempotent; no weights to download)."""
        if self.model_ is None:
            self.model_ = self._build_engine()

    def _make_val_split(self, X_num, y_enc):
        """Internal train/val split for xRFM's tuning-metric model selection.

        Falls back to validating on the training rows for tiny datasets where a
        stratified holdout is not possible (keeps every class in the context).
        """
        n = X_num.shape[0]
        n_val = int(round(self.val_size * n))
        min_count = int(np.bincount(y_enc).min()) if len(y_enc) else 0
        if n < 8 or n_val < self.n_classes_ or min_count < 2:
            logger.debug("[XRFM] Dataset too small for a holdout; validating on the training split.")
            return X_num, y_enc, X_num.copy(), y_enc.copy()
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_num, y_enc, test_size=self.val_size,
                random_state=self.random_state, stratify=y_enc,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_num, y_enc, test_size=self.val_size, random_state=self.random_state,
            )
        # Guard: xRFM infers n_classes from train+val, so every class must appear.
        if len(np.unique(y_tr)) < self.n_classes_:
            return X_num, y_enc, X_num.copy(), y_enc.copy()
        return X_tr, y_tr, X_val, y_val

    # -- core API -------------------------------------------------------------
    def fit(self, X, y):
        self.feature_encoder_ = XRFMFeatureEncoder(categorical_encoding=self.categorical_encoding)
        X_num = self.feature_encoder_.fit(X).transform(X)

        self.y_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        y_enc = self.y_encoder_.transform(np.asarray(y).ravel()).astype(np.int64)

        X_tr, y_tr, X_val, y_val = self._make_val_split(X_num, y_enc)

        # A fresh engine every fit: hyperparameters may have been updated by the
        # TuningManager ('finetune' = full retraining with new hyperparams).
        self.model_ = self._build_engine()
        logger.info(
            "[XRFM] Fitting xRFM classifier (n=%d, d=%d, %d classes, kernel=%s)",
            X_tr.shape[0], X_tr.shape[1], self.n_classes_, self.kernel,
        )
        self.model_.fit(X_tr, y_tr, X_val, y_val)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("XRFMClassifier must be fitted before predict().")
        X_num = self.feature_encoder_.transform(X)
        pred_enc = np.asarray(self.model_.predict(X_num)).ravel().astype(int)
        return self.y_encoder_.inverse_transform(pred_enc)

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("XRFMClassifier must be fitted before predict_proba().")
        X_num = self.feature_encoder_.transform(X)
        proba = np.asarray(self.model_.predict_proba(X_num), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    # -- adaptation hooks used by the TuningManager ---------------------------
    def transform_features(self, X):
        """Numeric float32 features in the fitted encoder's space."""
        if self.feature_encoder_ is None:
            raise RuntimeError("XRFMClassifier feature encoder not fitted yet.")
        return self.feature_encoder_.transform(X)

    def leaf_models(self):
        """All fitted per-leaf RFM kernel machines across the xRFM trees."""
        if self.model_ is None or self.model_.trees is None:
            return []
        leaves = []
        for tree in self.model_.trees:
            leaves.extend(node["model"] for node in self.model_._collect_leaf_nodes(tree))
        return leaves

    def numeric_targets(self, y):
        """Encode original-space labels to xRFM's numeric regression targets."""
        import torch

        y_enc = self.y_encoder_.transform(np.asarray(y).ravel()).astype(np.int64)
        y_t = torch.from_numpy(y_enc)
        converter = getattr(self.model_, "class_converter_", None)
        if converter is None:
            raise RuntimeError("XRFM engine has no fitted class converter; call fit() first.")
        return converter.labels_to_numerical(y_t)

    # -- torch-like passthroughs (the pipeline probes .to()) ------------------
    def to(self, device):
        self.device = str(device)
        if self.model_ is not None:
            self.model_.to(device)
        return self

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def parameters(self):
        return iter(())
