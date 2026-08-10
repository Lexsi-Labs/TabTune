"""
xRFM regression wrapper for the TabTune pipeline.

Drives the **vendored** xRFM engine (Recursive Feature Machines under
``tabtune/models/xrfm/``) behind the uniform TabTune contract.  Raw mixed-type
frames come in; features are one-hot/standardise encoded internally (upstream
README recipe) and the target is standardised (``y_mean_`` / ``y_std_``) before
kernel training.  The fitted vendored ``xRFM`` estimator is exposed as
``self.model_`` -- it is a kernel machine (AGOP-learned Mahalanobis matrix M +
kernel ridge weights), NOT a ``torch.nn.Module``, so the ``TuningManager``
adapts it by RFM refitting/refinement ('finetune') rather than gradient descent
and checkpoints it via joblib.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from ...xrfm.preprocessing import XRFMFeatureEncoder

logger = logging.getLogger(__name__)

_ENGINE_KWARGS = {
    "max_leaf_size", "min_subset_size", "number_of_splits", "n_trees",
    "n_tree_iters", "split_method", "categorical_info", "time_limit_s",
    "n_threads", "refill_size", "split_temperature", "overlap_fraction",
    "use_temperature_tuning", "keep_weight_frac_in_predict",
    "max_leaf_count_in_ensemble", "temp_tuning_space",
}


class XRFMRegressorWrapper(BaseEstimator, RegressorMixin):
    """xRFM (Recursive Feature Machine) tabular regressor with the TabTune contract."""

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
        tuning_metric: str = "mse",
        categorical_encoding: str = "onehot",
        val_size: float = 0.2,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        **kwargs,
    ):
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError("XRFM regression supports tuning_strategy in {'inference','finetune'}.")
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
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self._is_fitted = False

    def _engine_kwargs(self):
        kw = {}
        for k, v in self._extra_kwargs.items():
            if k in _ENGINE_KWARGS:
                kw[k] = v
            else:
                logger.debug("[XRFM] ignoring unknown regression model_param %r", k)
        return kw

    def _resolved_device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolved_rfm_params(self):
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
        from ...xrfm.xrfm import xRFM  # lazy: pulls torch + the full engine

        return xRFM(
            rfm_params=self._resolved_rfm_params(),
            device=self._resolved_device(),
            n_trees=self.n_trees,
            tuning_metric=self.tuning_metric,
            random_state=self.random_state,
            verbose=self.verbose,
            **self._engine_kwargs(),
        )

    def _initialize_model_variables(self):
        self._load_model()

    def _load_model(self):
        if self.model_ is None:
            self.model_ = self._build_engine()

    def _make_val_split(self, X_num, y_scaled):
        n = X_num.shape[0]
        if n < 8 or int(round(self.val_size * n)) < 1:
            logger.debug("[XRFM] Dataset too small for a holdout; validating on the training split.")
            return X_num, y_scaled, X_num.copy(), y_scaled.copy()
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_num, y_scaled, test_size=self.val_size, random_state=self.random_state,
        )
        return X_tr, y_tr, X_val, y_val

    def fit(self, X, y):
        self.feature_encoder_ = XRFMFeatureEncoder(categorical_encoding=self.categorical_encoding)
        X_num = self.feature_encoder_.fit(X).transform(X)

        y_arr = np.asarray(y, dtype=float).ravel()
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr) + 1e-8)
        y_scaled = ((y_arr - self.y_mean_) / self.y_std_).astype(np.float32)

        X_tr, y_tr, X_val, y_val = self._make_val_split(X_num, y_scaled)

        self.model_ = self._build_engine()
        logger.info("[XRFM] Fitting xRFM regressor (n=%d, d=%d, kernel=%s)",
                    X_tr.shape[0], X_tr.shape[1], self.kernel)
        self.model_.fit(X_tr, y_tr, X_val, y_val)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("XRFMRegressorWrapper must be fitted before predict().")
        X_num = self.feature_encoder_.transform(X)
        pred = np.asarray(self.model_.predict(X_num), dtype=float).ravel()
        return pred * self.y_std_ + self.y_mean_

    # -- adaptation hooks used by the TuningManager ---------------------------
    def transform_features(self, X):
        if self.feature_encoder_ is None:
            raise RuntimeError("XRFMRegressorWrapper feature encoder not fitted yet.")
        return self.feature_encoder_.transform(X)

    def leaf_models(self):
        if self.model_ is None or self.model_.trees is None:
            return []
        leaves = []
        for tree in self.model_.trees:
            leaves.extend(node["model"] for node in self.model_._collect_leaf_nodes(tree))
        return leaves

    def numeric_targets(self, y):
        """Standardised (N, 1) float targets in the engine's training space."""
        import torch

        y_arr = np.asarray(y, dtype=float).ravel()
        y_scaled = ((y_arr - self.y_mean_) / self.y_std_).astype(np.float32)
        return torch.from_numpy(y_scaled).reshape(-1, 1)

    # -- torch-like passthroughs ----------------------------------------------
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
