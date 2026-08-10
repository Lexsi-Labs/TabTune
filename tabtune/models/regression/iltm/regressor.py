"""
iLTM regression wrapper for the TabTune pipeline.

Drives the **vendored** iLTM regression engine (real hypernetwork architecture
+ HF checkpoint loader + internal preprocessing/ensembling under
``tabtune/models/iltm/``) behind the uniform TabTune contract.  Inference is
delegated to the vendored ``iLTMRegressor`` on RAW mixed-type frames (iLTM
handles scaling, categoricals and missing values internally); the underlying
``torch.nn.Module`` (the iLTM hypernetwork) is exposed as ``self.model_`` so
the ``TuningManager`` can run episodic turn-by-turn fine-tuning, LoRA/PEFT and
``.pt`` state-dict checkpointing on the real modules.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

_NON_ENGINE_KWARGS = {"task_type", "tuning_strategy", "checkpoint_dir", "checkpoint_path"}


class ILTMRegressorWrapper(BaseEstimator, RegressorMixin):
    """iLTM tabular regressor with the TabTune contract."""

    def __init__(
        self,
        device: Optional[str] = None,
        checkpoint: str = "xgbrconcat",
        checkpoint_dir: Optional[str] = None,
        n_ensemble: Optional[int] = None,
        tuning_strategy: str = "inference",
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError("ILTM regression supports tuning_strategy in {'inference','finetune'}.")
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.n_ensemble = n_ensemble
        self.tuning_strategy = tuning_strategy
        self.random_state = random_state
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in _NON_ENGINE_KWARGS}

        self.estimator_ = None
        self.model_ = None  # the REAL torch.nn.Module (vendored iLTM hypernetwork)
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self.episode_featurizer_ = None
        self._is_fitted = False

    def _resolved_device(self) -> str:
        if self.device is not None:
            return str(self.device)
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _engine_kwargs(self, engine_cls) -> dict:
        import inspect

        allowed = set(inspect.signature(engine_cls.__init__).parameters) - {"self", "kwargs"}
        kw = {}
        if self.n_ensemble is not None:
            kw["n_ensemble"] = int(self.n_ensemble)
        if self.random_state is not None:
            kw["seed"] = int(self.random_state)
        for k, v in self._extra_kwargs.items():
            if k in allowed:
                kw[k] = v
            else:
                logger.debug("[ILTM] ignoring unknown regression model_param %r", k)
        # See ILTMClassifier._engine_kwargs: TabTune 'inference' must not run
        # gradient training; the TuningManager owns 'finetune'.
        kw.setdefault("finetuning", False)
        return kw

    def _initialize_model_variables(self):
        """Eager backbone load hook the pipeline calls for finetune/peft."""
        self._load_model()

    def _load_model(self):
        if self.estimator_ is not None and self.model_ is not None:
            return
        if self.checkpoint_dir:
            os.environ["ILTM_CKPT_DIR"] = str(self.checkpoint_dir)
        from ...iltm.engine import PinnedILTMRegressorEngine

        device = self._resolved_device()
        logger.info("[ILTM] Building vendored regression engine (checkpoint=%s, device=%s)", self.checkpoint, device)
        self.estimator_ = PinnedILTMRegressorEngine(
            device=device,
            checkpoint=self.checkpoint,
            **self._engine_kwargs(PinnedILTMRegressorEngine),
        )
        self.model_ = self.estimator_._initialize_model()
        self.estimator_._model = self.model_

    def fit(self, X, y):
        self._load_model()
        y_arr = np.asarray(y, dtype=float).ravel()
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr) + 1e-8)
        logger.info("[ILTM] Fitting regression engine ensemble (n=%d)", len(y_arr))
        self.estimator_.fit(_as_frame(X), y_arr)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("ILTMRegressorWrapper must be fitted before predict().")
        return np.asarray(self.estimator_.predict(_as_frame(X)), dtype=float).ravel()

    # -- fine-tuning helpers --------------------------------------------------
    def prepare_episode_features(self, X_raw):
        from ...iltm.episode_features import ILTMEpisodeFeaturizer

        if self.episode_featurizer_ is None or not self.episode_featurizer_._is_fitted:
            self.episode_featurizer_ = ILTMEpisodeFeaturizer()
            self.episode_featurizer_.fit(X_raw)
        return self.episode_featurizer_.transform(X_raw), None

    def episode_predict(self, x_support, y_support, x_query, training=True, dropout=0.0):
        """Real differentiable support/query forward; returns query scalars [Q]."""
        if self.model_ is None:
            raise RuntimeError("iLTM regression backbone not loaded.")
        from ...iltm.utils import full_main_forward

        rf, pca, main_network, norm = self.model_(x_support, y_support, 1, training=training)
        device = next(iter(p.device for p in self.model_.parameters()))
        out = full_main_forward(
            x_query, 1, int(x_query.shape[0]), vars(self.model_),
            rf, pca, norm, main_network, device, use_amp=False,
            training_finetuning=training, finetuning_dropout=dropout,
        )
        return out  # full_main_forward squeezes the regression output to [Q]

    # -- torch passthroughs ----------------------------------------------------
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
