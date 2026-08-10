"""
scikit-learn compatible iLTM classifier wrapper for TabTune.

This is **not** an API-call wrapper: TabTune vendors the entire iLTM PyTorch
stack (hypernetwork architecture + HF checkpoint loader + preprocessing /
ensembling / retrieval engine) under ``tabtune/models/iltm/``.  This class
gives that vendored engine the uniform TabTune contract so the
``TabularPipeline`` / ``TuningManager`` / leaderboard / ensemble machinery
treats it like every other model:

* ``fit`` / ``predict`` / ``predict_proba`` -- real inference on RAW
  mixed-type frames, delegated to the vendored ``iLTMClassifier`` engine
  (iLTM handles scaling, categoricals and missing values internally);
* ``model_`` -- the real ``torch.nn.Module`` (the vendored ``iLTM``
  hypernetwork) used by the ``TuningManager`` for episodic fine-tuning,
  LoRA/PEFT injection and ``.pt`` state-dict checkpoints;
* ``y_encoder_`` / ``classes_`` / ``n_classes_`` -- label bookkeeping for the
  pipeline's evaluation and probability-alignment code;
* ``episode_logits`` -- the model's **real** differentiable support/query
  forward (hypernetwork generates an MLP from the support set; the query set
  is pushed through the generated network) used for fine-tuning;
* ``prepare_episode_features`` -- numeric featurization for FT episodes.

Checkpoints are resolved by the vendored ``model_checkpoints`` module: a
registry name (``'xgbrconcat'``, ``'r128bn'``, ...) is downloaded from
Hugging Face (``dbonet/iLTM``) on first use, while a **local ``.pth`` path is
used as-is with no network access** (this is how the test-suite runs with a
tiny randomly-initialised model).  The cache directory can be set via the
``checkpoint_dir`` argument or the ``ILTM_CKPT_DIR`` env var.

All heavy imports (torch + the vendored engine) happen lazily inside
``_load_model`` so ``import tabtune`` works without touching the iLTM stack.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Wrapper-level constructor keys that must never be forwarded to the vendored
# engine (the pipeline sprays these onto every model).
_NON_ENGINE_KWARGS = {"task_type", "tuning_strategy", "checkpoint_dir", "checkpoint_path"}


class ILTMClassifier(ClassifierMixin, BaseEstimator):
    """iLTM (Integrated Large Tabular Model) classifier with the TabTune contract."""

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
        if tuning_strategy not in ("inference", "finetune", "peft"):
            raise ValueError(
                "ILTM supports tuning_strategy in {'inference', 'finetune', 'peft'}."
            )
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.n_ensemble = n_ensemble
        self.tuning_strategy = tuning_strategy
        self.random_state = random_state
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in _NON_ENGINE_KWARGS}

        self.estimator_ = None
        self.model_ = None  # the REAL torch.nn.Module (vendored iLTM hypernetwork)
        self.y_encoder_: Optional[LabelEncoder] = None
        self.classes_ = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.episode_featurizer_ = None
        self._is_fitted = False

    # -- sklearn plumbing -----------------------------------------------------
    def _more_tags(self):
        return {"non_deterministic": True, "allow_nan": True}

    def _resolved_device(self) -> str:
        if self.device is not None:
            return str(self.device)
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _engine_kwargs(self, engine_cls) -> dict:
        """Filter unknown pipeline keys against the vendored engine signature."""
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
                logger.debug("[ILTM] ignoring unknown model_param %r", k)
        # TabTune semantics: 'inference' must not run gradient training. The
        # engine's OWN per-predictor finetuning defaults OFF here; gradient
        # fine-tuning is done by the TuningManager ('finetune' / 'peft').
        # Users can re-enable the upstream behaviour with finetuning=True.
        kw.setdefault("finetuning", False)
        return kw

    def _load_model(self):
        """Resolve checkpoint + build the vendored engine + pin the backbone (idempotent)."""
        if self.estimator_ is not None and self.model_ is not None:
            return
        if self.checkpoint_dir:
            os.environ["ILTM_CKPT_DIR"] = str(self.checkpoint_dir)
        from .engine import PinnedILTMClassifierEngine

        device = self._resolved_device()
        logger.info("[ILTM] Building vendored engine (checkpoint=%s, device=%s)", self.checkpoint, device)
        self.estimator_ = PinnedILTMClassifierEngine(
            device=device,
            checkpoint=self.checkpoint,
            **self._engine_kwargs(PinnedILTMClassifierEngine),
        )
        # Load and PIN the real nn.Module now, so the TuningManager can inject
        # LoRA / fine-tune it and the engine is guaranteed to predict with it.
        self.model_ = self.estimator_._initialize_model()
        self.estimator_._model = self.model_

    # -- core API -------------------------------------------------------------
    def fit(self, X, y):
        self._load_model()
        self.y_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        logger.info("[ILTM] Fitting engine ensemble (%s classes)", self.n_classes_)
        self.estimator_.fit(_as_frame(X), _as_1d(y))
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("ILTMClassifier must be fitted before predict().")
        return np.asarray(self.estimator_.predict(_as_frame(X)))

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("ILTMClassifier must be fitted before predict_proba().")
        proba = np.asarray(self.estimator_.predict_proba(_as_frame(X)), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    # -- fine-tuning helpers --------------------------------------------------
    @property
    def max_classes(self) -> int:
        """Fixed model hyperparam (hypernetwork one-hot width), NOT dataset classes."""
        return int(getattr(self.model_, "n_classes_limit", 100))

    def prepare_episode_features(self, X_raw):
        """Numeric float32 features for FT episodes: ``(X[N, H], cat_mask=None)``."""
        from .episode_features import ILTMEpisodeFeaturizer

        if self.episode_featurizer_ is None or not self.episode_featurizer_._is_fitted:
            self.episode_featurizer_ = ILTMEpisodeFeaturizer()
            self.episode_featurizer_.fit(X_raw)
        return self.episode_featurizer_.transform(X_raw), None

    def episode_logits(self, x_support, y_support, x_query, n_classes, training=True, dropout=0.0):
        """Real differentiable support/query forward.

        The hypernetwork generates the main-network weights from the support
        set; the query rows are pushed through the SAME data-dependent
        transforms (random features -> PCA -> norm) and the generated MLP.
        Gradients flow from the query loss back into the hypernetwork
        parameters (or their LoRA adapters).
        """
        if self.model_ is None:
            raise RuntimeError("iLTM backbone not loaded; call _load_model() first.")
        from .utils import full_main_forward

        rf, pca, main_network, norm = self.model_(x_support, y_support, n_classes, training=training)
        device = next(iter(p.device for p in self.model_.parameters()))
        return full_main_forward(
            x_query, n_classes, int(x_query.shape[0]), vars(self.model_),
            rf, pca, norm, main_network, device, use_amp=False,
            training_finetuning=training, finetuning_dropout=dropout,
        )

    # -- torch passthroughs so the wrapper quacks like a module for the tuner --
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
