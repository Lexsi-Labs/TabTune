"""EXAONE Tabular regression wrapper for the TabTune pipeline.

Drives the **vendored** EXAONE regression stack (``RegressionModel`` — the same
Cross-axis Summary Transformer with a 999-quantile head — plus the preprocessor
and ensemble planner under ``tabtune/models/exaone/``) behind the uniform TabTune
contract. Inference goes through the vendored ``EXAONETabularRegressor``; the
underlying ``torch.nn.Module`` is exposed as ``self.model_`` so the
``TuningManager`` can fine-tune the real modules.

Weights availability
--------------------
LG AI Research has published **only the classification checkpoint**. The
regression architecture, preprocessing, prediction and fine-tuning paths are all
implemented here and exercised by the test suite against a locally built
checkpoint — but there is no file to download. Constructing this wrapper without
``checkpoint_path`` (or the ``EXAONETABULAR_REGRESSOR_WEIGHTS`` environment
variable) raises a ``FileNotFoundError`` that says so and names alternatives.

The registry marks ``'regression'`` in this model's ``experimental`` set for
exactly this reason: it is complete code with no released weights, and a user
picking a regressor off a capability list deserves to be told that up front
rather than at download time.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ...exaone import backbone as _bk
from ...exaone.episode_features import EXAONEFeatureEncoder

logger = logging.getLogger(__name__)

_NON_ENGINE_KWARGS = {"task_type", "tuning_strategy", "checkpoint_dir", "checkpoint_path"}
_MANIFEST_KWARGS = {"ensemble_count", "compute_dtype", "seed"}


class EXAONETabularRegressorWrapper(BaseEstimator, RegressorMixin):
    """EXAONE Tabular in-context regressor with the TabTune contract.

    Args:
        device: ``'cpu'`` / ``'cuda'`` / ``None`` (auto).
        dtype: Compute dtype; ``None`` means float32 on CPU.
        checkpoint_path: **Required in practice** — a local regression
            ``.safetensors`` file. See the module docstring.
        n_ensemble: Ensemble members (released default 8).
        tuning_strategy: ``'inference'`` or ``'finetune'``. PEFT is not offered
            for regression, matching the other regression wrappers.
        random_state: Seed for the ensemble generator and support subsampling.

    Attributes:
        model_: The vendored ``RegressionModel`` (a real ``nn.Module``).
        estimator_: The vendored ``EXAONETabularRegressor``.
        y_mean_ / y_std_: Target statistics. Kept for interface parity with the
            other regression wrappers; **both stay at their identity values**
            because the vendored regressor does its own target centring and
            scaling internally and ``predict`` already returns the original
            target space. Scaling here as well would double-apply it.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        dtype=None,
        checkpoint_path: Optional[str] = None,
        n_ensemble: Optional[int] = None,
        max_vram_bytes: Optional[int] = None,
        tuning_strategy: str = "inference",
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError(
                "EXAONE Tabular regression supports tuning_strategy in "
                f"{{'inference','finetune'}}; got {tuning_strategy!r}."
            )
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.n_ensemble = n_ensemble
        self.max_vram_bytes = max_vram_bytes
        self.tuning_strategy = tuning_strategy
        self.random_state = random_state
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in _NON_ENGINE_KWARGS}

        self.model_ = None
        self.estimator_ = None
        self.manifest_ = None
        self.feature_encoder_: Optional[EXAONEFeatureEncoder] = None
        self.n_features_in_: Optional[int] = None
        # Identity by design -- see the class docstring.
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self._is_fitted = False

    def _more_tags(self):
        return {"non_deterministic": True, "allow_nan": True}

    def _manifest_overrides(self) -> dict:
        overrides = {}
        if self.n_ensemble is not None:
            overrides["ensemble_count"] = int(self.n_ensemble)
        if self.random_state is not None:
            overrides["seed"] = int(self.random_state)
        for key, value in self._extra_kwargs.items():
            if key in _MANIFEST_KWARGS:
                overrides[key] = value
            else:
                logger.debug("[EXAONE] ignoring unknown regression model_param %r", key)
        return overrides

    def _resolved_dtype(self):
        """Compute dtype for this build; ``None`` defers to the manifest.

        Same three-way split as the classifier: float32 on CPU, the manifest's
        float16 for CUDA inference, and bfloat16 (or float16 plus a
        ``GradScaler``) for CUDA fine-tuning -- because
        ``feature_context_attentions`` is pinned to the flash SDPA backend,
        which has no float32 CUDA kernel. See the classifier's copy for the
        full account.
        """
        overrides = self._manifest_overrides()
        dtype = self.dtype if self.dtype is not None else overrides.get("compute_dtype")
        if dtype is not None:
            return dtype
        if self.tuning_strategy not in ("finetune", "peft"):
            return None
        if not str(_bk.resolve_device(self.device)).startswith("cuda"):
            return None

        import torch

        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    def _initialize_model_variables(self):
        """Pipeline hook — construct the backbone before fit."""
        self._load_model()

    def _load_model(self):
        if self.model_ is not None and self.estimator_ is not None:
            return
        overrides = self._manifest_overrides()
        dtype = self._resolved_dtype()
        self.model_, self.manifest_ = _bk.load_backbone(
            "regression",
            device=self.device,
            dtype=dtype,
            checkpoint_path=self.checkpoint_path,
            ensemble_count=overrides.get("ensemble_count"),
            seed=overrides.get("seed"),
        )
        self.estimator_ = _bk.build_estimator(
            "regression", self.model_, self.manifest_,
            device=self.device, max_vram_bytes=self.max_vram_bytes,
        )

    # -- core API ------------------------------------------------------------
    def fit(self, X, y):
        self._load_model()
        self.feature_encoder_ = EXAONEFeatureEncoder().fit(X)
        self.n_features_in_ = int(self.feature_encoder_.n_features_in_)
        features = self.feature_encoder_.transform(X)
        targets = np.asarray(y, dtype=np.float64).ravel()
        logger.info(
            "[EXAONE] Fitting regression support: %d rows x %d features",
            features.shape[0], features.shape[1],
        )
        self.estimator_.fit(features, targets)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict in the **original target space** (the vendored engine un-scales)."""
        if not self._is_fitted:
            raise RuntimeError(
                "EXAONETabularRegressorWrapper must be fitted before predict()."
            )
        features = self.feature_encoder_.transform(X)
        return np.asarray(self.estimator_.predict(features), dtype=float).ravel()

    # -- fine-tuning hooks ---------------------------------------------------
    def prepare_episode_features(self, X_raw):
        """Numeric features for a fine-tuning episode: ``(X[N, K] float32, None)``."""
        if self.feature_encoder_ is None:
            raise RuntimeError(
                "EXAONETabularRegressorWrapper must be fitted before "
                "prepare_episode_features()."
            )
        return self.feature_encoder_.transform(X_raw).astype(np.float32), None

    def episode_predictions(
        self, x_support, y_support, x_query, *, feedforward_token_chunk=None
    ):
        """Real differentiable support/query forward — ``(E, Q, quantile_count)``.

        The regression head emits 999 quantile levels; index 499 is the median and
        is what ``predict`` reports. Bypasses the vendored ``predict``, which runs
        under ``torch.inference_mode()`` and so produces tensors autograd can
        never accept.
        """
        if self.model_ is None:
            raise RuntimeError("EXAONE backbone not loaded; call _load_model() first.")
        kwargs = {}
        if feedforward_token_chunk is not None:
            kwargs["feedforward_token_chunk"] = int(feedforward_token_chunk)
        return _bk.icl_logits(self.model_, x_support, y_support, x_query, **kwargs)

    @property
    def median_quantile_index(self) -> int:
        """Index of the median level in the quantile head's output axis."""
        if self.manifest_ is None or self.manifest_.regression is None:
            return 499
        return int(self.manifest_.regression.quantile_count) // 2

    # -- torch passthroughs --------------------------------------------------
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


__all__ = ["EXAONETabularRegressorWrapper"]
