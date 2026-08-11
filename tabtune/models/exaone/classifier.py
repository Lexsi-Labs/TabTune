"""scikit-learn compatible EXAONE Tabular classifier wrapper for TabTune.

This is **not** an API-call wrapper. TabTune vendors the entire EXAONE Tabular
PyTorch stack under ``tabtune/models/exaone/`` — the Cross-axis Summary
Transformer, its feature/label encoders, the preprocessor, the ensemble planner,
the ECOC decomposition, the attention-based feature selector, the CUDA execution
planner and the checkpoint validator. This class gives that vendored stack the
uniform TabTune contract so the ``TabularPipeline`` / ``TuningManager`` /
leaderboard / ensemble / distillation / explainability machinery treats it like
every other model:

* ``fit`` / ``predict`` / ``predict_proba`` — real in-context inference through
  the vendored estimator (feature selection, preprocessing, 8-member ensembling,
  ECOC above ten classes);
* ``model_`` — the real ``torch.nn.Module`` the ``TuningManager`` fine-tunes and
  PEFT injects into;
* ``y_encoder_`` / ``classes_`` / ``n_classes_`` — label bookkeeping for the
  pipeline's evaluation and probability alignment;
* ``episode_logits`` — the model's **real, differentiable** support/query forward,
  used by fine-tuning and by support-set attribution;
* ``prepare_episode_features`` — the preprocessing those episodes run on.

All heavy imports (torch, the vendored stack) happen lazily inside
``_load_model`` so ``import tabtune`` works without ever touching this model.

Licensing
---------
The **code** is BSD-3-Clause-LG AI Research (commercial use permitted, see
``LICENSE`` in this directory). The **weights** are EXAONE AI Model License
Agreement 1.1 - NC: research use only, commercial use expressly prohibited. The
registry records ``commercial_use_ok=False`` accordingly.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from . import backbone as _bk
from .episode_features import EXAONEFeatureEncoder

logger = logging.getLogger(__name__)

#: Pipeline-level keys that must never be forwarded to the vendored stack.
_NON_ENGINE_KWARGS = {"task_type", "tuning_strategy", "checkpoint_dir", "checkpoint_path"}

#: The only knobs the released manifest exposes. The architecture is frozen: the
#: checkpoint loader rebuilds a fresh model and compares key order and shapes, so
#: any architectural override makes the weights unloadable.
_MANIFEST_KWARGS = {"ensemble_count", "compute_dtype", "seed"}


class EXAONETabularClassifier(ClassifierMixin, BaseEstimator):
    """EXAONE Tabular (LG AI Research) in-context classifier, TabTune contract.

    Args:
        device: ``'cpu'`` / ``'cuda'`` / ``None``. ``None`` picks CUDA when
            available.
        dtype: Compute dtype. ``None`` means float32 on CPU and the manifest's
            float16 on CUDA — the released manifest's float16 has no CPU kernel
            for the flash-pinned attention module.
        checkpoint_path: Local ``.safetensors`` file or an ``owner/name`` Hub
            repo. ``None`` fetches the released classification checkpoint.
        n_ensemble: Ensemble members (released default 8). Member 0 is the
            identity member; set ``1`` for interpretable per-row attribution.
        max_vram_bytes: CUDA-only budget for the execution planner.
        tuning_strategy: ``'inference'`` / ``'finetune'`` / ``'peft'``.
        random_state: Seed for the ensemble generator and support subsampling.

    Attributes:
        model_: The vendored ``ClassificationModel`` (a real ``nn.Module``).
        estimator_: The vendored ``EXAONETabularClassifier`` driving inference.
        manifest_: The ``InferenceManifest`` the backbone was built from.
        feature_encoder_: The fitted :class:`EXAONEFeatureEncoder`.
        y_encoder_, classes_, n_classes_, n_features_in_: Label bookkeeping.

    Note:
        The architecture caps at ten classes and one hundred features. Neither is
        an error: above ten classes the vendored stack runs an ECOC
        decomposition (one full ensemble forward per codebook row, so cost grows),
        and above one hundred features it selects a subset by attention. Above
        100,000 support rows it randomly subsamples. TabTune surfaces all three as
        soft limits in the capability envelope rather than letting them surprise
        you as a silent slowdown.
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
        if tuning_strategy not in ("inference", "finetune", "peft"):
            raise ValueError(
                f"Unknown tuning_strategy {tuning_strategy!r}; expected "
                "'inference', 'finetune' or 'peft'."
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
        self.y_encoder_: Optional[LabelEncoder] = None
        self.classes_ = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self._is_fitted = False

    # -- sklearn plumbing ----------------------------------------------------
    def _more_tags(self):
        return {"non_deterministic": True, "allow_nan": True}

    def _manifest_overrides(self) -> dict:
        """Collect the three manifest knobs, ignoring anything else."""
        overrides = {}
        if self.n_ensemble is not None:
            overrides["ensemble_count"] = int(self.n_ensemble)
        if self.random_state is not None:
            overrides["seed"] = int(self.random_state)
        for key, value in self._extra_kwargs.items():
            if key in _MANIFEST_KWARGS:
                overrides[key] = value
            else:
                logger.debug("[EXAONE] ignoring unknown model_param %r", key)
        return overrides

    # -- loading -------------------------------------------------------------
    def _resolved_dtype(self):
        """Compute dtype for this build; ``None`` defers to the manifest.

        The architecture constrains this more than it first appears.
        ``feature_context_attentions`` is pinned to the **flash** SDPA backend
        (``layer.py``), and on CUDA flash accepts only ``{Half, BFloat16}``. So
        float32 is not merely slower there, it has *no kernel at all* --
        ``RuntimeError: No available kernel``. float32 is a CPU-only choice for
        this model.

        That leaves a three-way split:

        * **CPU** -- ``None``; the manifest already forces float32, and SDPA
          falls back to its math kernel, which handles it.
        * **CUDA inference** -- ``None``; the manifest's float16 is what the
          checkpoint was released for and what it is fastest in.
        * **CUDA fine-tuning** -- half precision is mandatory, but plain float16
          underflows on the backward pass without loss scaling: the loss reaches
          NaN within a few steps and the optimizer writes NaN into every weight.
          The damage then surfaces somewhere unrelated, because the support-cache
          check compares with ``torch.equal``, which returns False for NaN even
          on bit-identical tensors -- so the user sees "inputs are incompatible
          with the support cache" from a completely different code path.

          **bfloat16** is the answer where the GPU has it (Ampere and later): it
          carries float32's exponent range, so there is nothing to underflow and
          no scaler is needed. On older cards (Turing, e.g. a T4) it falls back
          to float16, and the fine-tuning loop pairs that with a
          ``torch.amp.GradScaler``.

        An explicit ``dtype=`` (or ``compute_dtype`` in ``model_params``) always
        wins, for anyone who knows better than this heuristic.
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

    def _load_model(self):
        """Load the pretrained backbone and build the vendored estimator.

        Idempotent — the pipeline calls it at construction for finetune/peft and
        the ``TuningManager`` calls it again before tuning.
        """
        if self.model_ is not None and self.estimator_ is not None:
            return
        overrides = self._manifest_overrides()
        dtype = self._resolved_dtype()
        self.model_, self.manifest_ = _bk.load_backbone(
            "classification",
            device=self.device,
            dtype=dtype,
            checkpoint_path=self.checkpoint_path,
            ensemble_count=overrides.get("ensemble_count"),
            seed=overrides.get("seed"),
        )
        self.estimator_ = _bk.build_estimator(
            "classification", self.model_, self.manifest_,
            device=self.device, max_vram_bytes=self.max_vram_bytes,
        )

    # -- core API ------------------------------------------------------------
    def fit(self, X, y):
        self._load_model()
        self.feature_encoder_ = EXAONEFeatureEncoder().fit(X)
        self.y_encoder_ = LabelEncoder().fit(_as_1d(y))
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = int(self.feature_encoder_.n_features_in_)

        features = self.feature_encoder_.transform(X)
        targets = self.y_encoder_.transform(_as_1d(y))
        logger.info(
            "[EXAONE] Fitting in-context support: %d rows x %d features, %d classes%s",
            features.shape[0], features.shape[1], self.n_classes_,
            " (ECOC decomposition engages)" if self.n_classes_ > _bk.CLASS_CAPACITY else "",
        )
        self.estimator_.fit(features, targets)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError(
                "EXAONETabularClassifier must be fitted before predict_proba()."
            )
        features = self.feature_encoder_.transform(X)
        proba = np.asarray(self.estimator_.predict_proba(features), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("EXAONETabularClassifier must be fitted before predict().")
        encoded = np.asarray(self.estimator_.predict(self.feature_encoder_.transform(X)))
        # The vendored estimator returns the codes it was fitted on; map back to
        # the caller's original label space.
        return self.y_encoder_.inverse_transform(encoded.astype(int))

    # -- fine-tuning / attribution hooks -------------------------------------
    @property
    def max_classes(self) -> int:
        """Architectural head width, **not** the dataset's class count."""
        if self.manifest_ is not None:
            return int(self.manifest_.model.class_capacity)
        return int(_bk.CLASS_CAPACITY)

    def prepare_episode_features(self, X_raw):
        """Numeric features for a fine-tuning or attribution episode.

        Returns ``(X[N, K] float32, None)``. The second element is the categorical
        mask other TabTune models return; EXAONE has none to give — it infers
        categoricals itself, per ensemble member, from the support set's distinct
        value counts.
        """
        if self.feature_encoder_ is None:
            raise RuntimeError(
                "EXAONETabularClassifier must be fitted before "
                "prepare_episode_features(); the encoder's category vocabulary is "
                "learned from the training frame."
            )
        return self.feature_encoder_.transform(X_raw).astype(np.float32), None

    def episode_logits(
        self, x_support, y_support, x_query, *, feedforward_token_chunk=None
    ):
        """Real differentiable support/query forward — ``(E, Q, class_capacity)``.

        Deliberately bypasses the vendored ``predict_proba``, which runs under
        ``torch.inference_mode()``. Tensors produced there are inference tensors:
        they can never enter autograd, even later, so a fine-tuning loop built on
        that path raises rather than silently under-training.

        Note:
            Differentiable with respect to ``x_support``, ``x_query`` and the
            parameters. **Not** with respect to ``y_support`` — the label encoder
            ranks labels by comparison and count, which has zero gradient.
        """
        if self.model_ is None:
            raise RuntimeError(
                "EXAONE backbone not loaded; call _load_model() first."
            )
        kwargs = {}
        if feedforward_token_chunk is not None:
            kwargs["feedforward_token_chunk"] = int(feedforward_token_chunk)
        return _bk.icl_logits(self.model_, x_support, y_support, x_query, **kwargs)

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


def _as_1d(y):
    array = np.asarray(y)
    return array.ravel() if array.ndim > 1 else array


__all__ = ["EXAONETabularClassifier"]
