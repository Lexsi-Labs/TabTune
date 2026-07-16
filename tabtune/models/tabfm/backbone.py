"""
Loading + forward helpers for the **vendored** TabFM architecture.

Unlike a thin API wrapper, TabTune vendors the real TabFM PyTorch model
(``tabtune/models/tabfm/model/model.py``), the Hugging Face loader
(``model_loading.py``) and the full scikit-learn inference/preprocessing engine
(``classifier_and_regressor.py``) -- all ported from google-research/tabfm
(Apache-2.0), mirroring how ``tabpfnv3`` is vendored.  These helpers are the thin
glue TabTune uses to (1) load the pretrained backbone, (2) build the vendored
estimator around it, and (3) run the **real** differentiable in-context forward
for fine-tuning / PEFT.

All heavy imports (torch, the vendored engine) are done lazily inside functions
so ``import tabtune`` succeeds on machines without the optional TabFM stack.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

TABFM_HF_REPO = "google/tabfm-1.0.0-pytorch"


def _resolve_dtype(dtype):
    """Map a string / torch dtype / None to a torch dtype (or None = float32)."""
    import torch

    if dtype is None:
        return None  # keep float32 weights (best for CPU + fine-tuning stability)
    if isinstance(dtype, str):
        return {
            "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        }.get(dtype.lower(), None)
    return dtype


def load_backbone(model_type: str = "classification", device: Optional[str] = None,
                  dtype: Any = None, checkpoint_path: Optional[str] = None):
    """Load the pretrained TabFM backbone (real vendored nn.Module).

    Weights are pulled from the Hugging Face Hub (``google/tabfm-1.0.0-pytorch``)
    on first use, or from ``checkpoint_path`` if given.  Defaults to float32 for
    maximum compatibility; pass ``dtype='bf16'`` to engage TabFM's bfloat16
    compute design.
    """
    from .model_loading import load

    resolved = _resolve_dtype(dtype)
    logger.info("[TabFM] Loading vendored %s backbone (dtype=%s) from %s",
                model_type, resolved, checkpoint_path or TABFM_HF_REPO)
    return load(model_type=model_type, checkpoint_path=checkpoint_path,
                device=device, dtype=resolved)


def build_estimator(task_type: str, backbone: Any, **kwargs):
    """Instantiate the vendored scikit-learn estimator around a backbone."""
    from .classifier_and_regressor import TabFMClassifier as _VClf
    from .classifier_and_regressor import TabFMRegressor as _VReg

    if task_type == "regression":
        return _VReg(model=backbone, **kwargs)
    return _VClf(model=backbone, **kwargs)


def real_icl_logits(model, x, y, train_size, cat_mask=None, d=None):
    """The **real** TabFM in-context forward (differentiable, for fine-tuning).

    Parameters mirror the vendored model exactly:
      x          : [B, T, H] features (support rows followed by query rows)
      y          : [B, T] targets (only rows < train_size are used as context)
      train_size : [B] number of support/context rows per batch element
      cat_mask   : [B, H] bool or None (True = categorical column)
      d          : [B] active feature count per batch element (or None = H)

    Returns logits [B, T, K]; the caller slices ``[:, train_size:, :]`` for the
    query predictions.
    """
    return model(x, y, train_size, cat_mask=cat_mask, d=d)


def split_vendored_preprocessing(estimator, X_raw):
    """Produce model-ready normalised features + categorical mask for FT episodes.

    Reuses the vendored ``TransformToNumerical`` + ``PreprocessingPipeline``
    (the exact components the real inference path uses) so episode features live
    in the same normalised space the pretrained model expects, and derives the
    ``cat_mask`` so categorical columns route through the model's categorical
    Fourier path (``in_linear_cat``) exactly as at inference time -- otherwise
    those weights / LoRA adapters would receive no gradient during fine-tuning.

    ``TransformToNumerical`` wraps a ``ColumnTransformer`` whose transformers are
    listed ``[categorical, continuous, datetime]``, and its categorical
    ``OrdinalEncoder`` emits exactly one output column per categorical input, so
    the first ``n_cat`` output columns are the (ordinal-encoded) categoricals.

    Returns ``(X_numeric [N, H] float32, cat_mask [H] bool or None)``.
    """
    import numpy as np

    from .classifier_and_regressor import PreprocessingPipeline, TransformToNumerical

    xenc = TransformToNumerical()
    X_num = np.asarray(xenc.fit_transform(X_raw), dtype=np.float32)

    # Count categorical output columns from the fitted ColumnTransformer.
    n_cat = 0
    try:
        for name, tfm, cols in getattr(xenc.tfm_, "transformers_", []):
            if name == "categorical" and tfm != "drop":
                n_cat = len(cols)
                break
    except Exception:
        n_cat = 0

    # Normalise with the vendored power transform (matches a default ensemble view).
    try:
        norm = PreprocessingPipeline(normalization_method="power")
        X_num = np.asarray(norm.fit_transform(X_num), dtype=np.float32)
    except Exception as e:  # normalisation is best-effort; raw-numeric still trains
        logger.warning("[TabFM] vendored normalisation failed (%s); using numeric features", e)
    X_num = np.nan_to_num(X_num).astype(np.float32)

    H = X_num.shape[1]
    cat_mask = None
    if 0 < n_cat <= H:  # only if the width is unchanged and there ARE categoricals
        cat_mask = np.zeros(H, dtype=bool)
        cat_mask[:n_cat] = True
    return X_num, cat_mask
