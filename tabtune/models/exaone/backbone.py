"""Loading + forward helpers for the **vendored** EXAONE Tabular architecture.

TabTune vendors the whole ``exaonetabular`` runtime under
``tabtune/models/exaone/`` — the Cross-axis Summary Transformer, its encoders,
the preprocessor, the ensemble planner, the CUDA execution planner and the
checkpoint validator — the same way TabFM, TabPFN v3 and LimiX are vendored.
These helpers are the thin glue TabTune needs on top:

1. resolve and load a released checkpoint into a real ``torch.nn.Module``;
2. build the vendored estimator around it for inference;
3. run the **real, differentiable** in-context forward for fine-tuning and for
   support-set attribution.

Every heavy import (torch, the vendored stack) happens lazily inside a function
so ``import tabtune`` works on a machine that will never touch this model.

Provenance and licensing
------------------------
Code: LG AI Research, BSD-3-Clause-LG AI Research License — see ``LICENSE`` in
this directory. Commercial use of the **code** is permitted.

Weights: a *separate* licence, ``EXAONE AI Model License Agreement 1.1 - NC``,
which grants use "solely for research purposes" and expressly prohibits
commercial use. TabTune records that in the registry as
``commercial_use_ok=False``; :func:`tabtune.registry.list_models` will not return
this model under ``commercial_ok=True``. A fine-tuned checkpoint is a
"Derivative" under that agreement: still research-only, and the licence requires
its name to begin with "EXAONE".
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Hub repository hosting the released weights. Public, no token required.
EXAONE_HF_REPO = "LG-AI-Research/EXAONE-Tabular"

#: Environment variables the vendored resolver honours, per task. Point either at
#: a local ``.safetensors`` file or at an alternative ``owner/name`` repo.
WEIGHTS_ENV_VARS = {
    "classification": "EXAONETABULAR_CLASSIFIER_WEIGHTS",
    "regression": "EXAONETABULAR_REGRESSOR_WEIGHTS",
}

#: The architecture's own ceilings, read off the released manifests. Mirrored
#: into the registry's :class:`~tabtune.registry.CapabilityEnvelope`; kept here so
#: the model package is the single source of truth.
SUPPORT_ROW_LIMIT = 100_000
FEATURE_LIMIT = 100
CLASS_CAPACITY = 10


def resolve_dtype(dtype: Any):
    """Map a string / torch dtype / ``None`` to a torch dtype.

    ``None`` means "let the manifest decide" — the released manifests ask for
    float16. TabTune overrides that to float32 on CPU, where float16 matmuls are
    either unsupported or ruinously slow, and where fine-tuning in half precision
    is numerically unwise anyway.
    """
    import torch

    if dtype is None:
        return None
    if isinstance(dtype, str):
        return {
            "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
            "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        }.get(dtype.lower())
    return dtype


def resolve_device(device: Optional[str] = None) -> str:
    """Return a concrete device string safe to hand to ``torch.device``.

    Delegates to TabTune's own resolver rather than re-implementing the rules.
    That matters for more than tidiness: ``"auto"`` is the library-wide spelling
    for "pick the best backend" -- it is the default of
    :class:`~tabtune.config.schemas.TuningConfig` -- so the pipeline forwards it
    verbatim into ``model_params`` on the fine-tune path. An implementation that
    only special-cased ``None`` would pass the literal string ``"auto"`` to
    ``torch.device`` and raise, which is exactly what happened before 0.2.0.

    The shared resolver also clamps an out-of-range CUDA index and falls back to
    MPS or CPU with a warning, so EXAONE now behaves like every other model on a
    machine that cannot honour the request.
    """
    from ..._internal.device import resolve_device as _resolve

    return _resolve(device)


def build_manifest(
    task: str = "classification",
    *,
    device: Optional[str] = None,
    ensemble_count: Optional[int] = None,
    compute_dtype: Any = None,
    seed: Optional[int] = None,
):
    """Return the released :class:`InferenceManifest` with TabTune's overrides.

    Only three knobs are overridable upstream — ``ensemble_count``,
    ``compute_dtype`` and ``seed``. The architecture is immutable, because the
    checkpoint's key set and shapes are validated against a freshly built model,
    so any architectural change makes the weights unloadable.

    The one policy TabTune adds: **float32 on CPU**. The released manifests ask
    for float16, which upstream itself documents as CUDA-only (one attention
    module is pinned to the flash kernel, which has no fp32 CUDA path but also no
    fp16 CPU path). Left alone, a CPU run fails inside the first matmul.
    """
    from .presets import released_manifest

    manifest = released_manifest(task)
    if compute_dtype is None:
        resolved_device = resolve_device(device)
        compute_dtype = "float32" if str(resolved_device).startswith("cpu") else None
    if isinstance(compute_dtype, str):
        pass
    elif compute_dtype is not None:
        compute_dtype = str(compute_dtype).replace("torch.", "")
    if ensemble_count is None and compute_dtype is None and seed is None:
        return manifest
    return manifest.with_overrides(
        ensemble_count=ensemble_count, compute_dtype=compute_dtype, seed=seed
    )


def resolve_checkpoint(
    task: str = "classification",
    *,
    checkpoint_path: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Resolve a weights file, preferring an explicit path, then env, then the Hub.

    Returns the vendored ``WeightsSource`` (``.path``, ``.is_default``). Only the
    baked-in release is treated as default, which is the only case where the
    checkpoint SHA-256 pin is enforced — and both released manifests currently
    pin ``None``, so in practice structural validation is the real check.

    Raises:
        FileNotFoundError: If the regression checkpoint is requested and no local
            file was supplied. LG AI Research has not published regression
            weights; the code path is complete and exercised by tests against a
            locally built checkpoint, but there is nothing to download.
    """
    from .presets import released_checkpoint
    from .weights import resolve_weights

    record = released_checkpoint(task)
    if checkpoint_path is None and task == "regression":
        if not os.environ.get(WEIGHTS_ENV_VARS["regression"]):
            raise FileNotFoundError(
                "EXAONE Tabular regression weights are not published. LG AI "
                "Research has released only the classification checkpoint "
                f"({EXAONE_HF_REPO}); the regressor's architecture, preprocessing "
                "and fine-tuning paths are fully implemented here and work "
                "against a local file, but there is nothing to download. Supply "
                "one with TabularPipeline(..., model_params={'checkpoint_path': "
                "'/path/to/exaone-tabular-regressor.safetensors'}) or by setting "
                f"{WEIGHTS_ENV_VARS['regression']}. For a regression model that "
                "works out of the box, use TabICLv2, Mitra, LimiX or TabPFNv26."
            )
    return resolve_weights(record, weights=checkpoint_path,
                           revision=revision, cache_dir=cache_dir)


def load_backbone(
    task: str = "classification",
    *,
    device: Optional[str] = None,
    dtype: Any = None,
    checkpoint_path: Optional[str] = None,
    ensemble_count: Optional[int] = None,
    seed: Optional[int] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load the pretrained EXAONE backbone as a real vendored ``nn.Module``.

    Returns:
        ``(model, manifest)``. ``model`` is the vendored ``ClassificationModel``
        or ``RegressionModel`` with weights loaded and validated; ``manifest`` is
        the :class:`InferenceManifest` it was built from, which the estimator
        needs.
    """
    import torch

    from .checkpoint import load_classifier_checkpoint, load_regressor_checkpoint
    from .model.heads import build_model

    resolved_device = resolve_device(device)
    manifest = build_manifest(
        task, device=resolved_device, ensemble_count=ensemble_count,
        compute_dtype=None if dtype is None else _dtype_name(dtype), seed=seed,
    )
    source = resolve_checkpoint(
        task, checkpoint_path=checkpoint_path, revision=revision, cache_dir=cache_dir,
    )

    torch_dtype = resolve_dtype(manifest.runtime.compute_dtype)
    model = build_model(manifest, device=torch.device(resolved_device), dtype=torch_dtype)
    loader = load_classifier_checkpoint if task == "classification" else load_regressor_checkpoint
    loader(source.path, model, manifest, verify_checksum=source.is_default)
    model.eval()
    logger.info(
        "[EXAONE] loaded %s backbone from %s (device=%s, dtype=%s, ensemble=%d)",
        task, source.path, resolved_device, manifest.runtime.compute_dtype,
        manifest.runtime.ensemble_count,
    )
    return model, manifest


def _dtype_name(dtype: Any) -> Optional[str]:
    """Normalise a dtype to the string spelling ``RuntimeConfig`` accepts."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        canonical = {
            "fp32": "float32", "f32": "float32", "float32": "float32",
            "fp16": "float16", "f16": "float16", "half": "float16", "float16": "float16",
            "bf16": "bfloat16", "bfloat16": "bfloat16",
        }
        return canonical.get(dtype.lower(), dtype.lower())
    return str(dtype).replace("torch.", "")


def build_estimator(task: str, model, manifest, *, device: Optional[str] = None,
                    max_vram_bytes: Optional[int] = None):
    """Build the vendored estimator around an already-loaded backbone."""
    from ._upstream_classifier import EXAONETabularClassifier as _VendoredClassifier
    from ._upstream_regressor import EXAONETabularRegressor as _VendoredRegressor

    cls = _VendoredClassifier if task == "classification" else _VendoredRegressor
    return cls(manifest, device=resolve_device(device), model=model,
               max_vram_bytes=max_vram_bytes)


# --------------------------------------------------------------------------- #
# The differentiable in-context forward
# --------------------------------------------------------------------------- #
#: Token budget for the output feed-forward. The vendored CUDA planner searches
#: for this under a memory model; a fixed generous value is the faithful
#: substitute when TabTune drives the forward itself, and matches what the CPU
#: path uses.
DEFAULT_FEEDFORWARD_TOKEN_CHUNK = 1 << 19


def icl_logits(
    model,
    x_support,
    y_support,
    x_query,
    *,
    feedforward_token_chunk: int = DEFAULT_FEEDFORWARD_TOKEN_CHUNK,
    trusted_internal_inputs: bool = False,
):
    """The model's real support/query forward, **with autograd intact**.

    This is the forward both fine-tuning and support-set attribution run on. It
    deliberately does not go through the vendored ``predict_proba``: that wraps
    the call in ``torch.inference_mode()``, which produces inference tensors that
    can never enter autograd — not merely "no gradient now", but permanently
    ineligible, so even a later backward on a derived tensor raises.

    Args:
        model: A loaded ``ClassificationModel`` or ``RegressionModel``.
        x_support: ``(E, S, K)`` preprocessed support features.
        y_support: ``(E, S)`` support targets — class indices for classification,
            normalised values for regression.
        x_query: ``(E, Q, K)`` preprocessed query features.
        feedforward_token_chunk: Output feed-forward token budget.
        trusted_internal_inputs: Skip the vendored input validation. Needed when
            feeding half-precision tensors, which the validator rejects.

    Returns:
        ``(E, Q, output_width)`` — logits over ``class_capacity`` for
        classification, quantile predictions for regression. Differentiable with
        respect to ``x_support``, ``x_query`` and the parameters.

    Note:
        ``y_support`` is **not** differentiable, and cannot be made so: the label
        encoder converts labels to ordinal ranks with a comparison-and-count
        (``(values[:, None] > uniques[None, :]).sum(dim=1)``), which has zero
        gradient almost everywhere. Attribution against the *label* axis needs a
        different mechanism, such as leave-one-out.
    """
    return model(
        x_support, y_support, x_query,
        feedforward_token_chunk=int(feedforward_token_chunk),
        trusted_internal_inputs=bool(trusted_internal_inputs),
    )


def supports_grad(model) -> bool:
    """Whether this build carries the support-side gradient patch.

    Upstream detaches the cached key/value pair unconditionally in
    ``TensorAttention``, which severs the support-to-query path in every layer
    while still returning a non-``None`` — and simply wrong — gradient. TabTune's
    vendored copy gates that detach on :func:`torch.is_grad_enabled`. This probe
    lets a caller assert the patch is present rather than trust the file.
    """
    from .model.attention import _retain_or_detach

    import torch

    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    with torch.enable_grad():
        kept = _retain_or_detach(a * 1.0, b * 1.0)
    return all(t.requires_grad for t in kept)


__all__ = [
    "EXAONE_HF_REPO",
    "WEIGHTS_ENV_VARS",
    "SUPPORT_ROW_LIMIT",
    "FEATURE_LIMIT",
    "CLASS_CAPACITY",
    "DEFAULT_FEEDFORWARD_TOKEN_CHUNK",
    "build_manifest",
    "build_estimator",
    "icl_logits",
    "load_backbone",
    "resolve_checkpoint",
    "resolve_device",
    "resolve_dtype",
    "supports_grad",
]
