"""Safe loading of explicit classification inference checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import logging
import os

import torch

from .config import InferenceManifest
from .model.heads import ClassificationModel, RegressionModel


_logger = logging.getLogger(__name__)


_CHECKPOINT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_DTYPES_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _model_config(model: ClassificationModel):
    for name in ("model_config", "config"):
        if hasattr(model, name):
            return getattr(model, name)
    return None


def _validate_model_manifest(model: ClassificationModel, manifest: InferenceManifest) -> None:
    manifest_model = manifest.model
    configured = _model_config(model)
    if configured is not None and configured != manifest_model:
        raise ValueError("model architecture does not match the manifest")

    for name in ("class_capacity", "columns_per_group"):
        expected = getattr(manifest_model, name)
        actual = getattr(model, name, getattr(configured, name, None))
        if actual != expected:
            raise ValueError("model structure does not match the manifest")

    dtype_name = manifest.runtime.compute_dtype
    expected_dtype = _DTYPES_BY_NAME.get(str(dtype_name))
    if expected_dtype is None:
        expected_dtype = _DTYPES_BY_NAME.get(getattr(dtype_name, "value", ""))
    state = model.state_dict()
    if not state or expected_dtype is None:
        raise ValueError("model persistent state does not match the manifest")
    if any(not tensor.is_floating_point() or tensor.dtype != expected_dtype for tensor in state.values()):
        raise ValueError("model persistent dtype does not match the manifest")

    # A freshly constructed manifest model is an unambiguous description of the
    # complete public persistent layout.  Preserve the caller's random stream:
    # initialization values are irrelevant here; only keys and shapes matter.
    random_state = torch.random.get_rng_state()
    try:
        expected_model = ClassificationModel(manifest_model)
    except Exception as error:
        raise ValueError("manifest does not describe this model architecture") from error
    finally:
        torch.random.set_rng_state(random_state)
    expected_state = expected_model.state_dict()
    if tuple(state) != tuple(expected_state):
        raise ValueError("model persistent layout does not match the manifest")
    if any(state[key].shape != expected_state[key].shape for key in state):
        raise ValueError("model persistent layout does not match the manifest")


def _validate_regression_model_manifest(
    model: RegressionModel, manifest: InferenceManifest
) -> None:
    if manifest.task != "regression" or manifest.regression is None:
        raise ValueError("manifest does not describe quantile regression")
    if model.config != manifest.model:
        raise ValueError("model architecture does not match the manifest")
    if model.regression_config != manifest.regression:
        raise ValueError("regression architecture does not match the manifest")
    if model.output_width != manifest.regression.quantile_count:
        raise ValueError("model output width does not match the manifest")

    expected_dtype = _DTYPES_BY_NAME.get(manifest.runtime.compute_dtype)
    state = model.state_dict()
    if not state or expected_dtype is None:
        raise ValueError("model persistent state does not match the manifest")
    for key, tensor in state.items():
        if not tensor.is_floating_point():
            raise ValueError("model persistent state must be floating point")
        required_dtype = torch.float32 if key == "quantile_levels" else expected_dtype
        if tensor.dtype != required_dtype:
            raise ValueError("model persistent dtype does not match the manifest")

    random_state = torch.random.get_rng_state()
    try:
        expected_model = RegressionModel(manifest.model, manifest.regression)
    except Exception as error:
        raise ValueError("manifest does not describe this model architecture") from error
    finally:
        torch.random.set_rng_state(random_state)
    expected_state = expected_model.state_dict()
    if tuple(state) != tuple(expected_state):
        raise ValueError("model persistent layout does not match the manifest")
    if any(state[key].shape != expected_state[key].shape for key in state):
        raise ValueError("model persistent layout does not match the manifest")
    if not torch.equal(
        model.quantile_levels.detach().cpu(), expected_model.quantile_levels
    ):
        raise ValueError("model quantile levels do not match the manifest")


def _digest(path: str | bytes) -> str:
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def _verify_digest(filesystem_path: str, manifest: InferenceManifest) -> None:
    """Enforce the manifest's SHA-256 pin, or say out loud that there isn't one."""
    if manifest.checkpoint_sha256 is None:
        _logger.warning(
            "manifest pins no checkpoint_sha256; loading %s without an integrity "
            "check (structural validation still runs)",
            filesystem_path,
        )
        return
    if _digest(filesystem_path) != manifest.checkpoint_sha256:
        raise ValueError("checkpoint digest does not match the manifest")


def _read_state_dict(filesystem_path: str):
    """Deserialize a weights-only checkpoint, dispatching on file format.

    ``.safetensors`` files are read with the safetensors reader (the released
    format); anything else is treated as a torch ``weights_only`` archive. Both
    yield a flat mapping of string keys to CPU tensors, which the callers then
    validate against the model — the choice of reader changes only how bytes are
    decoded, never what is accepted.
    """
    if os.path.splitext(filesystem_path)[1].lower() == ".safetensors":
        from safetensors.torch import load_file

        try:
            return load_file(filesystem_path, device="cpu")
        except Exception as error:
            raise ValueError("checkpoint could not be safely deserialized") from error
    try:
        return torch.load(filesystem_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint could not be safely deserialized") from error


def _clear_runtime_caches(model: ClassificationModel) -> None:
    """Clear the model's explicitly exposed inference caches."""
    clear_model = getattr(model, "clear_cache", None)
    if callable(clear_model):
        clear_model()
        return

    transformer = getattr(model, "transformer", None)
    clear_transformer = getattr(transformer, "clear_cache", None)
    if callable(clear_transformer):
        clear_transformer()

    for owner, name in (
        (model, "support_cache"),
        (model, "attention_cache"),
        (transformer, "support_cache"),
        (transformer, "attention_cache"),
    ):
        cache = getattr(owner, name, None)
        if isinstance(cache, (dict, list, set)):
            cache.clear()
        elif cache is not None:
            setattr(owner, name, None)


def load_classifier_checkpoint(path, model, manifest, *, verify_checksum=True):
    """Validate and copy one local weights-only checkpoint into ``model``.

    ``verify_checksum`` enforces the manifest's SHA-256 pin. It is left on for the
    released checkpoint and turned off for caller-supplied weights, which have no
    canonical digest to match; a manifest that pins no digest warns and continues.
    Every other validation (keys, shapes, dtype, finiteness) always runs.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like object")
    filesystem_path = os.fspath(path)
    if not isinstance(model, ClassificationModel):
        raise TypeError("model must be a ClassificationModel")
    if not isinstance(manifest, InferenceManifest):
        raise TypeError("manifest must be an InferenceManifest")
    _validate_model_manifest(model, manifest)

    if not os.path.exists(filesystem_path):
        raise FileNotFoundError(filesystem_path)
    if not os.path.isfile(filesystem_path):
        raise IsADirectoryError(filesystem_path)
    if verify_checksum:
        _verify_digest(filesystem_path, manifest)

    checkpoint = _read_state_dict(filesystem_path)

    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise TypeError("checkpoint root must be a nonempty mapping")
    if any(not isinstance(key, str) for key in checkpoint):
        raise TypeError("checkpoint keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError("checkpoint values must be tensors")

    targets = model.state_dict()
    if set(checkpoint) != set(targets):
        raise ValueError("checkpoint state keys do not match the model")
    for key, value in checkpoint.items():
        target = targets[key]
        if value.layout is not torch.strided or value.dtype not in _CHECKPOINT_DTYPES:
            raise ValueError("checkpoint tensor has an unsupported domain")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError("checkpoint tensor shape does not match the model")
        if value.numel() == 0 and target.numel() != 0:
            raise ValueError("checkpoint tensor emptiness does not match the model")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("checkpoint tensor contains a nonfinite value")

    backups = {key: value.detach().clone() for key, value in targets.items()}
    try:
        with torch.no_grad():
            for key, value in checkpoint.items():
                targets[key].copy_(value)
        _clear_runtime_caches(model)
    except Exception:
        with torch.no_grad():
            for key, value in backups.items():
                targets[key].copy_(value)
        raise
    del backups, targets, checkpoint
    return model


def load_regressor_checkpoint(path, model, manifest, *, fs_cfg=None, verify_checksum=True):
    """Validate and atomically load a regression checkpoint.

    ``verify_checksum`` gates the manifest SHA-256 pin (see
    ``load_classifier_checkpoint``); all structural validation always runs.
    """
    if fs_cfg is not None:
        raise ValueError("fs_cfg is unavailable for regression")
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like object")
    if not isinstance(model, RegressionModel):
        raise TypeError("model must be a RegressionModel")
    if not isinstance(manifest, InferenceManifest):
        raise TypeError("manifest must be an InferenceManifest")
    if manifest.task != "regression":
        raise ValueError("manifest task must be regression")
    _validate_regression_model_manifest(model, manifest)
    filesystem_path = os.fspath(path)
    if not os.path.exists(filesystem_path):
        raise FileNotFoundError(filesystem_path)
    if not os.path.isfile(filesystem_path):
        raise IsADirectoryError(filesystem_path)
    if verify_checksum:
        _verify_digest(filesystem_path, manifest)
    checkpoint = _read_state_dict(filesystem_path)
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise TypeError("checkpoint root must be a nonempty mapping")
    targets = model.state_dict()
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
        raise TypeError("checkpoint must map string keys to tensors")
    if set(checkpoint) != set(targets):
        raise ValueError("checkpoint state keys do not match the model")
    for key, value in checkpoint.items():
        target = targets[key]
        if value.layout is not torch.strided or value.dtype not in _CHECKPOINT_DTYPES:
            raise ValueError("checkpoint tensor has an unsupported domain")
        if value.shape != target.shape:
            raise ValueError("checkpoint tensor metadata does not match the model")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("checkpoint tensor contains a nonfinite value")
    levels = checkpoint.get("quantile_levels")
    if not isinstance(levels, torch.Tensor) or levels.dtype != torch.float32:
        raise ValueError("checkpoint quantile levels must use float32")
    if not torch.equal(levels, model.quantile_levels.detach().cpu()):
        raise ValueError("checkpoint quantile levels do not match the model")
    backups = {key: value.detach().clone() for key, value in targets.items()}
    try:
        with torch.no_grad():
            for key, value in checkpoint.items():
                targets[key].copy_(value)
        _clear_runtime_caches(model)
    except Exception:
        with torch.no_grad():
            for key, value in backups.items():
                targets[key].copy_(value)
        raise
    del backups, targets, checkpoint
    return model


def load_checkpoint(path, model, manifest, *, fs_cfg=None, verify_checksum=True):
    """Dispatch safe loading according to the manifest task."""
    if not isinstance(manifest, InferenceManifest):
        raise TypeError("manifest must be an InferenceManifest")
    if manifest.task == "regression":
        if fs_cfg is not None:
            raise ValueError("fs_cfg is unavailable for regression")
        return load_regressor_checkpoint(
            path, model, manifest, verify_checksum=verify_checksum
        )
    return load_classifier_checkpoint(
        path, model, manifest, verify_checksum=verify_checksum
    )
