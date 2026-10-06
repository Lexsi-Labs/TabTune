"""Store fitted context without weights and restore only into a matching runtime."""

import math
import sys
from dataclasses import fields, is_dataclass, replace
from importlib.metadata import version

import torch

from . import checkpoints
from .ecoc import ECOCCodec
from .engine import Engine, FitState
from .execution.memory import Stage
from .execution.precision import stage_dtype
from .model import ModelConfig

STATE_SCHEMA = "causilo-fitted-1"


def runtime_versions() -> dict[str, str]:
    """Record versions that determine estimator, array, and tensor reconstruction."""
    dependencies = ("numpy", "pandas", "scipy", "scikit-learn", "torch")
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        **{name: version(name) for name in dependencies},
    }


def validate_cache_layout(state: FitState, config: ModelConfig) -> None:
    """Check stored layer counts and K/V dimensions before moving caches to a device."""
    cache_groups = [] if state.caches is None else [(state.dataset, state.caches)]
    if state.codec is not None:
        rows = len(state.codec.codebook)
        if state.caches is not None or state.code_datasets is None or len(state.code_datasets) != rows:
            raise ValueError("Saved ECOC state has the wrong context count")
        if state.code_caches is not None:
            if len(state.code_caches) != rows:
                raise ValueError("Saved ECOC cache layout has the wrong code row count")
            cache_groups.extend(zip(state.code_datasets, state.code_caches))
    elif state.code_caches is not None or state.code_datasets is not None:
        raise ValueError("Saved ECOC contexts require a codebook")
    # Column caches have a feature-group axis; prediction caches attend to the
    # entire training-row sequence after feature groups have been pooled.
    for dataset, caches in cache_groups:
        groups = math.ceil(dataset.features.shape[1] / config.group_size)
        column = (1, groups, config.column_heads, config.column_latents, config.width // config.column_heads)
        prediction = (
            1,
            config.prediction_heads,
            len(dataset.features),
            config.width * config.row_latents // config.prediction_heads,
        )
        if len(caches) != len(dataset.members):
            raise ValueError("Saved cache layout has the wrong ensemble count")
        for context in caches:
            for saved, depth, dimensions in (
                (context.columns[0], config.column_depths[0], column),
                (context.columns[1], config.column_depths[1], column),
                (context.prediction, config.prediction_depth, prediction),
            ):
                if len(saved.layers) != depth or any(
                    tuple(layer.key.shape) != dimensions or tuple(layer.value.shape) != dimensions
                    for layer in saved.layers
                ):
                    raise ValueError("Saved cache layout does not match the pretrained architecture")


def transfer_state(value, device, *, clone: bool = False, dtype=None):
    """Recursively move tensors in fitted dataclasses, leaving non-tensor state intact.

    Export clones tensors onto CPU so the saved state does not alias live cache
    storage. Restoration may also convert floating tensors to the destination
    stage's precision; integer tensors retain their dtype.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        return tensor.clone() if clone else tensor
    if isinstance(value, ECOCCodec):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: transfer_state(getattr(value, field.name), device, clone=clone, dtype=dtype)
                for field in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(transfer_state(item, device, clone=clone, dtype=dtype) for item in value)
    if isinstance(value, dict):
        return {key: transfer_state(item, device, clone=clone, dtype=dtype) for key, item in value.items()}
    return value


def export_estimator(estimator) -> dict:
    """Return a versioned fitted payload; identify weights by revision rather than copying them."""
    from . import __version__

    engine = getattr(estimator, "_engine", None)
    state = engine.state if engine is not None else None
    if state is not None and state.codec is not None:
        # Code contexts share preprocessing and only differ in encoded targets.
        # Joblib duplicates aliased NumPy arrays, so persist the common dataset once.
        state = replace(state, code_datasets=None)
    return {
        "version": __version__,
        "schema": STATE_SCHEMA,
        "dependencies": runtime_versions(),
        "revision": checkpoints.RELEASE_COMMIT,
        "parameters": estimator.get_params(deep=False),
        "task": engine.task if engine is not None else None,
        "model_config": engine.model.config.record()
        if engine is not None and engine.model is not None
        else None,
        "fitted": transfer_state(state, "cpu", clone=True),
    }


def import_estimator(estimator, saved: dict) -> None:
    """Restore parameters and fitted context, loading weights but never rebuilding K/V."""
    from . import __version__

    if saved["version"] != __version__ or saved["schema"] != STATE_SCHEMA:
        raise ValueError("Restore requires the same Causilo version and fitted-state schema")
    if saved.get("dependencies") != runtime_versions():
        raise ValueError("Restore requires matching Python and dependency versions")
    if saved["revision"] != checkpoints.RELEASE_COMMIT:
        raise ValueError("Saved state belongs to a different pretrained checkpoint")
    estimator.__dict__.update(saved["parameters"])
    if saved["fitted"] is None:
        return
    # Auto resolves again on the current host; an explicit unavailable device
    # is rejected by Engine before any context is installed.
    engine = Engine(saved["task"], estimator.device)
    engine.model = checkpoints.load_pretrained_model(engine.task).to(engine.device)
    if engine.model.config.record() != saved["model_config"]:
        raise ValueError("Saved cache does not match the model shape")
    state = saved["fitted"]
    if state.codec is not None and state.code_datasets is None:
        # Reuse the saved codebook and transforms; neither fitting nor K/V rebuilds
        # are needed to restore each row's targets and shared preprocessing.
        encoded = state.codec.encode(state.dataset.targets)
        state = replace(
            state, code_datasets=tuple(replace(state.dataset, targets=row) for row in encoded)
        )
    validate_cache_layout(state, engine.model.config)
    if state.caches is not None or state.code_caches is not None:
        # Apply the same stage precision policy used by cache construction.
        column_dtype = stage_dtype(engine.task, Stage.COLUMN, engine.device)
        prediction_dtype = stage_dtype(engine.task, Stage.PREDICTION, engine.device)

        def move(caches):
            if caches is None:
                return None
            return tuple(
                replace(
                    context,
                    columns=tuple(
                        transfer_state(column, engine.device, dtype=column_dtype)
                        for column in context.columns
                    ),
                    prediction=transfer_state(context.prediction, engine.device, dtype=prediction_dtype),
                )
                for context in caches
            )

        state = replace(
            state,
            caches=move(state.caches),
            code_caches=None
            if state.code_caches is None
            else tuple(move(caches) for caches in state.code_caches),
        )
    engine.state = state
    estimator._engine = engine
    table = engine.state.dataset
    estimator.n_features_in_ = table.encoder.width
    if table.encoder.names is not None:
        import numpy as np

        if all(isinstance(name, str) for name in table.encoder.names):
            estimator.feature_names_in_ = np.asarray(table.encoder.names, dtype=object)
    if engine.task == "classification":
        estimator.classes_ = table.target_encoder.classes_
