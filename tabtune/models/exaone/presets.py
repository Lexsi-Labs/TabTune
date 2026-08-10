"""Frozen descriptions of the released EXAONETabular checkpoints.

This module is the single source of truth for what ``from_pretrained`` loads: the
immutable :class:`InferenceManifest` plus the Hugging Face coordinates (repo id,
filename, revision) for each released checkpoint. Nothing here is data-dependent —
every value is fully determined by *which* released checkpoint you want, which is
why the estimators can hydrate themselves from a single task keyword.

Keeping the manifests here (rather than in the README or scattered call sites)
means the class-head width and the regression topology live in exactly one place
and cannot drift apart.

Neither manifest pins a ``checkpoint_sha256`` yet: the released files are not
final, and a digest can only be computed from the exact bytes that ship. Fill each
one in at release time; until then every load warns that it is unverified.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    InferenceManifest,
    ModelConfig,
    RegressionConfig,
    RuntimeConfig,
)

__all__ = [
    "ReleasedCheckpoint",
    "released_checkpoint",
    "released_manifest",
    "RELEASED_CHECKPOINTS",
    "DEFAULT_REPO_ID",
    "DEFAULT_REVISION",
]

# The Hub repository that hosts the released weights.
DEFAULT_REPO_ID = "LG-AI-Research/EXAONE-Tabular"
DEFAULT_REVISION = "main"


@dataclass(frozen=True)
class ReleasedCheckpoint:
    """Everything ``from_pretrained`` needs to fetch and verify one release."""

    task: str
    repo_id: str
    filename: str
    revision: str
    manifest: InferenceManifest
    weights_env_var: str


_CLASSIFIER = ReleasedCheckpoint(
    task="classification",
    repo_id=DEFAULT_REPO_ID,
    filename="exaone-tabular-classifier-v1_default.safetensors",
    revision=DEFAULT_REVISION,
    weights_env_var="EXAONETABULAR_CLASSIFIER_WEIGHTS",
    manifest=InferenceManifest(
        task="classification",
        model=ModelConfig(class_capacity=10),
        runtime=RuntimeConfig(ensemble_count=8, compute_dtype="float16", seed=0),
    ),
)

_REGRESSOR = ReleasedCheckpoint(
    task="regression",
    repo_id=DEFAULT_REPO_ID,
    filename="exaone-tabular-regressor-v1_default.safetensors",
    revision=DEFAULT_REVISION,
    weights_env_var="EXAONETABULAR_REGRESSOR_WEIGHTS",
    manifest=InferenceManifest(
        task="regression",
        model=ModelConfig(feature_attention_repeats=2),
        runtime=RuntimeConfig(ensemble_count=8, compute_dtype="float16", seed=0),
        regression=RegressionConfig(quantile_count=999, decoder_hidden_width=384),
    ),
)

RELEASED_CHECKPOINTS: dict[str, ReleasedCheckpoint] = {
    "classification": _CLASSIFIER,
    "regression": _REGRESSOR,
}

_TASK_ALIASES = {
    "classification": "classification",
    "classifier": "classification",
    "regression": "regression",
    "regressor": "regression",
}


def released_checkpoint(task: str) -> ReleasedCheckpoint:
    """Return the released-checkpoint record for a task ("classification"/"regression")."""
    try:
        key = _TASK_ALIASES[task]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown released checkpoint task: {task!r}; "
            f"expected one of {sorted(set(_TASK_ALIASES))}"
        ) from None
    return RELEASED_CHECKPOINTS[key]


def released_manifest(task: str) -> InferenceManifest:
    """Return the frozen inference manifest for a released checkpoint."""
    return released_checkpoint(task).manifest
