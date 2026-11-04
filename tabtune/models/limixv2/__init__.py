"""LimiX-2 (StableAI LimiX v2.0), vendored for TabTune.

``_upstream`` is LimiX ``inference/v2_0`` and ``model/v2_0`` with package-relative
imports and optional-dependency fallbacks for nvtx, triton and kditransform.
Built with StableAI LimiX; see ``_upstream/LICENSE.txt``.
"""

from .tabtune_support import (
    DEFAULT_CONFIGS,
    DEFAULT_REPO_IDS,
    LIMIXV2_LORA_EXCLUDE,
    LIMIXV2_LORA_TARGETS,
    LimiXV2Classifier,
    LimiXV2Regressor,
    finetune,
    resolve_checkpoint,
)

VENDORED_FROM = "LimiX-main (arch 2.0), 2026-09"

__all__ = [
    "DEFAULT_CONFIGS",
    "DEFAULT_REPO_IDS",
    "LIMIXV2_LORA_EXCLUDE",
    "LIMIXV2_LORA_TARGETS",
    "LimiXV2Classifier",
    "LimiXV2Regressor",
    "VENDORED_FROM",
    "finetune",
    "resolve_checkpoint",
]
