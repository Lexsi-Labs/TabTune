"""NVIDIA Kumo Tabular, vendored for TabTune.

``_sdm`` holds the parts of NVIDIA structured-data-models (Apache-2.0) that
KumoTabular needs, with package-relative imports and Python 3.10 shims.
"""

from .tabtune_support import (
    DEFAULT_NUM_ESTIMATORS,
    DEFAULT_REPO_ID,
    KUMO_FUNCTIONAL_WEIGHT_LEAVES,
    KUMO_LORA_EXCLUDE,
    KUMO_LORA_TARGETS,
    KumoTabularClassifier,
    KumoTabularRegressor,
    finetune,
    load_model,
    resolve_checkpoint,
)

VENDORED_FROM = "NVIDIA/structured-data-models main, 2026-09-29"

__all__ = [
    "DEFAULT_NUM_ESTIMATORS",
    "DEFAULT_REPO_ID",
    "KUMO_FUNCTIONAL_WEIGHT_LEAVES",
    "KUMO_LORA_EXCLUDE",
    "KUMO_LORA_TARGETS",
    "KumoTabularClassifier",
    "KumoTabularRegressor",
    "VENDORED_FROM",
    "finetune",
    "load_model",
    "resolve_checkpoint",
]
