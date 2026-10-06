"""Causilo — full integration into TabTune.

Vendored source copy of https://github.com/nums-ai/causilo (v1.0.2). The tree is
unmodified: every import inside it was already relative, so nothing needed
rewriting. TabTune-specific additions live in ``tabtune_support``.

Model name in the pipeline: ``Causilo``. Classification and regression are both
supported, in inference, fine-tuning and PEFT modes.

Weights: ``nums-ai/causilo`` on the Hugging Face Hub, pinned by ``checkpoints.py``
to a release commit, downloaded on the first fit. Code is Apache-2.0; the weights
carry the separate Causilo License v1.0.
"""

#: Upstream release this tree was vendored from.
VENDORED_FROM = "1.0.2"
__version__ = VENDORED_FROM

from .estimators import CausiloClassifier, CausiloRegressor
from .tabtune_support import (
    CAUSILO_LORA_EXCLUDE,
    CAUSILO_LORA_TARGETS,
    CausiloTabTuneClassifier,
    CausiloTabTuneRegressor,
    finetune,
    torch_module,
)

__all__ = [
    "CAUSILO_LORA_EXCLUDE",
    "CAUSILO_LORA_TARGETS",
    "CausiloClassifier",
    "CausiloRegressor",
    "CausiloTabTuneClassifier",
    "CausiloTabTuneRegressor",
    "VENDORED_FROM",
    "__version__",
    "finetune",
    "torch_module",
]
