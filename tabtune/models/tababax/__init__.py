"""BaiRong TabAbax, vendored for TabTune.

The upstream ``tababax`` 1.0.0 source tree (``src/tababax``), unchanged except
that its absolute ``tababax.`` imports were rewritten to this package root.
TabICL-derived modules stay BSD-3-Clause; see ``LICENSE``, ``NOTICE``,
``PROVENANCE.md`` and ``LICENSES/``. The checkpoints on ``tababax-ai/TabAbax``
are CC-BY-NC-4.0 (non-commercial). TabTune's additions live in
``tabtune_support``.
"""

import os
import sys

# The Xet downloader can sit at zero bytes on Windows. Plain HTTP completes.
# Limited to Windows so that importing this package does not switch the download backend for
# every later Hugging Face call in the same process (and its subprocesses) on other platforms.
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .__about__ import __version__
from ._model import InferenceConfig
from ._sklearn import TabAbaxClassifier, TabAbaxRegressor
from .tabtune_support import (
    TABABAX_LORA_EXCLUDE,
    TABABAX_LORA_TARGETS,
    TabAbaxTabTuneClassifier,
    TabAbaxTabTuneRegressor,
    finetune,
)

#: Upstream release this tree was copied from.
VENDORED_FROM = __version__

__all__ = [
    "InferenceConfig",
    "TABABAX_LORA_EXCLUDE",
    "TABABAX_LORA_TARGETS",
    "TabAbaxClassifier",
    "TabAbaxRegressor",
    "TabAbaxTabTuneClassifier",
    "TabAbaxTabTuneRegressor",
    "VENDORED_FROM",
    "finetune",
]
