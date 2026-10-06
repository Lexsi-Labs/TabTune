# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Xiaomi TabLDM, vendored for TabTune.

Upstream ``Xiaomi-TabLDM`` 0.1.0 (Apache-2.0), unchanged except that its ten
absolute ``tabldm.`` imports were rewritten to this package root. TabTune's own
additions - fine-tuning, LoRA and the TabTune constructor conventions - live in
``tabtune_support`` rather than in the vendored modules.

``__version__`` is pinned to ``VENDORED_FROM`` because the vendored copy has no
distribution metadata of its own for ``importlib.metadata`` to read.
"""

# ``_model`` first: the vendored ``_sklearn`` modules import ``InferenceConfig``
# from this package root, so it has to be bound before they are imported.
from ._model import InferenceConfig
from ._sklearn import TabLDMClassifier, TabLDMRegressor
from .tabtune_support import (
    TABLDM_LORA_EXCLUDE,
    TABLDM_LORA_TARGETS,
    TabLDMTabTuneClassifier,
    TabLDMTabTuneRegressor,
    finetune,
    torch_module,
)

#: Upstream release this tree was copied from.
VENDORED_FROM = "0.1.0"
__version__ = VENDORED_FROM

__all__ = [
    "InferenceConfig",
    "TABLDM_LORA_EXCLUDE",
    "TABLDM_LORA_TARGETS",
    "TabLDMClassifier",
    "TabLDMRegressor",
    "TabLDMTabTuneClassifier",
    "TabLDMTabTuneRegressor",
    "VENDORED_FROM",
    "finetune",
    "torch_module",
]
