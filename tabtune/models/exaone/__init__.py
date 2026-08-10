"""EXAONE Tabular (LG AI Research) — vendored in-context tabular foundation model.

TabTune vendors the complete ``exaonetabular`` inference runtime rather than
depending on the upstream package: the Cross-axis Summary Transformer (CAST)
architecture, its feature and label encoders, the preprocessor, the ensemble
planner, the ECOC decomposition for >10 classes, the attention-based feature
selector, the CUDA execution planner, and the checkpoint validator.

Provenance
----------
Upstream: https://github.com/LGAI-Research/EXAONE-Tabular (``src/exaonetabular``).
Vendored under ``tabtune/models/exaone/`` with three deliberate changes, each
marked ``TabTune modification`` at the site:

1. ``model/layer.py`` and ``model/transformer.py`` — absolute ``exaonetabular.*``
   imports rewritten as relative.
2. ``model/attention.py`` — the retained key/value pair is detached only when
   gradients are disabled. Upstream detaches unconditionally, which severs the
   support-to-query path in every layer while still returning a non-``None``
   gradient, so ``d(query logit)/d(support row)`` came back wrong in magnitude
   and often in sign, with no error. Inference is unchanged (it already runs
   under ``inference_mode``); the support-side backward is now exact to
   finite-difference precision.
3. ``runtime.py`` — ``run_in_chunks`` no longer forces ``no_grad``. It is the
   feature-attention row-chunking path, which the CUDA planner engages on its own
   under memory pressure; under upstream's unconditional ``no_grad`` a caller who
   enabled gradients got ``x.grad is None`` on some machines and a real gradient
   on others, decided by how full the card was.

Licensing
---------
Code: BSD-3-Clause-LG AI Research License (``LICENSE``). Commercial use permitted.

Weights: EXAONE AI Model License Agreement 1.1 - NC — research use only,
commercial use expressly prohibited, and derivatives must be named ``EXAONE*``.
The registry records this as ``commercial_use_ok=False``.
"""

from .classifier import EXAONETabularClassifier
from .episode_features import EXAONEFeatureEncoder

__all__ = ["EXAONETabularClassifier", "EXAONEFeatureEncoder"]
