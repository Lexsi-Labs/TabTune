"""TabFM (Google Research) integration for TabTune.

TabTune vendors the full TabFM PyTorch stack under this package -- the real
architecture (``model/model.py``), the Hugging Face loader (``model_loading.py``)
and the scikit-learn inference/preprocessing engine
(``classifier_and_regressor.py``), all ported from google-research/tabfm
(Apache-2.0) -- rather than shelling out to the ``tabfm`` pip package. This mirrors
how ``tabpfnv3`` is vendored and is what makes end-to-end fine-tuning / PEFT on
the real modules possible.

Only the lightweight ``TabFMClassifier`` wrapper and the ``backbone`` helpers are
exported here; the heavy torch modules are imported lazily so ``import tabtune``
never requires the optional TabFM stack.
"""
from .classifier import TabFMClassifier
from .backbone import build_estimator, load_backbone, real_icl_logits

__all__ = [
    "TabFMClassifier",
    "load_backbone",
    "build_estimator",
    "real_icl_logits",
]
