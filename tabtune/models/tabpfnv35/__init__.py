"""TabPFN v3.5 — full integration into TabTune.

Vendored source copy of https://github.com/PriorLabs/TabPFN (v9.0.0), which
ships the v3.5 model (``ModelVersion.V3_5``, architecture ``tabpfn_v3_5``) and
its faster sibling ``v3.5-fast``. Imports were rewritten from the upstream
package root (``tabpfn.xxx``) to this vendored path
(``tabtune.models.tabpfnv35.xxx``); nothing else in the model code was changed.

Why a separate tree rather than a patch to ``tabtune/models/tabpfnv3``:

* v3.5 is a *new architecture*, not a new checkpoint for an existing one. The
  runtime selects the model graph by name (``ARCHITECTURES[architecture_name]``
  in ``model_loading.load_model``), and ``tabpfn_v3_5`` does not exist in the
  v8.0.3 tree TabTune vendors as ``tabpfnv3``.
* From v3.5 on, *one multitask checkpoint carries both the classification and
  the regression head*, whereas every earlier version shipped one file per
  task. The loading code differs accordingly.
* TabTune already keeps one vendored tree per major TabPFN line
  (``tabpfn`` = v2, ``tabpfnv26``, ``tabpfnv3``), so a fourth tree is the house
  pattern and leaves the existing TabPFNv3 integration untouched.

Model names in the pipeline: ``TabPFNv35`` and ``TabPFNv35Fast``.

Two upstream behaviours are deliberately dropped here. The legacy
``sys.modules['tabpfn.model']`` aliases, which upstream installs for
tabpfn-extensions and AutoGluon, are not registered: they squat a global module
name that a co-installed real ``tabpfn`` would also claim. And ``__version__``
is pinned to the vendored release instead of read from installed distribution
metadata, which this package does not have.
"""

from __future__ import annotations

#: The upstream TabPFN release this tree was vendored from.
VENDORED_FROM = "9.0.0"
__version__ = VENDORED_FROM

from .classifier import TabPFNClassifier as TabPFNClassifier
from .constants import ModelVersion
from .errors import TabPFNCUDAOutOfMemoryError, TabPFNMPSOutOfMemoryError
from .misc.debug_versions import display_debug_info
from .model_loading import (
    ModelSource,
    load_fitted_tabpfn_model,
    prepend_cache_path,
    save_fitted_tabpfn_model,
)
from .regressor import TabPFNRegressor as TabPFNRegressor

_BaseClassifier = TabPFNClassifier
_BaseRegressor = TabPFNRegressor


def _v3_5_default_path() -> str:
    """Cache path of the multitask v3.5 checkpoint (both heads in one file)."""
    return prepend_cache_path(ModelSource.get_v3_5().default_filename)


def _v3_5_fast_default_path() -> str:
    return prepend_cache_path(ModelSource.get_v3_5_fast().default_filename)


class TabPFNv35Classifier(_BaseClassifier):
    """TabPFN v3.5 classifier pinned to the v3.5 checkpoint.

    Upstream already defaults ``settings.tabpfn.model_version`` to v3.5; the
    path is pinned explicitly so the model TabTune loads does not change if that
    default moves in a later release.

    Inputs:  sklearn-style X (n_samples, n_features), y (n_samples,).
    Outputs: sklearn-compatible ``predict`` / ``predict_proba``.
    """

    def __init__(self, *args, model_path="auto", **kwargs):
        if model_path == "auto":
            model_path = _v3_5_default_path()
        super().__init__(*args, model_path=model_path, **kwargs)


class TabPFNv35Regressor(_BaseRegressor):
    """TabPFN v3.5 regressor pinned to the same multitask v3.5 checkpoint."""

    def __init__(self, *args, model_path="auto", **kwargs):
        if model_path == "auto":
            model_path = _v3_5_default_path()
        super().__init__(*args, model_path=model_path, **kwargs)


class TabPFNv35FastClassifier(TabPFNv35Classifier):
    """The ``v3.5-fast`` checkpoint: a separate, faster model, not a re-export."""

    def __init__(self, *args, model_path="auto", **kwargs):
        if model_path == "auto":
            model_path = _v3_5_fast_default_path()
        super().__init__(*args, model_path=model_path, **kwargs)


class TabPFNv35FastRegressor(TabPFNv35Regressor):
    """Regression head of the ``v3.5-fast`` multitask checkpoint."""

    def __init__(self, *args, model_path="auto", **kwargs):
        if model_path == "auto":
            model_path = _v3_5_fast_default_path()
        super().__init__(*args, model_path=model_path, **kwargs)


__all__ = [
    "VENDORED_FROM",
    "ModelVersion",
    "ModelSource",
    "TabPFNCUDAOutOfMemoryError",
    "TabPFNClassifier",
    "TabPFNMPSOutOfMemoryError",
    "TabPFNRegressor",
    "TabPFNv35Classifier",
    "TabPFNv35FastClassifier",
    "TabPFNv35FastRegressor",
    "TabPFNv35Regressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "prepend_cache_path",
    "save_fitted_tabpfn_model",
]
