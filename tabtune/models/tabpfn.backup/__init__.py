from importlib.metadata import version

from .classifier import TabPFNClassifier
from .misc.debug_versions import display_debug_info
from .model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from .regressor import TabPFNRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
