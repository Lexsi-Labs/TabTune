"""
TabPFN v2.6 — Full integration into TabTune
=============================================
Complete source copy from https://github.com/PriorLabs/TabPFN
All imports rewritten from absolute (tabpfn.xxx) to relative (.xxx).
External dependency tabpfn_common_utils replaced with internal no-op shim.

Model name in pipeline: 'TabPFNv26'
Supports: classification, regression, finetune, peft, calibration, fairness
"""

from .classifier import TabPFNClassifier as TabPFNv26Classifier
from .regressor import TabPFNRegressor as TabPFNv26Regressor
from .model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model

__all__ = [
    "TabPFNv26Classifier",
    "TabPFNv26Regressor",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
