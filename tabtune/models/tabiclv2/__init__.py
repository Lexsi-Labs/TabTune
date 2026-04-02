"""
TabICLv2 - Tabular In-Context Learning v2
==========================================
Full integration into TabTune with all model files.
Based on: https://github.com/soda-inria/tabicl (v2)
"""

from .model.tabicl import TabICL
from .model.kv_cache import TabICLCache
from .model.inference_config import InferenceConfig

from .sklearn.classifier import TabICLClassifier as TabICLv2Classifier
from .sklearn.regressor import TabICLRegressor as TabICLv2Regressor
