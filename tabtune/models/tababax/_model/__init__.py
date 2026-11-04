"""Private. Users should not import from this subpackage.

``TabAbax`` is wrapped by :class:`tababax.TabAbaxClassifier`
and :class:`tababax.TabAbaxRegressor`.
``InferenceConfig`` is re-exported at :class:`tababax.InferenceConfig`.
"""

from .model import TabAbax
from .inference_config import InferenceConfig
