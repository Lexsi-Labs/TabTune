"""
xRFM-specific regression data processor.

The xRFM wrapper standardises the target internally (``y_mean_`` / ``y_std_``),
so no target scaling is applied here (mirrors the TabFM / Limix regression
processors).
"""
from .base_processor import RegressionDataProcessor


class XRFMRegressionProcessor(RegressionDataProcessor):
    """xRFM regression processor -- target scaling disabled (handled internally)."""

    def __init__(self, target_scaling_strategy: str = "none", **kwargs):
        # The xRFM wrapper always standardises the target internally, so target
        # scaling is forced to 'none' regardless of the argument (kept for
        # signature parity with the other regression processors).
        super().__init__(target_scaling_strategy="none")
