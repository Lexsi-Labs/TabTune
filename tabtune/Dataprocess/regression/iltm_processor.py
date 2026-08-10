"""
iLTM-specific regression data processor.

iLTM normalises the regression target internally (the vendored engine
standardises ``y`` before the hypernetwork forward and de-standardises its
predictions), so no TabTune-side target scaling is applied here (mirrors the
TabFM / XRFM regression processors).
"""
from .base_processor import RegressionDataProcessor


class ILTMRegressionProcessor(RegressionDataProcessor):
    """iLTM regression processor -- target scaling disabled (handled internally)."""

    def __init__(self, target_scaling_strategy: str = "none", **kwargs):
        # iLTM always normalises the target internally, so target scaling is
        # forced to 'none' regardless of the argument (kept for signature
        # parity with the other regression processors).
        super().__init__(target_scaling_strategy="none")
