"""
TabFM-specific regression data processor.

TabFM performs its own feature encoding and target handling internally, so no
target scaling is applied here (mirrors the TabDPT / Mitra regression processors).
"""
from .base_processor import RegressionDataProcessor


class TabFMRegressionProcessor(RegressionDataProcessor):
    """TabFM regression processor -- target scaling disabled (handled internally)."""

    def __init__(self, target_scaling_strategy: str = "none", **kwargs):
        # TabFM always normalises the target internally, so target scaling is
        # forced to 'none' regardless of the argument (kept for signature parity
        # with the other regression processors).
        super().__init__(target_scaling_strategy="none")
