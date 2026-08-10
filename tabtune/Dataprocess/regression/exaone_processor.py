"""
EXAONE Tabular-specific regression data processor.

The vendored EXAONE regression engine centres and scales the target itself
(support mean / support std, ``_upstream_regressor``) and un-scales the median
quantile before returning, so ``EXAONETabularRegressorWrapper.predict`` already
speaks the **original target space**.

That makes TabTune-side target scaling not merely redundant but wrong: the
pipeline would scale ``y`` before ``fit`` and then never invert the model's
output, so every prediction would come back in the standardised range while the
caller compared it against raw targets. Target scaling is therefore forced to
``'none'`` here, mirroring the TabFM / iLTM / Mitra regression processors.
"""
from .base_processor import RegressionDataProcessor


class EXAONERegressionProcessor(RegressionDataProcessor):
    """EXAONE regression processor -- target scaling disabled (handled internally)."""

    def __init__(self, target_scaling_strategy: str = "none", **kwargs):
        # EXAONE always centres/scales the target internally and returns the
        # original space, so target scaling is forced to 'none' regardless of the
        # argument (kept for signature parity with the other regression
        # processors).
        super().__init__(target_scaling_strategy="none")
