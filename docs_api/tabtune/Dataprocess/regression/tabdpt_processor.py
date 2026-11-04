"""
TabDPT-specific regression data processor.

Handles regression-specific preprocessing for TabDPT models.
Note: TabDPT handles all preprocessing internally, including target handling,
so this processor should NOT scale the target.
"""
from .base_processor import RegressionDataProcessor


class TabDPTRegressionProcessor(RegressionDataProcessor):
    """
    TabDPT-specific regression data processor.
    
    Handles regression-specific preprocessing for TabDPT models.
    TabDPT handles all preprocessing internally, so target scaling is disabled.
    """
    def __init__(self, target_scaling_strategy='none', **kwargs):
        """
        Initialize TabDPTRegressionProcessor.
        
        Args:
            target_scaling_strategy: Always 'none' for TabDPT (it handles preprocessing internally)
            **kwargs: Additional arguments for future extensions
        """
        # TabDPT handles all preprocessing internally, including target normalization
        # So we should not scale the target here
        super().__init__(target_scaling_strategy='none')
