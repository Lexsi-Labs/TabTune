"""
LimiX-specific regression data processor.

Handles regression-specific preprocessing for LimiX models.
Note: LimiX handles target normalization internally (via y_mean_ and y_std_),
so target_scaling_strategy should be 'none' by default.
"""
from .base_processor import RegressionDataProcessor


class LimixRegressionProcessor(RegressionDataProcessor):
    """
    LimiX-specific regression data processor.
    
    Handles regression-specific preprocessing for LimiX models.
    LimiX normalizes targets internally (see LimixRegressor.fit, lines 247-252),
    so target_scaling_strategy defaults to 'none' to avoid double scaling.
    """
    def __init__(self, target_scaling_strategy='none', **kwargs):
        """
        Initialize LimixRegressionProcessor.
        
        Args:
            target_scaling_strategy: Strategy for scaling target variable
                ('standard', 'minmax', 'robust', 'power_transform', 'none')
                Default is 'none' because LimiX handles target normalization internally.
            **kwargs: Additional arguments for future extensions
        """
        # LimiX normalizes targets internally (stores y_mean_ and y_std_, then normalizes)
        # So we should not scale the target externally
        super().__init__(target_scaling_strategy=target_scaling_strategy)
