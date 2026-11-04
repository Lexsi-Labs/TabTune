"""
Mitra-specific regression data processor.

Handles regression-specific preprocessing for Mitra (Tab2D) models.
Note: Mitra handles target normalization internally (via Preprocessor.normalize_y),
so target_scaling_strategy should be 'none' by default.
"""
from .base_processor import RegressionDataProcessor


class MitraRegressionProcessor(RegressionDataProcessor):
    """
    Mitra-specific regression data processor.
    
    Handles regression-specific preprocessing for Mitra (Tab2D) models.
    Mitra normalizes targets internally using min-max scaling (see AutoGluon's Preprocessor.normalize_y),
    so target_scaling_strategy defaults to 'none' to avoid double scaling.
    """
    def __init__(self, target_scaling_strategy='none', **kwargs):
        """
        Initialize MitraRegressionProcessor.
        
        Args:
            target_scaling_strategy: Strategy for scaling target variable
                ('standard', 'minmax', 'robust', 'power_transform', 'none')
                Default is 'none' because Mitra handles target normalization internally.
            **kwargs: Additional arguments for future extensions
        """
        # Mitra normalizes targets internally (min-max scaling via Preprocessor.normalize_y)
        # So we should not scale the target externally
        super().__init__(target_scaling_strategy=target_scaling_strategy)
