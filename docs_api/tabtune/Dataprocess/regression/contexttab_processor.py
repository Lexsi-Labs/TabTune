"""
ContextTab-specific regression data processor.

Handles regression-specific preprocessing for ContextTab models,
including target scaling and regression type configuration.

Note: ContextTab handles target normalization internally (via standard_scale_column in the tokenizer),
so target_scaling_strategy should be 'none' by default.
"""
from .base_processor import RegressionDataProcessor


class ContextTabRegressionProcessor(RegressionDataProcessor):
    """
    ContextTab-specific regression data processor.
    
    Handles regression-specific preprocessing for ContextTab models.
    ContextTab supports two regression types:
    - 'l2': Direct L2 loss regression (uses standard_scale_column internally)
    - 'reg-as-classif': Binned regression (quantile-based classification)
    
    Note: ContextTab handles target normalization internally for l2 regression,
    so target_scaling_strategy defaults to 'none'.
    """
    def __init__(self, target_scaling_strategy='none', regression_type='l2', num_regression_bins=16):
        """
        Initialize ContextTabRegressionProcessor.
        
        Args:
            target_scaling_strategy: Strategy for scaling target variable
                ('standard', 'minmax', 'robust', 'power_transform', 'none')
                Default is 'none' because ContextTab handles target normalization internally.
            regression_type: Type of regression ('l2' or 'reg-as-classif')
            num_regression_bins: Number of bins for reg-as-classif mode
        """
        # ContextTab handles target normalization internally via standard_scale_column
        # So we should not scale the target externally
        super().__init__(target_scaling_strategy=target_scaling_strategy)
        self.regression_type = regression_type
        self.num_regression_bins = num_regression_bins
