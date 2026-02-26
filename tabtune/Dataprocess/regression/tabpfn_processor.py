from .base_processor import RegressionDataProcessor

class TabPFNRegressionProcessor(RegressionDataProcessor):
    """
    TabPFN-specific regression data processor.
    
    Note: TabPFN handles target normalization internally (via y_train_mean_ and y_train_std_),
    so target_scaling_strategy should be 'none' by default to avoid double scaling.
    """
    def __init__(self, target_scaling_strategy='none', **kwargs):
        """
        Initialize TabPFNRegressionProcessor.
        
        Args:
            target_scaling_strategy: Strategy for scaling target variable
                ('standard', 'minmax', 'robust', 'power_transform', 'none')
                Default is 'none' because TabPFN handles target normalization internally.
            **kwargs: Additional arguments for future extensions
        """
        # TabPFN normalizes targets internally (see TabPFNRegressor.fit, lines 755-758)
        # So we should not scale the target externally
        super().__init__(target_scaling_strategy=target_scaling_strategy)
