import numpy as np
import pandas as pd
import torch
import logging
from scipy.sparse import issparse
from tabtune.models.tabpfn.regressor import TabPFNRegressor

logger = logging.getLogger(__name__)

class TabPFNRegressorWrapper(TabPFNRegressor):
    """
    Wrapper for TabPFNRegressor to ensure compatibility with TabTune pipeline.
    
    TabPFNRegressor handles all preprocessing internally:
    - Feature preprocessing: dtype fixing, categorical inference, ordinal encoding
    - Target normalization: y = (y - mean) / std
    - Predictions: Already denormalized via raw_space_bardist_
    """
    def __init__(self, tuning_strategy='inference', **kwargs):
        """
        Initialize TabPFNRegressorWrapper.
        
        Args:
            tuning_strategy: Must be 'inference' for regression (default: 'inference')
            **kwargs: Additional arguments passed to TabPFNRegressor
        """
        # Validate that only inference mode is used for regression
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError(
                f"Regression models only support 'inference' mode. "
                f"Received: '{tuning_strategy}'. Fine-tuning for regression is not supported."
            )
        
        # Filter out keys that TabPFNRegressor might not accept if passed from pipeline
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['task_type', 'tuning_strategy']}
        super().__init__(**filtered_kwargs)
        self.tuning_strategy = tuning_strategy
        self.model = self  # Alias for pipeline compatibility
                           
    def fit(self, X, y):
        # Convert sparse matrices to dense (TabPFN requires dense data)
        if issparse(X):
            logger.warning("[TabPFNRegressorWrapper] Converting sparse matrix to dense array")
            X = X.toarray()
        elif isinstance(X, pd.DataFrame):
            # Check if any columns are sparse
            for col in X.columns:
                if hasattr(X[col], 'sparse') and X[col].sparse:
                    X = X.copy()
                    X[col] = X[col].sparse.to_dense()
        
        # Ensure y is 1D array for regression
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
            
        y = np.array(y).flatten()
        return super().fit(X, y)
        
    def predict(self, X, output_type="mean", quantiles=None):
        """
        Predict target values or quantiles.
        
        Args:
            X: Input features
            output_type: Type of output ('mean', 'median', 'mode', 'quantiles', 'main', 'full')
            quantiles: List of quantiles if output_type='quantiles'
        
        Returns:
            Predictions or quantile predictions based on output_type
        """
        # Convert sparse matrices to dense (TabPFN requires dense data)
        if issparse(X):
            logger.warning("[TabPFNRegressorWrapper] Converting sparse matrix to dense array")
            X = X.toarray()
        elif isinstance(X, pd.DataFrame):
            # Check if any columns are sparse
            for col in X.columns:
                if hasattr(X[col], 'sparse') and X[col].sparse:
                    X = X.copy()
                    X[col] = X[col].sparse.to_dense()
        
        return super().predict(X, output_type=output_type, quantiles=quantiles)
