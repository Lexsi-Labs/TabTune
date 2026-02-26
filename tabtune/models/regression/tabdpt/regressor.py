"""
TabDPT Regression Model Wrapper for TabTune Pipeline.

This module provides a wrapper around TabDPTRegressor to ensure
compatibility with the TabTune pipeline and enforce inference-only mode.
"""
import numpy as np
import pandas as pd
import logging
from tabtune.models.tabdpt.regressor import TabDPTRegressor

logger = logging.getLogger(__name__)


class TabDPTRegressorWrapper(TabDPTRegressor):
    """
    Wrapper for TabDPTRegressor to ensure compatibility with TabTune pipeline.
    
    This wrapper ensures:
    - Only inference mode is supported (no fine-tuning)
    - Proper integration with TabTune's DataProcessor
    - Consistent API with other regression models
    """
    
    def __init__(self, tuning_strategy='inference', **kwargs):
        """
        Initialize TabDPTRegressorWrapper.
        
        Args:
            tuning_strategy: Must be 'inference' for regression models
            **kwargs: Additional arguments passed to TabDPTRegressor
        """
        # Validate that only inference mode is used for regression
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError("TabDPT regression supports tuning_strategy in {'inference','finetune'}")

        
        # Filter out pipeline-specific parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['task_type', 'tuning_strategy']
        }
        
        super().__init__(**filtered_kwargs)
        self.tuning_strategy = tuning_strategy
        logger.info(f"[TabDPTRegressorWrapper] Initialized with inference-only mode")
    
    def fit(self, X, y):
        """
        Fit the regressor on training data.
        
        Args:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training targets (pd.Series or np.ndarray)
        
        Returns:
            self
        """
        # Convert X to numpy array if needed
        # TabDPT expects numeric numpy arrays
        if isinstance(X, pd.DataFrame):
            # Check for non-numeric columns and convert them
            X = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # Convert categorical/string columns to numeric codes
                    X[col] = pd.Categorical(X[col]).codes
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Handle object/string arrays - convert to numeric
        if X.dtype == object:
            # Try to convert object arrays to float, handling errors
            try:
                X = X.astype(float)
            except (ValueError, TypeError):
                # If direct conversion fails, use pandas to handle mixed types
                X_df = pd.DataFrame(X)
                for col in X_df.columns:
                    if X_df[col].dtype == 'object':
                        X_df[col] = pd.Categorical(X_df[col]).codes
                X = X_df.values.astype(float)
        
        # Ensure X is float32 (TabDPT expects this)
        if X.dtype != np.float32:
            try:
                X = X.astype(np.float32)
            except (ValueError, TypeError) as e:
                logger.error(f"[TabDPTRegressorWrapper] Failed to convert X to float32: {e}")
                raise ValueError(
                    f"Cannot convert input data to float32. "
                    f"Data may contain non-numeric values. Original error: {e}"
                )
        
        # Ensure y is numeric for regression
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
        
        y = np.array(y).flatten().astype(float)
        return super().fit(X, y)
    
    def predict(self, X, n_ensembles: int = 8, context_size: int | None = 2048, seed: int | None = None):
        """
        Predict target values for input data.
        
        Args:
            X: Input features (pd.DataFrame or np.ndarray)
            n_ensembles: Number of ensemble members (default: 8)
            context_size: Context size for retrieval (default: 2048, None for full context)
            seed: Random seed for reproducibility
        
        Returns:
            np.ndarray: Predicted target values
        """
        # Convert X to numpy array if needed
        if isinstance(X, pd.DataFrame):
            # Check for non-numeric columns and convert them
            X = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # Convert categorical/string columns to numeric codes
                    X[col] = pd.Categorical(X[col]).codes
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Handle object/string arrays - convert to numeric
        if X.dtype == object:
            # Try to convert object arrays to float, handling errors
            try:
                X = X.astype(float)
            except (ValueError, TypeError):
                # If direct conversion fails, use pandas to handle mixed types
                X_df = pd.DataFrame(X)
                for col in X_df.columns:
                    if X_df[col].dtype == 'object':
                        X_df[col] = pd.Categorical(X_df[col]).codes
                X = X_df.values.astype(float)
        
        # Ensure X is float32 (TabDPT expects this)
        if X.dtype != np.float32:
            try:
                X = X.astype(np.float32)
            except (ValueError, TypeError) as e:
                logger.error(f"[TabDPTRegressorWrapper] Failed to convert X to float32: {e}")
                raise ValueError(
                    f"Cannot convert input data to float32. "
                    f"Data may contain non-numeric values. Original error: {e}"
                )
        
        return super().predict(X, n_ensembles=n_ensembles, context_size=context_size, seed=seed)
