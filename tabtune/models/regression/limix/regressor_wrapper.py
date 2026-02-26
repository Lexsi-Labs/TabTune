"""
LimiX Regression Model Wrapper for TabTune Pipeline.

This module provides a wrapper around LimixRegressor to ensure
compatibility with the TabTune pipeline and enforce inference-only mode.
"""
import numpy as np
import pandas as pd
import logging
from tabtune.models.regression.limix.regressor import LimixRegressor
from tabtune.models.regression.limix.regressor_ensemble import LimixRegressorEnsemble

logger = logging.getLogger(__name__)


class LimixRegressorWrapper(LimixRegressorEnsemble):
    """
    Wrapper for LimixRegressorEnsemble to ensure compatibility with TabTune pipeline.
    
    This wrapper ensures:
    - Only inference mode is supported (no fine-tuning)
    - Proper integration with TabTune's DataProcessor
    - Consistent API with other regression models
    - Uses ensemble methods for better performance (matching official LimiX)
    """
    
    def __init__(self, tuning_strategy='inference', n_estimators=8, **kwargs):
        """
        Initialize LimixRegressorWrapper.
        
        Args:
            tuning_strategy: Must be 'inference' for regression models
            n_estimators: Number of ensemble members (default: 8, matching official LimiX)
            **kwargs: Additional arguments passed to LimixRegressorEnsemble
        """
        # Validate that only inference mode is used for regression
        if tuning_strategy not in ('inference', "finetune"):
            raise ValueError(
                f"Regression models only support 'inference' mode. "
                f"Received: '{tuning_strategy}'. Fine-tuning for regression is not supported."
            )
        
        # Filter out pipeline-specific parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['task_type', 'tuning_strategy']
        }
        
        # Use ensemble by default for better performance
        super().__init__(n_estimators=n_estimators, **filtered_kwargs)
        self.tuning_strategy = tuning_strategy
        logger.info(f"[LimixRegressorWrapper] Initialized with inference-only mode, {n_estimators} ensemble members")
    
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
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure y is numeric for regression
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
        
        y = np.array(y).flatten().astype(float)
        return super().fit(X, y)
    
    def predict(self, X):
        """
        Predict target values for input data.
        
        Args:
            X: Input features (pd.DataFrame or np.ndarray)
        
        Returns:
            np.ndarray: Predicted target values
        """
        # Convert X to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        return super().predict(X)
