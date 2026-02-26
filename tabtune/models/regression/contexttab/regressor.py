"""
ConTextTab Regression Model Wrapper for TabTune Pipeline.

This module provides a wrapper around ConTextTabRegressor to ensure
compatibility with the TabTune pipeline and enforce inference-only mode.
"""
import numpy as np
import pandas as pd
import logging
from tabtune.models.contexttab.contexttab import ConTextTabRegressor

logger = logging.getLogger(__name__)


class ConTextTabRegressorWrapper(ConTextTabRegressor):
    """
    Wrapper for ConTextTabRegressor to ensure compatibility with TabTune pipeline.
    
    This wrapper ensures:
    - inference mode is supported, fine-tuning too
    - Proper integration with TabTune's DataProcessor
    - Consistent API with other regression models
    """
    
    def __init__(self, tuning_strategy='inference', **kwargs):
        """
        Initialize ConTextTabRegressorWrapper.
        
        Args:
            tuning_strategy: Must be 'inference' for regression models
            **kwargs: Additional arguments passed to ConTextTabRegressor
        """
        # Validate that only inference mode is used for regression
        if tuning_strategy not in ('inference', 'finetune'):
            raise ValueError(
                f"Unsupported tuning_strategy='{tuning_strategy}' for ContextTab regression. "
                f"Use 'inference' or 'finetune'."
            )
        
        # Filter out pipeline-specific parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['task_type', 'tuning_strategy']
        }
        
        super().__init__(**filtered_kwargs)
        self.tuning_strategy = tuning_strategy
        logger.info(f"[ConTextTabRegressorWrapper] Initialized with inference-only mode")
    
    def fit(self, X, y):
        """
        Fit the regressor on training data.
        
        Args:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training targets (pd.Series or np.ndarray)
        
        Returns:
            self
        """
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
            np.ndarray: Predicted target values (NaN values replaced with mean target)
        """
        predictions = super().predict(X)
        
        # Handle NaN values in predictions
        # This can happen if target_std is 0 or if there are issues with the model output
        if np.any(np.isnan(predictions)):
            logger.warning(
                f"[ConTextTabRegressorWrapper] Found {np.sum(np.isnan(predictions))} NaN predictions. "
                f"Replacing with mean target value: {np.nanmean(self.y_) if hasattr(self, 'y_') else 0.0}"
            )
            # Replace NaN with mean of training target (or 0 if not available)
            nan_mask = np.isnan(predictions)
            replacement_value = np.nanmean(self.y_) if hasattr(self, 'y_') and len(self.y_) > 0 else 0.0
            predictions[nan_mask] = replacement_value
        
        # Also handle inf values
        if np.any(np.isinf(predictions)):
            logger.warning(
                f"[ConTextTabRegressorWrapper] Found {np.sum(np.isinf(predictions))} inf predictions. "
                f"Replacing with mean target value."
            )
            inf_mask = np.isinf(predictions)
            replacement_value = np.nanmean(self.y_) if hasattr(self, 'y_') and len(self.y_) > 0 else 0.0
            predictions[inf_mask] = replacement_value
        
        return predictions
