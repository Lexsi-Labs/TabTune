import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

class RegressionDataProcessor(BaseEstimator, TransformerMixin):
    """
    Handles regression-specific data processing, particularly target variable scaling.
    """
    def __init__(self, target_scaling_strategy='standard'):
        self.target_scaling_strategy = target_scaling_strategy
        self.target_scaler_ = None
        self._is_fitted = False

    def fit(self, y):
        """
        Fits the target scaler on the target variable y.
        """
        if self.target_scaling_strategy == 'none':
            self._is_fitted = True
            return self

        # Reshape for sklearn scaler requirements (n_samples, 1)
        y_reshaped = self._reshape_y(y)

        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'power_transform': PowerTransformer()
        }
        
        self.target_scaler_ = scaler_map.get(self.target_scaling_strategy, StandardScaler())
        self.target_scaler_.fit(y_reshaped)
        self._is_fitted = True
        return self

    def transform(self, y):
        """
        Transforms the target variable y using the fitted scaler.
        """
        if not self._is_fitted:
            # If not fitted (e.g. during inference without fit first?), we can't transform.
            # But usually fit is called before transform in pipeline.
            raise RuntimeError("RegressionDataProcessor must be fitted before transform.")

        if self.target_scaling_strategy == 'none' or self.target_scaler_ is None:
            return y

        y_reshaped = self._reshape_y(y)
        y_scaled = self.target_scaler_.transform(y_reshaped)
        
        #Flatten back to 1D array/series if input was series/1d array
        if isinstance(y, pd.Series):
             return pd.Series(y_scaled.flatten(), index=y.index, name=y.name)
        return y_scaled.flatten()

    def inverse_transform(self, y):
        """
        Inverse transforms the target variable y (e.g. for predictions).
        """
        if not self._is_fitted:
             raise RuntimeError("RegressionDataProcessor must be fitted before inverse_transform.")

        if self.target_scaling_strategy == 'none' or self.target_scaler_ is None:
            return y
            
        y_reshaped = self._reshape_y(y)
        y_inv = self.target_scaler_.inverse_transform(y_reshaped)
        
        if isinstance(y, pd.Series):
             return pd.Series(y_inv.flatten(), index=y.index, name=y.name)

        return y_inv.flatten()

    def _reshape_y(self, y):
        if isinstance(y, pd.Series):
            return y.values.reshape(-1, 1)
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                return y.reshape(-1, 1)
        return y
