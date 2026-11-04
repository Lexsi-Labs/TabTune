import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder
import logging
logger = logging.getLogger(__name__)

class TabDPTPreprocessor(BaseEstimator, TransformerMixin):
    """
    Minimal preprocessor for TabDPT to handle basic data format conversions.
    1. Converts categorical features to numerical format (OrdinalEncoder)
    2. Encodes the target variable (LabelEncoder) - only for classification
    3. Ensures pandas DataFrames are converted to numpy arrays
    
    The standalone TabDPT model handles all advanced preprocessing internally
    (normalization, missing indicators, outlier clipping, feature reduction, etc.)
    """
    def __init__(self, task_type='classification'):
        self.feature_encoder_ = None
        self.label_encoder_ = None
        self.task_type = task_type
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting TabDPT Preprocessor...")

        to_convert = ["category", "string", "object", "boolean"]
        self.feature_encoder_ = ColumnTransformer(
            transformers=[
                ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                 make_column_selector(dtype_include=to_convert))
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )
        self.feature_encoder_.fit(X)
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
        else:
            # For regression, target should remain numeric
            self.label_encoder_ = None

        self._is_fitted = True
        
        logger.info(" TabDPT Preprocessor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        if self.feature_encoder_ is None:
            raise logger.error("You must fit the preprocessor before transforming data.")
        X_transformed = self.feature_encoder_.transform(X)
        X_processed = np.nan_to_num(X_transformed.astype(np.float32))
        
        if y is not None:
            if self.task_type == 'classification':
                if self.label_encoder_ is None:
                    raise logger.error("Label encoder not fitted.")
                y_final = self.label_encoder_.transform(y)
            else:
                y_final = np.array(y).flatten().astype(float) if not isinstance(y, np.ndarray) else y.astype(float)
            return X_processed, y_final
            
        return X_processed


    def get_summary(self):
        """
        Returns a rich dictionary with column-level details for each processing step.
        """
        if not self._is_fitted:
            return {"Error": "Preprocessor has not been fitted yet."}
        try:
            encoded_cols = self.feature_encoder_.transformers_[0][2]
        except (AttributeError, IndexError):
            encoded_cols = "N/A"

        summary = {
            "Basic Data Conversion": {
                "description": "Converts pandas DataFrames to numpy arrays and handles categorical encoding.",
                "details": [
                    f"Applied OrdinalEncoder to {len(encoded_cols)} categorical columns.",
                    "Converted data to float32 numpy arrays for TabDPT compatibility.",
                    "The standalone TabDPT model handles all advanced preprocessing internally."
                ]
            },
            "Target Encoding": {
                "description": "Encoded the target variable into numerical labels." if self.task_type == 'classification' else "Target variable remains numeric (regression).",
                "details": [
                    f"Fitted LabelEncoder on target, identifying {len(self.label_encoder_.classes_)} unique classes."
                ] if self.task_type == 'classification' and self.label_encoder_ else [
                    "Target variable kept as numeric values for regression."
                ]
            }
        }
        
        return summary