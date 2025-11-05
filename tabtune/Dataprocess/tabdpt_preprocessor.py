import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder
# Removed torch imports - no longer needed for projection layer
# import torch
# import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

class TabDPTPreprocessor(BaseEstimator, TransformerMixin):
    """
    Minimal preprocessor for TabDPT to handle basic data format conversions.
    1. Converts categorical features to numerical format (OrdinalEncoder)
    2. Encodes the target variable (LabelEncoder)
    3. Ensures pandas DataFrames are converted to numpy arrays
    
    The standalone TabDPT model handles all advanced preprocessing internally
    (normalization, missing indicators, outlier clipping, feature reduction, etc.)
    """
    def __init__(self):
        # Removed: model_input_dim parameter - not needed anymore
        self.feature_encoder_ = None
        self.label_encoder_ = None
        # Removed: self.projector_ - standalone TabDPT handles feature projection
        # Removed: self.fitted_input_features_ - not needed
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting TabDPT Preprocessor...")

        # 1. Fit the standard feature encoder (categorical -> numerical)
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

        # 2. REMOVED: Projection layer initialization - standalone TabDPT handles this internally
        # self.fitted_input_features_ = self.feature_encoder_.transform(X.iloc[0:1]).shape[1]
        # 
        # # REMOVED: Projection layer - standalone TabDPT handles this internally
        # print(f"  - Data has {self.fitted_input_features_} features after encoding.")
        # print(f"  - Initializing a projection layer to map features to {self.model_input_dim} dimensions.")
        # self.projector_ = nn.Linear(self.fitted_input_features_, self.model_input_dim)
        # self.projector_.train() # Set to training mode for the finetuning loop

        # 3. Fit the label encoder for the target variable
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        self._is_fitted = True
        
        logger.info(" TabDPT Preprocessor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        if self.feature_encoder_ is None:
            raise logger.error("You must fit the preprocessor before transforming data.")
            
        # Apply categorical encoding
        X_transformed = self.feature_encoder_.transform(X)
        X_processed = np.nan_to_num(X_transformed.astype(np.float32))
        
        # REMOVED: Projection layer application - standalone TabDPT handles feature projection
        # X_tensor = torch.from_numpy(X_processed).float()
        # 
        # # The fine-tuning loop will handle gradient calculations.
        # # For standard transform calls, we use torch.no_grad().
        # with torch.no_grad():
        #      X_projected_tensor = self.projector_(X_tensor)
        # 
        # X_final = X_projected_tensor.detach().numpy()
        
        if y is not None:
            if self.label_encoder_ is None:
                raise logger.error("Label encoder not fitted.")
            y_final = self.label_encoder_.transform(y)
            return X_processed, y_final
            
        return X_processed


    def get_summary(self):
        """
        Returns a rich dictionary with column-level details for each processing step.
        """
        if not self._is_fitted:
            return {"Error": "Preprocessor has not been fitted yet."}
            
        # Get the names of columns that were ordinally encoded
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
            # REMOVED: "Feature Projection" section - not applicable anymore
            "Target Encoding": {
                "description": "Encoded the target variable into numerical labels.",
                "details": [
                    f"Fitted LabelEncoder on target, identifying {len(self.label_encoder_.classes_)} unique classes."
                ]
            }
        }
        
        return summary