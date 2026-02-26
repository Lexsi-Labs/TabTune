import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import logging
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

logger = logging.getLogger(__name__)

def _fix_dtypes(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Standardizes input to a pandas DataFrame with corrected types."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    X = X.convert_dtypes()
    numeric_cols = X.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].astype("float64")
    return X

def _get_ordinal_encoder() -> ColumnTransformer:
    """Creates the specific OrdinalEncoder configuration used by TabPFN."""
    oe = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    to_convert = ["category", "string", "object"]
    return ColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
        remainder="passthrough", 
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

class TabPFNPreprocessor(BaseEstimator, TransformerMixin):
    """
    The initial preprocessing steps (dtype fixing and ordinal encoding)
    required by the TabPFN model before scaling.
    """
    def __init__(self):
        self.encoder_ = None
        self.original_columns_ = None
        self._is_fitted = False

    def fit(self, X, y=None):
        logger.info("Fitting TabPFN Preprocessor")
        X_fixed = _fix_dtypes(X)
        self.original_columns_ = X_fixed.columns.tolist()
        self.encoder_ = _get_ordinal_encoder()
        self.encoder_.fit(X_fixed)
        self._is_fitted = True
        logger.info(" TabPFN Preprocessor fitted successfully.")
        return self

    def transform(self, X, y=None):
        if not self._is_fitted:
            raise logger.error("You must call fit() before calling transform().")

        X_fixed = _fix_dtypes(X)
        
        # Ensure columns are in the same order as during fit
        X_fixed = X_fixed.reindex(columns=self.original_columns_, fill_value=0)

        X_transformed = self.encoder_.transform(X_fixed)
        
        # Return a pandas DataFrame to preserve column names for the next steps
        X_final = pd.DataFrame(X_transformed, index=X.index, columns=self.original_columns_)
        
        if y is not None:
            return X_final, y
        return X_final
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    # --- NEW METHOD ---
    def get_summary(self):
        """
        Returns a rich dictionary with column-level details for each processing step.
        """
        if not self._is_fitted:
            return {"Error": "Preprocessor has not been fitted yet."}

        # Introspect the ColumnTransformer to find which columns were processed
        try:
            encoded_cols = self.encoder_.transformers_[0][2]
            passthrough_cols = [col for col in self.original_columns_ if col not in encoded_cols]
        except (AttributeError, IndexError):
            encoded_cols = "N/A"
            passthrough_cols = "N/A"
            
        summary = {
            "Data Type Standardization": {
                "description": "Ensured input is a pandas DataFrame and converted numerical columns to a consistent `float64` type.",
                "details": [
                    "This initial step guarantees a uniform data format for subsequent processing."
                ]
            },
            "Feature Encoding": {
                "description": "Applied ordinal encoding to non-numerical features and passed numerical features through without modification.",
                "details": [
                    f"The OrdinalEncoder was fitted on {len(encoded_cols)} non-numerical columns: {encoded_cols}",
                    f"{len(passthrough_cols)} numerical columns were passed through untouched: {passthrough_cols}"
                ]
            }
        }
        
        return summary