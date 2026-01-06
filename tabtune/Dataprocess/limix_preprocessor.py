import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class LimixPreprocessor(BaseEstimator, TransformerMixin):
    """
    Minimal preprocessor for Limix. 
    Limix handles normalization and NaN encoding internally within its architecture.
    This processor converts categorical strings to integers (Ordinal Encoding).
    """
    def __init__(self):
        self.column_transformer_ = None
        self.label_encoder_ = None
        self.categorical_cols_ = []
        self.numerical_cols_ = []

    def fit(self, X, y=None):
        logger.info("Fitting Limix Preprocessor...")
        
        # Identify columns
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        transformers = []
        
        # Ordinal Encode Categoricals (handle unknown by setting to -1 or similar, Limix handles inputs)
        if self.categorical_cols_:
            transformers.append(
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), self.categorical_cols_)
            )
        
        # Pass numericals through
        if self.numerical_cols_:
            transformers.append(('num', 'passthrough', self.numerical_cols_))

        self.column_transformer_ = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        self.column_transformer_.fit(X)

        if y is not None:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)

        return self

    def transform(self, X, y=None):
        if self.column_transformer_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
            
        X_out = self.column_transformer_.transform(X)
        
        # Ensure float32 for PyTorch compatibility
        X_out = X_out.astype(np.float32)

        if y is not None:
            y_out = self.label_encoder_.transform(y)
            return X_out, y_out
            
        return X_out

    def get_summary(self):
        return {
            "strategy": "Limix Specific",
            "categorical": len(self.categorical_cols_),
            "numerical": len(self.numerical_cols_)
        }