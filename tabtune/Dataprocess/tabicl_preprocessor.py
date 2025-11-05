import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np

logger = logging.getLogger(__name__)

from ..models.tabicl.sklearn.preprocessing import (
    TransformToNumerical, 
    PreprocessingPipeline, 
    OutlierRemover
)

class TabICLPreprocessor(BaseEstimator, TransformerMixin):
    """
    The full, four-step preprocessing pipeline required by TabICL:
    1. Transform all features to be numerical.
    2. Normalize features using a power transform.
    3. Remove outliers.
    4. Encode the target variable.
    """
    def __init__(self):
        self.feature_transformer_ = TransformToNumerical()
        self.normalizer_ = PreprocessingPipeline(normalization_method="power")
        self.outlier_remover_ = OutlierRemover(threshold=4.0)
        self.label_encoder_ = LabelEncoder()
        self.all_numerical_cols_ = []


        self.categorical_cols_ = []
        self.numerical_cols_ = []
        self.all_feature_cols_ = []

    def fit(self, X, y):
        logger.info("Fitting TabICL Preprocessor")
        
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        
        
        logger.debug("Step 1/4: Fitting feature-to-numerical transformer...")
        X_proc = self.feature_transformer_.fit_transform(X)
        self.all_feature_cols_ = X.columns.tolist()
        
        
        # The fit calls will now work correctly
        
        logger.debug("Step 2/4: Fitting power normalizer...")
        self.normalizer_.fit(X_proc)
        X_proc_norm = self.normalizer_.transform(X_proc)
        logger.debug("Step 3/4: Fitting outlier remover...")
        self.outlier_remover_.fit(X_proc_norm)
        logger.debug("Step 4/4: Fitting label encoder...")
        self.label_encoder_.fit(y)
        logger.info(" TabICL Preprocessor fitted.")
        return self

    def transform(self, X, y=None):
        logger.debug("Applying TabICL transformations...")
        X_proc = self.feature_transformer_.transform(X)
        X_proc_norm = self.normalizer_.transform(X_proc)
        X_final = self.outlier_remover_.transform(X_proc_norm)
        
        if y is not None:
            y_final = self.label_encoder_.transform(y)
            logger.debug("Feature and target transformation complete.")
            return X_final, y_final
        logger.debug("Feature transformation complete.")
        return X_final


    def get_summary(self):
        """
        Returns a rich dictionary with column-level details for each processing step.
        """
        # Access attributes that were learned during the .fit() call
        cat_cols = self.categorical_cols_
        num_cols = self.numerical_cols_
        all_cols_count = len(self.all_feature_cols_)
        
        summary = {
            "Feature Transformation": {
                "description": "Converted features to a purely numerical format.",
                "details": [
                    f"Identified and OrdinalEncoded {len(cat_cols)} categorical columns: \n {', '.join(f'`{c}`' for c in cat_cols)}.",
                    f"Kept {len(num_cols)} numerical columns as-is \n: {', '.join(f'`{c}`' for c in num_cols)}."
                ]
            },
            "Normalization": {
                "description": "Applied a power transform to normalize feature distributions.",
                ## --- REVERTED TO CONCISE SUMMARY ---
                "details": [
                    f"This step was applied to all {all_cols_count} features after the initial transformation."
                ]
            },
            "Outlier Handling": {
                "description": f"Clipped feature values exceeding a threshold of {self.outlier_remover_.threshold} standard deviations.",
                ## --- REVERTED TO CONCISE SUMMARY ---
                "details": [
                    f"This step was applied to all {all_cols_count} normalized features."
                ]
            }
        }
        
        return summary