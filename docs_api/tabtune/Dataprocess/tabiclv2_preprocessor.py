"""
TabICLv2 Preprocessor for TabTune's DataProcessor.
Same pattern as TabICLPreprocessor (v1) but using the v2 preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

from ..models.tabiclv2.sklearn.preprocessing import (
    TransformToNumerical,
    EnsembleGenerator,
)


class TabICLv2Preprocessor(BaseEstimator, TransformerMixin):
    """
    The full preprocessing pipeline required by TabICLv2:
    1. Transform all features to be numerical (TransformToNumerical).
    2. Encode the target variable.
    
    Note: TabICLv2 handles normalization and outlier removal internally
    via its EnsembleGenerator during fit/predict, so we only do the
    feature-to-numerical transform and label encoding here.
    """

    def __init__(self):
        self.feature_transformer_ = TransformToNumerical()
        self.label_encoder_ = LabelEncoder()
        self.categorical_cols_ = []
        self.numerical_cols_ = []
        self.all_feature_cols_ = []

    def fit(self, X, y):
        logger.info("[TabICLv2Preprocessor] Fitting preprocessor")
        
        if isinstance(X, pd.DataFrame):
            self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
            self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
            self.all_feature_cols_ = X.columns.tolist()
        
        logger.debug("Step 1/2: Fitting feature-to-numerical transformer...")
        self.feature_transformer_.fit_transform(X)
        
        logger.debug("Step 2/2: Fitting label encoder...")
        if y is not None:
            self.label_encoder_.fit(y)
        
        logger.info("[TabICLv2Preprocessor] Preprocessor fitted.")
        return self

    def transform(self, X, y=None):
        logger.debug("Applying TabICLv2 transformations...")
        X_proc = self.feature_transformer_.transform(X)
        
        if y is not None:
            y_final = self.label_encoder_.transform(y)
            logger.debug("Feature and target transformation complete.")
            return X_proc, y_final
        logger.debug("Feature transformation complete.")
        return X_proc

    def get_summary(self):
        cat_cols = self.categorical_cols_
        num_cols = self.numerical_cols_
        all_cols_count = len(self.all_feature_cols_)
        
        summary = {
            "Feature Transformation": {
                "description": "Converted features to a purely numerical format (TabICLv2).",
                "details": [
                    f"Identified and encoded {len(cat_cols)} categorical columns: "
                    f"{', '.join(f'`{c}`' for c in cat_cols)}.",
                    f"Kept {len(num_cols)} numerical columns as-is: "
                    f"{', '.join(f'`{c}`' for c in num_cols)}."
                ]
            },
            "Internal Preprocessing": {
                "description": "TabICLv2 applies normalization and outlier removal internally via EnsembleGenerator.",
                "details": [
                    f"Applied to all {all_cols_count} features during model fit/predict.",
                    "Normalization methods: power, quantile, robust, none (configurable via n_estimators).",
                    "Outlier clipping at 4.0 standard deviations (default)."
                ]
            }
        }
        return summary
