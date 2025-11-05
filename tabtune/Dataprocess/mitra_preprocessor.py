import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest

import logging
logger = logging.getLogger(__name__)


class MitraPreprocessor(BaseEstimator, TransformerMixin):
    """
    Encapsulates a robust and augmented preprocessing pipeline for the Mitra (Tab2D) model.
    
    This preprocessor handles:
    1.  Identification of numerical and categorical features.
    2.  Removal of zero-variance (constant) features.
    3.  Optional selection of the top 'k' most informative features.
    4.  Imputation of missing numerical values.
    5.  Scaling of numerical features using either StandardScaler or a robust QuantileTransformer.
    6.  Ordinal encoding of categorical features.
    7.  Label encoding of the target variable.
    8.  Optional training-time augmentations like feature shuffling and mirroring.
    """
    def __init__(self, 
                 scaler: str = 'standard', 
                 max_features: int | None = None,
                 shuffle_features: bool = False,
                 random_mirror_x: bool = False):
        """
        Args:
            scaler (str): The type of scaler to use for numerical features. 
                          Options: 'standard' (default) or 'quantile'.
            max_features (int | None): The maximum number of features to select using SelectKBest. 
                                       If None (default), all features are kept.
            shuffle_features (bool): If True, randomly shuffles the column order during transform.
                                     Should be False for inference. Defaults to False.
            random_mirror_x (bool): If True, randomly multiplies numerical features by -1.
                                    Should be False for inference. Defaults to False.
        """
        if scaler not in ['standard', 'quantile']:
            raise ValueError("Scaler must be 'standard' or 'quantile'")
        
        self.scaler_type_ = scaler
        self.max_features_ = max_features
        self.shuffle_features_ = shuffle_features
        self.random_mirror_x_ = random_mirror_x

        # --- Initialize all attributes to prevent AttributeErrors ---
        self._is_fitted = False
        self.original_numerical_cols_ = []
        self.original_categorical_cols_ = []
        self.final_numerical_cols_ = []
        self.final_categorical_cols_ = []
        self.final_features_ = []
        self.feature_order_ = None

        self.imputer_ = None
        self.variance_selector_ = None
        self.feature_selector_ = None
        self.scaler_ = None
        self.categorical_encoder_ = None
        self.label_encoder_ = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the entire preprocessing pipeline."""
        logger.info("Fitting Mitra Preprocessor...")

        # --- Step 1: Identify column types ---
        self.original_numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.original_categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
        
        logger.info(f"  Numerical columns ({len(self.original_numerical_cols_)}): {self.original_numerical_cols_[:10]}...")
        logger.info(f"  Categorical columns ({len(self.original_categorical_cols_)}): {self.original_categorical_cols_}")

        # --- Step 2: Process numerical columns ---
        if self.original_numerical_cols_:
            X_num = X[self.original_numerical_cols_]
            
            # --- CRITICAL FIX: Remove zero-variance columns FIRST (before imputation) ---
            # This ensures imputer only sees the columns we'll actually use
            self.variance_selector_ = VarianceThreshold(threshold=0.0)
            self.variance_selector_.fit(X_num)
            non_constant_mask = self.variance_selector_.get_support()
            non_constant_cols = [col for col, keep in zip(self.original_numerical_cols_, non_constant_mask) if keep]
            
            logger.info(f"  After variance removal: {len(non_constant_cols)} numerical columns")
            
            # --- Feature selection (k-best) ---
            if self.max_features_ and len(non_constant_cols) > self.max_features_:
                self.feature_selector_ = SelectKBest(k=self.max_features_)
                X_temp = X[non_constant_cols]
                self.feature_selector_.fit(X_temp, y)
                selected_mask = self.feature_selector_.get_support()
                self.final_numerical_cols_ = [col for col, keep in zip(non_constant_cols, selected_mask) if keep]
                logger.info(f"  After k-best selection: {len(self.final_numerical_cols_)} numerical columns")
            else:
                self.final_numerical_cols_ = non_constant_cols
                logger.info(f"  Keeping all {len(self.final_numerical_cols_)} numerical columns")

            # --- CRITICAL FIX: Fit imputer ONLY on final numerical columns ---
            # This ensures imputer knows exactly which columns to expect during transform
            self.imputer_ = SimpleImputer(strategy='mean')
            X_final_num = X[self.final_numerical_cols_]
            self.imputer_.fit(X_final_num)
            logger.info(f"  Imputer fitted on {len(self.final_numerical_cols_)} columns")

            # --- Scaling ---
            if self.scaler_type_ == 'standard':
                self.scaler_ = StandardScaler()
            else:
                self.scaler_ = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(X) // 10, 1000), 10))
            
            # Fit scaler on imputed data
            X_imputed = self.imputer_.transform(X_final_num)
            self.scaler_.fit(X_imputed)
            logger.info(f"  Scaler fitted on {X_imputed.shape[1]} columns")

        # --- Step 3: Fit Categorical Pipeline ---
        if self.original_categorical_cols_:
            self.categorical_encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.categorical_encoder_.fit(X[self.original_categorical_cols_])
            self.final_categorical_cols_ = self.original_categorical_cols_
            logger.info(f"  Categorical encoder fitted for {len(self.final_categorical_cols_)} columns")

        # --- Step 4: Fit Target Encoder ---
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        # --- Step 5: Store final feature list (in order) ---
        self.final_features_ = self.final_numerical_cols_ + self.final_categorical_cols_
        logger.info(f"  Final feature list: {len(self.final_features_)} total columns")
        
        # --- Step 6: Setup Augmentation ---
        if self.shuffle_features_:
            self.feature_order_ = np.random.permutation(len(self.final_features_))
        if self.random_mirror_x_ and self.final_numerical_cols_:
            self.mirror_vector_ = np.random.choice([-1, 1], size=len(self.final_numerical_cols_))

        self._is_fitted = True
        logger.info("✓ Mitra Preprocessor fitted successfully")
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Applies the fitted preprocessing pipeline."""
        if not self._is_fitted:
            raise RuntimeError("You must call fit() before calling transform().")

        logger.debug(f"Transform input: {X.shape}")

        # --- CRITICAL FIX: Ensure input has EXACT columns imputer expects ---
        # Add missing columns and reorder to match fit time
        for col in self.final_features_:
            if col not in X.columns:
                logger.debug(f"  Adding missing column: {col}")
                X = X.copy()
                X[col] = np.nan

        X = X[self.final_features_].copy()
        logger.debug(f"After column alignment: {X.shape}")

        X_transformed_parts = []
        
        # --- Transform Numerical Features ---
        if self.final_numerical_cols_:
            X_num = X[self.final_numerical_cols_]
            
            # --- CRITICAL: Transform with correct columns ---
            try:
                X_num_imputed = self.imputer_.transform(X_num)
            except ValueError as e:
                logger.error(f"Imputer error: {e}")
                if hasattr(self.imputer_, 'feature_names_in_'):
                    logger.error(f"Expected columns: {self.imputer_.feature_names_in_.tolist()}")
                logger.error(f"Got columns: {X_num.columns.tolist()}")
                raise
            
            X_num_scaled = self.scaler_.transform(X_num_imputed)
            
            if self.random_mirror_x_ and hasattr(self, 'mirror_vector_'):
                X_num_scaled = X_num_scaled * self.mirror_vector_
            
            X_transformed_parts.append(X_num_scaled)

        # --- Transform Categorical Features ---
        if self.final_categorical_cols_:
            X_cat = X[self.final_categorical_cols_]
            X_cat_encoded = self.categorical_encoder_.transform(X_cat)
            X_transformed_parts.append(X_cat_encoded)
        
        if not X_transformed_parts:
            return np.array([[] for _ in range(len(X))])

        X_final = np.hstack(X_transformed_parts)

        if self.shuffle_features_ and self.feature_order_ is not None:
            X_final = X_final[:, self.feature_order_]
        
        X_final = X_final.astype(np.float32)

        if y is not None:
            y_final = self.label_encoder_.transform(y.astype(str))
            return X_final, y_final
        
        return X_final

    def get_summary(self):
        """Returns a rich dictionary with column-level details for each processing step."""
        if not self._is_fitted:
            logger.error("Error - Preprocessor has not been fitted yet.")
            return {}

        summary = {
            "Column Type Identification": {
                "description": "Identified and separated features into numerical and categorical types.",
                "details": [
                    f"Found {len(self.original_numerical_cols_)} numerical columns: {self.original_numerical_cols_}",
                    f"Found {len(self.original_categorical_cols_)} categorical columns: {self.original_categorical_cols_}"
                ]
            }
        }
        
        if self.original_numerical_cols_:
            summary["Numerical Preprocessing"] = {
                "description": "Cleaned and selected numerical features through a sequential pipeline.",
                "details": [
                    f"{len(self.original_numerical_cols_) - len(self.final_numerical_cols_)} constant columns were removed.",
                    f"Feature Selector: {'SelectKBest(k=' + str(self.max_features_) + ')' if self.feature_selector_ else 'None'}.",
                    f"Final numerical features used ({len(self.final_numerical_cols_)}): {self.final_numerical_cols_}"
                ]
            }
            summary["Numerical Scaling"] = {
                "description": f"Applied '{self.scaler_type_}' scaling to all final numerical features.",
                "details": [
                    f"The {type(self.scaler_).__name__} was fitted on {len(self.final_numerical_cols_)} columns."
                ]
            }

        if self.original_categorical_cols_:
            summary["Categorical Encoding"] = {
                "description": "Applied ordinal encoding to convert categorical features into a numerical format.",
                "details": [
                    f"The OrdinalEncoder was fitted on {len(self.original_categorical_cols_)} columns. Unknown categories during transform will be encoded as -1."
                ]
            }

        summary["Target Encoding"] = {
                "description": "Encoded the target variable into numerical labels.",
                "details": [
                    f"Fitted a LabelEncoder on the target, identifying {len(self.label_encoder_.classes_)} unique classes."
                ]
            }
        summary["Data Augmentation"] = {
                "description": "Configuration for training-time data augmentation.",
                "details": [
                    f"Shuffle Features: {self.shuffle_features_}",
                    f"Random Mirroring (Numerical): {self.random_mirror_x_}"
                ]
            }
        return summary