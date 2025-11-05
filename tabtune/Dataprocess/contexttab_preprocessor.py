import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Suppress pandas date parsing warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='Could not infer format')

class ContextTabPreprocessor(BaseEstimator, TransformerMixin):
    """
    Encapsulates the special preprocessing for ContextTab, which now handles three types of features:
    1. Standard scaling for numerical features (L2 mode) or quantization-based binning (non-L2 mode).
    2. Generating sentence embeddings for each categorical/text feature.
    3. Extracting year, month, day, and weekday components from date features.
    It also generates embeddings for the column names themselves.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', regression_type='l2', num_regression_bins=16):
        self.scaler_ = StandardScaler()
        self.model_name_ = model_name
        self.embedding_model_ = SentenceTransformer(model_name)
        self.label_encoder_ = LabelEncoder()
        self.numerical_cols_ = []
        self.categorical_cols_ = []
        self.date_cols_ = []
        self.column_embeddings_ = None
        self._is_fitted = False
        self.regression_type = regression_type
        self.num_regression_bins = num_regression_bins

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting ContextTab Preprocessor...")
        
        # 1. column types identification
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns
        
        for col in non_numeric_cols:
            # check for date columns 
            try:
                if pd.to_datetime(X[col], errors='raise').notna().all():
                    self.date_cols_.append(col)
            except (ValueError, TypeError):
                continue
        
        self.categorical_cols_ = [
            col for col in non_numeric_cols if col not in self.date_cols_
        ]

        print(f" Found {len(self.numerical_cols_)} numerical, {len(self.date_cols_)} date, and {len(self.categorical_cols_)} categorical columns.")

        # scaler and encoder fitting
        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])
        self.label_encoder_.fit(y)

        # embeddings generation for column names
        # order must match the concatenation order in transform(): Numerical -> Date -> Categorical
        all_feature_cols = self.numerical_cols_ + self.date_cols_ + self.categorical_cols_
        if all_feature_cols:
             logger.debug(" generating embeddings for column names")
             self.column_embeddings_ = self.embedding_model_.encode(all_feature_cols, show_progress_bar=False)
        
        # Store training data for quantization
        self.X_train_ = X.copy()
        
        self._is_fitted = True
        logger.info("ContextTab Preprocessor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        if not self._is_fitted:
            raise logger.error("You must call fit() before calling transform().")

        X_transformed_parts = []
        
        #  Processing numerical features
        if self.numerical_cols_:
            if self.regression_type == 'l2':
                # Standard scaling with mean imputation (current behavior)
                X_num_scaled = self.scaler_.transform(X[self.numerical_cols_])
                X_transformed_parts.append(X_num_scaled)
            else:
                # Quantization-based binning (new behavior)
                X_num_quantized = self._quantize_numerical_features(X[self.numerical_cols_])
                X_transformed_parts.append(X_num_quantized)
        
        # Processing date features
        if self.date_cols_:
            date_features_list = []
            for col in self.date_cols_:
                dt_col = pd.to_datetime(X[col])
                date_features = np.vstack([
                    dt_col.dt.year, dt_col.dt.month,
                    dt_col.dt.day, dt_col.dt.weekday,
                ]).T
                date_features_list.append(date_features)
            
            # creating 3d tensors
            X_date_3d = np.stack(date_features_list, axis=1)
            
            # flatten to 2D to pass through the pipeline
            num_samples = X_date_3d.shape[0]
            X_date_flat = X_date_3d.reshape(num_samples, -1)
            X_transformed_parts.append(X_date_flat)

        # processing categorical features
        if self.categorical_cols_:
            column_embeddings = []
            logger.info(f"    - Generating embeddings for {len(self.categorical_cols_)} categorical columns...")
            for col in self.categorical_cols_:
                embeddings = self.embedding_model_.encode(X[col].astype(str).tolist(), show_progress_bar=False)
                column_embeddings.append(embeddings)

            X_cat_embedded_3d = np.stack(column_embeddings, axis=1)
            num_samples = X_cat_embedded_3d.shape[0]
            X_cat_embedded_2d_flat = X_cat_embedded_3d.reshape(num_samples, -1)
            X_transformed_parts.append(X_cat_embedded_2d_flat)
        
        # concatenating all parts, handling the case where there are no features
        if X_transformed_parts:
            X_final = np.concatenate(X_transformed_parts, axis=1)
        else:
            X_final = np.array([[] for _ in range(len(X))]) 

        if y is not None:
            y_final = self.label_encoder_.transform(y)
            return X_final, y_final
        
        return X_final

    def replace_inf_values(self, column_values: pd.Series):
        """
        Replace infinite values with clipped values based on finite data range.
        From original ConTextTab tokenizer.
        """
        array_values = column_values.values
        if not np.isfinite(array_values).any():
            clipped_values = np.full(array_values.shape, np.nan)
        else:
            max_value = array_values[np.isfinite(array_values)].max()
            min_value = array_values[np.isfinite(array_values)].min()
            clipped_values = np.clip(array_values, min_value - 1, max_value + 1)
        return pd.Series(clipped_values, index=column_values.index)

    def quantize_column(self, y_context: pd.DataFrame, y_query: pd.DataFrame):
        """
        Creates num_bins=self.num_regression_bins bins based on column[train_indices] data.
        From original ConTextTab tokenizer.
        """
        a = y_context.values.flatten()
        b = np.concatenate([a, y_query.values.flatten()])
        num_bins = self.num_regression_bins

        q = np.linspace(1 / (2 * num_bins), (2 * num_bins - 1) / (2 * num_bins), num_bins)
        quantiles = np.quantile(a, q)
        extended_quantiles = np.concatenate(([np.min(a)], quantiles, [np.max(a)]))

        # Digitize b to find which interval each value belongs to
        indices = np.digitize(b, extended_quantiles) - 1
        indices = np.clip(indices, 1, num_bins - 1)

        # Compute delta
        lower_bounds = extended_quantiles[indices]
        upper_bounds = extended_quantiles[indices + 1]
        delta = (b - lower_bounds) / np.maximum(upper_bounds - lower_bounds, 1e-10)
        delta = np.clip(delta, 0, 1)

        lower_bound_index = indices - 1
        bin_index = np.round(lower_bound_index + delta).astype(int)

        return lower_bound_index, delta, bin_index, quantiles

    def _quantize_numerical_features(self, X_num: pd.DataFrame):
        """
        Apply quantization to numerical features based on training data.
        """
        X_quantized = np.zeros((len(X_num), len(X_num.columns)))
        
        for i, col in enumerate(X_num.columns):
            try:
                # For quantization, we need to use the training data to fit the quantiles
                # and then apply to the test data
                if hasattr(self, 'X_train_') and col in self.X_train_.columns:
                    # Use training data for context, test data for query
                    context_df = pd.DataFrame({col: self.X_train_[col]})
                    query_df = pd.DataFrame({col: X_num[col]})
                else:
                    # Fallback: use the same data for both (not ideal but works)
                    context_df = pd.DataFrame({col: X_num[col]})
                    query_df = pd.DataFrame({col: X_num[col]})
                
                lower_bound_index, delta, bin_index, quantiles = self.quantize_column(context_df, query_df)
                # Use bin_index as the quantized values
                # Only take the query portion (last part of bin_index)
                query_start = len(context_df)
                X_quantized[:, i] = bin_index[query_start:]
            except Exception as e:
                print(f"Warning: Quantization failed for column {col}: {e}")
                # Fallback to standard scaling
                X_quantized[:, i] = self.scaler_.transform(X_num[[col]]).flatten()
        
        return X_quantized

    def get_summary(self):
        """
        Returns a rich dictionary with column-level details for each processing step.
        """
        if not self._is_fitted:
            return {"Error": "Preprocessor has not been fitted yet."}
            
        summary = {
            "Column Type Identification": {
                "description": "Identified and separated features into three types: numerical, date, and categorical.",
                "details": [
                    f"Found {len(self.numerical_cols_)} numerical columns: {self.numerical_cols_}",
                    f"Found {len(self.date_cols_)} date columns: {self.date_cols_}",
                    f"Found {len(self.categorical_cols_)} categorical/text columns: {self.categorical_cols_}"
                ]
            },
            "Numerical Scaling": {
                "description": "Applied standard scaling (zero mean, unit variance) to all numerical features.",
                "details": [
                    f"The scaler was fitted on {len(self.numerical_cols_)} numerical columns."
                ]
            },
            "Date Feature Engineering": {
                "description": "Extracted year, month, day, and weekday as separate numerical features from each date column.",
                "details": [
                    f"This transformation was applied to {len(self.date_cols_)} date columns: {self.date_cols_}."
                ]
            },
            "Embedding Generation": {
                "description": f"Used the '{self.model_name_}' sentence transformer to generate dense vector embeddings.",
                "details": [
                    f"Generated embeddings for the values within all {len(self.categorical_cols_)} categorical columns.",
                    f"Generated a single embedding for each of the {len(self.column_embeddings_)} column names."
                ]
            },
            "Target Encoding": {
                "description": "Encoded the target variable into numerical labels.",
                "details": [
                    f"Fitted a LabelEncoder on the target, identifying {len(self.label_encoder_.classes_)} unique classes."
                ]
            }
        }
        
        return summary