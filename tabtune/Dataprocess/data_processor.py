import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from category_encoders import TargetEncoder, HashingEncoder, BinaryEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NeighbourhoodCleaningRule

logger = logging.getLogger(__name__)

# --- Links to Custom Model-Specific Preprocessors ---
from .tabpfn_preprocessor import TabPFNPreprocessor
from .tabicl_preprocessor import TabICLPreprocessor
from .orion_msp_preprocessor import OrionMSPPreprocessor
from .contexttab_preprocessor import ContextTabPreprocessor
from .mitra_preprocessor import MitraPreprocessor 
from .orion_bix_preprocessor import OrionBixPreprocessor
from .tabdpt_preprocessor import TabDPTPreprocessor

class DataProcessor(BaseEstimator, TransformerMixin):
    """
    The complete Data Preparation Engine for the TabTune library. Integrates
    a full suite of standard preprocessing tools with custom, model-specific
    logic.
    """
    def __init__(self, model_name=None, override_types=None, 
                 imputation_strategy='mean', categorical_encoding='onehot', 
                 scaling_strategy='standard', resampling_strategy=None,
                 feature_selection_strategy=None, feature_selection_k=10,
                 model_params=None):
        
        self.model_name = model_name
        self.override_types = override_types
        self.imputation_strategy = imputation_strategy
        self.categorical_encoding = categorical_encoding
        self.scaling_strategy = scaling_strategy
        self.resampling_strategy = resampling_strategy
        self.feature_selection_strategy = feature_selection_strategy
        self.feature_selection_k = feature_selection_k
        self.model_params = model_params or {}

        self._set_model_aware_defaults()
        
        # --- Internal State Attributes ---
        self.column_types_ = {}
        self._is_fitted = False
        self.custom_preprocessor_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.encoder_ = None
        self.resampler_ = None
        self.selector_ = None
        self.label_encoder_ = None
        self._correlation_cols_to_drop = []
        self.original_cols_ = None
        self.processing_summary_ = {}

    def _set_model_aware_defaults(self):
        """Sets default strategies based on the model_name to ensure compatibility."""
        if self.model_name:
            model_defaults = {
                'TabPFN': {'categorical_encoding': 'tabpfn_special'},
                'TabICL': {'categorical_encoding': 'tabicl_special'},
                'OrionMSP': {'categorical_encoding': 'orion_msp_special'},
                'ContextTab': {'categorical_encoding': 'contexttab_special'},
                'Mitra': {'categorical_encoding': 'mitra_special'}, 
                'OrionBix': {'categorical_encoding': 'orion_bix_special'},
                'TabDPT': {'categorical_encoding': 'tabdpt_special'},
            }
            config = model_defaults.get(self.model_name)
            if config:
                self.imputation_strategy = config.get('imputation_strategy', self.imputation_strategy)
                self.categorical_encoding = config.get('categorical_encoding', self.categorical_encoding)
                self.scaling_strategy = config.get('scaling_strategy', self.scaling_strategy)

    def _get_custom_preprocessor(self):
        """Factory method to return the correct custom preprocessor instance."""
        special_encoders = {
            'tabpfn_special': TabPFNPreprocessor,
            'tabicl_special': TabICLPreprocessor,
            'orion_msp_special': OrionMSPPreprocessor,
            'contexttab_special': ContextTabPreprocessor,
            'mitra_special': MitraPreprocessor,
            'orion_bix_special': OrionBixPreprocessor,
            'tabdpt_special': TabDPTPreprocessor,
        }
        if self.categorical_encoding in special_encoders:
            logger.info(f"[DataProcessor] Using special preprocessor for: {self.model_name}")
            PreprocessorClass = special_encoders[self.categorical_encoding]
            if self.categorical_encoding == 'contexttab_special':
                # Extract regression parameters from model_params
                regression_type = self.model_params.get('regression_type', 'l2')
                num_regression_bins = self.model_params.get('num_regression_bins', 16)
                return PreprocessorClass(regression_type=regression_type, num_regression_bins=num_regression_bins)
            return PreprocessorClass()
        return None

    def fit(self, X, y=None):
        X_fit = X.copy()
        self.original_cols_ = X_fit.columns.tolist()
        y_fit = y.copy() if y is not None else None
        
        self.custom_preprocessor_ = self._get_custom_preprocessor()
        
        if self.custom_preprocessor_:
            self.custom_preprocessor_.fit(X, y)
            
            self.processing_summary_['strategy'] = 'custom'
            if hasattr(self.custom_preprocessor_, 'get_summary'):
                self.processing_summary_['steps'] = self.custom_preprocessor_.get_summary()
        else:
            # Log summary for standard path
            self.processing_summary_['strategy'] = 'standard'
            self.processing_summary_['steps'] = {}
            self._infer_column_types(X_fit)
            if y_fit is not None:
                self.label_encoder_ = LabelEncoder().fit(y_fit)
                self.processing_summary_['target_encoding'] = 'LabelEncoder'
            self._fit_standard_components(X_fit, y_fit)
            
        self._is_fitted = True
        logger.info("[DataProcessor] All components for pipeline have been fitted.")
        return self

    def transform(self, X, y=None):
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before calling transform().")
        
        X_transformed = X.copy()
        
        if self.custom_preprocessor_:
            if self.model_name == 'TabPFN':
                return self.custom_preprocessor_.transform(X_transformed)
            else:
                return self.custom_preprocessor_.transform(X_transformed, y)

        X_transformed = self._apply_standard_transforms(X_transformed)

        if y is not None and self.label_encoder_:
            y_transformed = self.label_encoder_.transform(y)
            return X_transformed, y_transformed
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        
        if self.custom_preprocessor_:
            return self.transform(X, y)

        X_transformed, y_transformed = self.transform(X, y)
        
        if self.resampling_strategy and y is not None:
            self._fit_resampler(y_transformed)
            if self.resampler_:
                logger.info(f"[DataProcessor] Resampling data with '{self.resampling_strategy}'...")
                X_transformed, y_transformed = self.resampler_.fit_resample(X_transformed, y_transformed)
                self.processing_summary_['resampling'] = self.resampling_strategy
        
        return X_transformed, y_transformed
    
    def get_processing_summary(self):
        """
        Returns a formatted string summarizing the data processing steps.
        """
        if not self._is_fitted:
            logger.warning("[DataProcessor] DataProcessor has not been fitted yet.")
            raise RuntimeError("DataProcessor has not been fitted yet.")

        summary_lines = ["--- Data Processing Summary ---"]

        if self.processing_summary_.get('strategy') == 'custom':
            summary_lines.append(f"\n[Custom Preprocessing for '{self.model_name}']")
            
            steps = self.processing_summary_.get('steps', {})
            if not steps:
                summary_lines.append("  - No detailed summary available for this preprocessor.")
            else:
                summary_lines.append("\n  Applied Steps:")
                # --- NEW: Detailed loop for rich summary ---
                for i, (step_name, step_info) in enumerate(steps.items()):
                    summary_lines.append(f"    {i+1}. {step_name}:")
                    summary_lines.append(f"       - {step_info['description']}")
                    for detail_line in step_info.get('details', []):
                        summary_lines.append(f"       - {detail_line}")
                
        elif self.processing_summary_.get('strategy') == 'standard':
            summary_lines.append("\n[Standard Preprocessing Pipeline]")
            
            steps = self.processing_summary_.get('steps', {})
            processed_cols = set()

            if 'imputation' in steps:
                step_info = steps['imputation']
                summary_lines.append(f"\n1. Imputation (Strategy: '{step_info['strategy']}')")
                summary_lines.append(f"   - Applied to {len(step_info['columns'])} numerical features: {', '.join(f'`{c}`' for c in step_info['columns'])}")
                processed_cols.update(step_info['columns'])

            if 'categorical_encoding' in steps:
                step_info = steps['categorical_encoding']
                summary_lines.append(f"\n2. Categorical Encoding (Strategy: '{step_info['strategy']}')")
                summary_lines.append(f"   - Applied to {len(step_info['columns'])} categorical features: {', '.join(f'`{c}`' for c in step_info['columns'])}")
                processed_cols.update(step_info['columns'])
            
            if 'scaling' in steps:
                step_info = steps['scaling']
                summary_lines.append(f"\n3. Scaling (Strategy: '{step_info['strategy']}')")
                summary_lines.append(f"   - Applied to {len(step_info['columns'])} features (original numerical + encoded categorical).")
                processed_cols.update(step_info['columns'])

            if 'feature_selection' in steps:
                step_info = steps['feature_selection']
                summary_lines.append(f"\n4. Feature Selection (Strategy: '{step_info['strategy']}')")
                if 'dropped_columns' in step_info and step_info['dropped_columns']:
                    summary_lines.append(f"   - Removed {len(step_info['dropped_columns'])} features: {', '.join(f'`{c}`' for c in step_info['dropped_columns'])}")
                else:
                    summary_lines.append("   - No features were removed by this step.")
            
            untouched_features = [col for col in self.original_cols_ if col not in processed_cols and col not in (steps.get('feature_selection', {}).get('dropped_columns', []))]
            summary_lines.append(f"\n[Untouched Features]")
            if untouched_features:
                summary_lines.append(f"  - {len(untouched_features)} features were not modified: {', '.join(f'`{c}`' for c in untouched_features)}")
            else:
                summary_lines.append("  - All features were processed by at least one step.")
        else:
            summary_lines.append("No processing steps were recorded.")

        if 'resampling' in self.processing_summary_:
             summary_lines.append(f"\n[Resampling]")
             summary_lines.append(f"  - Strategy: '{self.processing_summary_['resampling']}' applied to the training data.")

        return "\n".join(summary_lines)

    def _infer_column_types(self, X):
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()

    def _fit_standard_components(self, X, y):
        if self.imputation_strategy != 'none' and self.numerical_cols_:
            imputer_map = {'mean': SimpleImputer(strategy='mean'), 'median': SimpleImputer(strategy='median'), 'iterative': IterativeImputer(random_state=42), 'knn': KNNImputer()}
            self.imputer_ = imputer_map.get(self.imputation_strategy, SimpleImputer(strategy='mean'))
            X[self.numerical_cols_] = self.imputer_.fit_transform(X[self.numerical_cols_])
            self.processing_summary_['steps']['imputation'] = {'strategy': self.imputation_strategy, 'columns': self.numerical_cols_}

        if self.categorical_encoding != 'none' and self.categorical_cols_:
            encoder_map = {'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False), 'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 'target': TargetEncoder(), 'hashing': HashingEncoder(), 'binary': BinaryEncoder()}
            self.encoder_ = encoder_map.get(self.categorical_encoding, OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            self.encoder_.fit(X[self.categorical_cols_], y)
            self.processing_summary_['steps']['categorical_encoding'] = {'strategy': self.categorical_encoding, 'columns': self.categorical_cols_}
        
        X_encoded = self._apply_encoding(X)
        numeric_for_scaling = X_encoded.select_dtypes(include=np.number).columns.tolist()

        if self.scaling_strategy != 'none' and numeric_for_scaling:
            scaler_map = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler(), 'power_transform': PowerTransformer()}
            self.scaler_ = scaler_map.get(self.scaling_strategy, StandardScaler())
            self.scaler_.fit(X_encoded[numeric_for_scaling])
            self.processing_summary_['steps']['scaling'] = {'strategy': self.scaling_strategy, 'columns': numeric_for_scaling}
        
        X_scaled = self._apply_scaling(X_encoded, numeric_for_scaling)

        if self.feature_selection_strategy:
            self._fit_feature_selector(X_scaled, y)

    def _apply_standard_transforms(self, X):
        X_transformed = X.copy()
        if self.imputer_ and self.numerical_cols_:
            X_transformed[self.numerical_cols_] = self.imputer_.transform(X_transformed[self.numerical_cols_])
        
        X_transformed = self._apply_encoding(X_transformed)
        numeric_for_scaling = X_transformed.select_dtypes(include=np.number).columns.tolist()
        X_transformed = self._apply_scaling(X_transformed, numeric_for_scaling)

        if self.selector_ or self._correlation_cols_to_drop:
            X_transformed = self._apply_feature_selection(X_transformed)
            
        return X_transformed
        
    def _apply_encoding(self, X):
        if not self.encoder_ or not self.categorical_cols_: return X
        encoded_data = self.encoder_.transform(X[self.categorical_cols_])
        try:
            encoded_cols = self.encoder_.get_feature_names_out(self.categorical_cols_)
        except:
            encoded_cols = [f"cat_{i}" for i in range(encoded_data.shape[1])]
        encoded_df = pd.DataFrame(encoded_data, index=X.index, columns=encoded_cols)
        X_transformed = X.drop(columns=self.categorical_cols_)
        return pd.concat([X_transformed, encoded_df], axis=1)

    def _apply_scaling(self, X, cols_to_scale):
        if not self.scaler_ or not cols_to_scale: return X
        X_scaled = X.copy()
        X_scaled[cols_to_scale] = self.scaler_.transform(X_scaled[cols_to_scale])
        return X_scaled
        
    def _fit_feature_selector(self, X, y):
        logger.debug(f"[DataProcessor] Fitting feature selector: '{self.feature_selection_strategy}'...")
        selector_map = {'variance': VarianceThreshold(threshold=0.0), 'select_k_best_anova': SelectKBest(f_classif, k=self.feature_selection_k), 'select_k_best_chi2': SelectKBest(chi2, k=self.feature_selection_k)}
        self.selector_ = selector_map.get(self.feature_selection_strategy)
        
        if self.selector_:
            X_to_fit = X.copy()
            if self.feature_selection_strategy == 'select_k_best_chi2':
                X_to_fit = MinMaxScaler().fit_transform(X_to_fit) 
            self.selector_.fit(X_to_fit, y)
            dropped_cols = X.columns[~self.selector_.get_support()].tolist()
            self.processing_summary_['steps']['feature_selection'] = {'strategy': self.feature_selection_strategy, 'k': self.feature_selection_k, 'dropped_columns': dropped_cols}
        
        if self.feature_selection_strategy == 'correlation':
            self._fit_correlation_selector(X)
            
    def _apply_feature_selection(self, X):
        X_selected = X
        if self.selector_ and self.feature_selection_strategy != 'correlation':
             selected_cols = X.columns[self.selector_.get_support()]
             X_selected = pd.DataFrame(self.selector_.transform(X), index=X.index, columns=selected_cols)
        
        if self._correlation_cols_to_drop:
             X_selected = X_selected.drop(columns=self._correlation_cols_to_drop, errors='ignore')
             
        return X_selected
        
    def _fit_correlation_selector(self, X, threshold=0.9):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self._correlation_cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        logger.debug(f"[DataProcessor] Correlation filter identified {len(self._correlation_cols_to_drop)} columns to drop.")
        self.processing_summary_['steps']['feature_selection'] = {'strategy': 'correlation', 'threshold': threshold, 'dropped_columns': self._correlation_cols_to_drop}

    def _fit_resampler(self, y):
        if self.resampling_strategy:
            k_neighbors = 5
            if self.resampling_strategy == 'smote':
                min_class_count = pd.Series(y).value_counts().min()
                k_neighbors = max(1, min_class_count - 1)
            
            resampler_map = {
                'smote': SMOTE(random_state=42, k_neighbors=k_neighbors),
                'random_over': RandomOverSampler(random_state=42),
                'random_under': RandomUnderSampler(random_state=42),
                'tomek': TomekLinks(),
                'kmeans': ClusterCentroids(random_state=42),
                'knn': NeighbourhoodCleaningRule()
            }
            self.resampler_ = resampler_map.get(self.resampling_strategy)
