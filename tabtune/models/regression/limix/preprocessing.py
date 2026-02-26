"""
Preprocessing pipeline for Limix regression to match official LimiX performance.

This module implements the preprocessing steps used by the official LimiX implementation:
RebalanceFeatureDistribution, CategoricalFeatureEncoder, FeatureShuffler, FilterValidFeatures,
RobustPowerTransformer, and SelectiveInversePipeline.
"""
import numpy as np
import pandas as pd
from typing import Literal, Optional, Any
from sklearn.preprocessing import (
    QuantileTransformer, PowerTransformer, OrdinalEncoder, OneHotEncoder,
    FunctionTransformer, StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted
import warnings
import logging

logger = logging.getLogger(__name__)


def infer_random_state(random_state):
    """Infer random state and return generator."""
    MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)
    if random_state is None:
        np_rng = np.random.default_rng()
        return int(np_rng.integers(0, MAXINT_RANDOM_SEED)), np_rng
    if isinstance(random_state, (int, np.integer)):
        return int(random_state), np.random.default_rng(int(random_state))
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, MAXINT_RANDOM_SEED)), random_state
    raise ValueError(f"Invalid random_state {random_state}")


class SelectiveInversePipeline(Pipeline):
    """Pipeline that can skip inverse_transform for specified steps."""
    def __init__(self, steps, skip_inverse=None):
        super().__init__(steps)
        self.skip_inverse = skip_inverse or []
    
    def inverse_transform(self, X):
        """Skip inverse_transform for specified steps."""
        if X.shape[1] == 0:
            return X
        for step_idx in range(len(self.steps) - 1, -1, -1):
            name, transformer = self.steps[step_idx]
            try:
                check_is_fitted(transformer)
            except:
                continue
            
            if name in self.skip_inverse:
                continue
                
            if hasattr(transformer, 'inverse_transform'):
                X = transformer.inverse_transform(X)
                if np.any(np.isnan(X)):
                    logger.warning(f"After reverse {name}, there is nan")
        return X


class RobustPowerTransformer(PowerTransformer):
    """PowerTransformer with automatic feature reversion when variance or value constraints fail."""
    
    def __init__(self, var_tolerance: float = 1e-3,
                 max_abs_value: float = 100,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.var_tolerance = var_tolerance
        self.max_abs_value = max_abs_value
        self.restore_indices_: np.ndarray | None = None
    
    def fit(self, X, y=None):
        fitted = super().fit(X, y)
        self.restore_indices_ = np.array([], dtype=int)
        return fitted
    
    def fit_transform(self, X, y=None):
        Z = super().fit_transform(X, y)
        self.restore_indices_ = self._should_revert(Z)
        return Z
    
    def _should_revert(self, Z: np.ndarray) -> np.ndarray:
        """Determine which columns to revert to their original values."""
        variances = np.nanvar(Z, axis=0)
        bad_var = np.flatnonzero(np.abs(variances - 1.0) > self.var_tolerance)
        bad_large = np.flatnonzero(np.any(Z > self.max_abs_value, axis=0))
        return np.unique(np.concatenate([bad_var, bad_large]))
    
    def _apply_reversion(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.restore_indices_ is not None and self.restore_indices_.size > 0:
            Z[:, self.restore_indices_] = X[:, self.restore_indices_]
        return Z
    
    def transform(self, X):
        Z = super().transform(X)
        return self._apply_reversion(Z, X)
    
    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        """Override to avoid crashes caused by NaN and Inf."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message=r"overflow encountered",
                                        category=RuntimeWarning)
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except Exception as e:
            return np.nan
    
    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Override to avoid crashes caused by NaN."""
        if np.isnan(lmbda):
            return x
        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore


class FilterValidFeatures:
    """Filter out invalid features (constant, NaN-only, etc.)."""
    
    def __init__(self):
        self.valid_features = None
        self.categorical_idx = None
        self.invalid_indices = None
        self.invalid_features = None
    
    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, y: Optional[np.ndarray] = None, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Remove invalid features."""
        self.categorical_idx = categorical_features
        
        # Find constant features (all values same as first row)
        self.valid_features = ((x[0:1, :] == x).mean(axis=0) < 1.0).tolist()
        self.invalid_indices = ((x[0:1, :] == x).mean(axis=0) == 1.0).tolist()
        
        # Check for NaN-only features if y is provided (to split train/test)
        if y is not None:
            eval_pos = len(y)
            nan_train = np.isnan(x[:eval_pos, :])
            all_nan_train = np.all(nan_train, axis=0)
            nan_test = np.isnan(x[eval_pos:, :])
            all_nan_test = np.all(nan_test, axis=0)
            
            features_nan = all_nan_train | all_nan_test
            self.valid_features = [v and not n for v, n in zip(self.valid_features, features_nan)]
            self.invalid_indices = [i or n for i, n in zip(self.invalid_indices, features_nan)]
        
        if not any(self.valid_features):
            raise ValueError("All features are constant! Please check your data.")
        
        # Convert to numpy arrays for indexing
        valid_features_array = np.array(self.valid_features)
        invalid_indices_array = np.array(self.invalid_indices)
        
        # Store invalid features
        if np.any(invalid_indices_array):
            self.invalid_features = x[:, invalid_indices_array].copy()
            logger.info(f"[FilterValidFeatures] Removing {np.sum(invalid_indices_array)} invalid features")
        
        # Update categorical features indices
        valid_indices_where = np.where(valid_features_array)[0]
        self.categorical_idx = [
            index
            for index, idx in enumerate(valid_indices_where)
            if idx in categorical_features
        ]
        
        # Filter features
        x_filtered = x[:, valid_features_array]
        
        return x_filtered, self.categorical_idx
    
    def transform(self, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Transform data by filtering invalid features."""
        if self.valid_features is None:
            raise RuntimeError("Must call fit_transform first")
        valid_features_array = np.array(self.valid_features)
        self.invalid_features = x[:, ~valid_features_array] if np.any(~valid_features_array) else None
        return x[:, valid_features_array], self.categorical_idx or []


class RebalanceFeatureDistribution:
    """
    RebalanceFeatureDistribution for Limix regression.
    Supports 'quantile_uniform_all_data' and 'power' worker_tags.
    Includes SVD feature union when svd_tag is set.
    """
    
    def __init__(
        self,
        worker_tags: list[str] = ["quantile"],
        original_flag: bool = False,
        svd_tag: Literal['svd'] | None = None,
        discrete_flag: bool = False,
    ):
        self.worker_tags = worker_tags
        self.original_flag = original_flag
        self.svd_tag = svd_tag
        self.discrete_flag = discrete_flag
        self.random_state = None
        self.worker = None
        self.categorical_features = None
        self.n_quantile_features = 0
        self.svd_n_comp = 0
        self.feature_indices = None
        self.dis_ix = []
        
    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, y: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[int]]:
        """
        Fit and transform data.
        
        CRITICAL: x should contain BOTH train and test data concatenated.
        y should be only the training targets (used to determine train/test split).
        The preprocessing fits on train portion only, but transforms entire x.
        """
        self.random_state = seed
        self.categorical_features = categorical_features
        
        # Determine train/test split from y length
        if y is not None:
            n_train = len(y)
            x_train = x[:n_train]
        else:
            n_train = len(x)
            x_train = x
        
        n_samples, n_features = x_train.shape  # Use train size for n_quantiles calculation
        all_ix = list(range(n_features))
        cont_ix = [i for i in all_ix if i not in categorical_features]
        
        workers = []
        trans_ixs = cont_ix
        
        if self.original_flag:
            # Keep original features
            if self.discrete_flag:
                trans_ixs = categorical_features + cont_ix
            workers.append(("original", "passthrough", all_ix))
            self.dis_ix = categorical_features
        elif self.discrete_flag:
            trans_ixs = categorical_features + cont_ix
            self.feature_indices = categorical_features + cont_ix
            self.dis_ix = []
        else:
            workers.append(("discrete", "passthrough", categorical_features))
            trans_ixs = cont_ix
            self.dis_ix = list(range(len(categorical_features)))
        
        static_seed, rng = infer_random_state(self.random_state)
        
        for worker_tag in self.worker_tags:
            if worker_tag == "quantile_uniform_all_data" or worker_tag == "quantile":
                # Quantile transformation to uniform distribution
                # Match official: n_quantiles=max(n_samples // 5, 2), subsample=n_samples
                n_quantiles = max(n_samples // 5, 2)
                worker = Pipeline([
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)),
                    ("quantile", QuantileTransformer(
                        n_quantiles=n_quantiles,
                        output_distribution='uniform',
                        subsample=n_samples,  # Use all training data
                        random_state=static_seed
                    ))
                ])
                workers.append((f"feat_transform_{worker_tag}", worker, trans_ixs))
                self.n_quantile_features = len(trans_ixs)
                
            elif worker_tag == "power":
                # Power transformation (Yeo-Johnson) - use RobustPowerTransformer
                self.feature_indices = categorical_features + cont_ix
                nan_to_mean_transformer = SimpleImputer(
                    missing_values=np.nan,
                    strategy="mean",
                    keep_empty_features=True
                )
                
                worker = SelectiveInversePipeline(
                    steps=[
                        ("power_transformer", RobustPowerTransformer(standardize=False)),
                        ("inf_to_nan_1", FunctionTransformer(
                            func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                            inverse_func=lambda x: x,
                            check_inverse=False
                        )),
                        ("nan_to_mean_1", nan_to_mean_transformer),
                        ("scaler", StandardScaler()),
                        ("inf_to_nan_2", FunctionTransformer(
                            func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                            inverse_func=lambda x: x,
                            check_inverse=False
                        )),
                        ("nan_to_mean_2", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True))
                    ],
                    skip_inverse=['nan_to_mean_1', 'nan_to_mean_2']
                )
                workers.append((f"feat_transform_{worker_tag}", worker, trans_ixs))
        
        CT_worker = ColumnTransformer(workers, remainder="drop", sparse_threshold=0.0)
        
        # Add SVD feature union if svd_tag is set
        if self.svd_tag == "svd" and n_features >= 2:
            svd_worker = FeatureUnion([
                ("default", FunctionTransformer(func=lambda x: x)),
                ("svd", Pipeline(steps=[
                    ("save_standard", Pipeline(steps=[
                        ("i2n_pre", FunctionTransformer(
                            func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                            inverse_func=lambda x: x,
                            check_inverse=False
                        )),
                        ("fill_missing_pre", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)),
                        ("standard", StandardScaler(with_mean=False)),
                        ("i2n_post", FunctionTransformer(
                            func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                            inverse_func=lambda x: x,
                            check_inverse=False
                        )),
                        ("fill_missing_post", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True))
                    ])),
                    ("svd", TruncatedSVD(
                        algorithm="arpack",
                        n_components=max(1, min(n_samples // 10 + 1, n_features // 2)),
                        random_state=static_seed
                    ))
                ]))
            ])
            self.svd_n_comp = max(1, min(n_samples // 10 + 1, n_features // 2))
            self.worker = Pipeline([("worker", CT_worker), ("svd_worker", svd_worker)])
        else:
            self.svd_n_comp = 0
            self.worker = CT_worker
        
        # Fit on training data only, but transform entire x (train + test)
        self.worker.fit(x_train)
        x_transformed = self.worker.transform(x)
        return x_transformed, categorical_features
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using fitted transformer."""
        if self.worker is None:
            raise RuntimeError("Must call fit_transform first")
        return self.worker.transform(x)


class CategoricalFeatureEncoder:
    """
    Simplified CategoricalFeatureEncoder for Limix regression.
    Supports 'ordinal_strict_feature_shuffled' and 'onehot' encoding strategies.
    """
    
    def __init__(
        self,
        encoding_strategy: Literal['ordinal', 'ordinal_strict_feature_shuffled', 'ordinal_shuffled', 'onehot'] = "ordinal",
    ):
        self.encoding_strategy = encoding_strategy
        self.random_seed = None
        self.transformer = None
        self.category_mappings = None
        self.categorical_features = None
        
    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Fit and transform categorical features."""
        self.random_seed = seed
        self.categorical_features = categorical_features
        
        if len(categorical_features) == 0:
            self.transformer = None
            return x, []
        
        static_seed, rng = infer_random_state(self.random_seed)
        
        if self.encoding_strategy.startswith("ordinal"):
            # Ordinal encoding
            ct = ColumnTransformer(
                [("ordinal_encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", 
                    unknown_value=np.nan
                ), categorical_features)],
                remainder="passthrough"
            )
            
            Xt = ct.fit_transform(x)
            new_categorical_features = list(range(len(categorical_features)))
            
            if self.encoding_strategy.endswith("_shuffled") or "shuffled" in self.encoding_strategy:
                # Shuffle category mappings
                self.category_mappings = {}
                for col_ix in new_categorical_features:
                    # Get number of categories (approximate from data)
                    col_data = Xt[:, col_ix]
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        n_cats = int(valid_data.max()) + 1
                        perm = rng.permutation(n_cats)
                        self.category_mappings[col_ix] = perm
                        
                        # Apply permutation
                        valid_mask = ~np.isnan(col_data)
                        col_data_int = col_data[valid_mask].astype(int)
                        col_data[valid_mask] = perm[col_data_int].astype(col_data.dtype)
            
            self.transformer = ct
            self.categorical_features = new_categorical_features
            return Xt, new_categorical_features
            
        elif self.encoding_strategy == "onehot":
            # One-hot encoding
            ct = ColumnTransformer(
                [("one_hot_encoder", OneHotEncoder(
                    drop="if_binary",
                    sparse_output=False,
                    handle_unknown="ignore"
                ), categorical_features)],
                remainder="passthrough"
            )
            
            Xt = ct.fit_transform(x)
            # Update categorical features to include all one-hot columns
            if hasattr(ct, 'output_indices_') and 'one_hot_encoder' in ct.output_indices_:
                new_categorical_features = list(range(len(ct.output_indices_['one_hot_encoder'])))
            else:
                # Fallback: assume all new columns from one-hot are categorical
                n_original = x.shape[1]
                n_new = Xt.shape[1]
                new_categorical_features = list(range(n_new - (n_original - len(categorical_features))))
            
            self.transformer = ct
            self.categorical_features = new_categorical_features
            return Xt, new_categorical_features
        else:
            raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using fitted transformer."""
        if self.transformer is None:
            return x
        return self.transformer.transform(x)


class FeatureShuffler:
    """
    Simplified FeatureShuffler for Limix regression.
    Supports 'shuffle' and 'rotate' modes.
    """
    
    def __init__(
        self,
        mode: Literal['rotate', 'shuffle'] = "shuffle",
        offset: int = 0,
    ):
        self.mode = mode
        self.offset = offset
        self.random_seed = None
        self.feature_indices = None
        self.categorical_indices = None
    
    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Fit and transform by shuffling features."""
        self.random_seed = seed
        n_features = x.shape[1]
        
        indices = np.arange(n_features)
        
        static_seed, rng = infer_random_state(self.random_seed)
        
        if self.mode == "rotate":
            self.feature_indices = np.roll(indices, self.offset)
        elif self.mode == "shuffle":
            self.feature_indices = rng.permutation(indices)
        else:
            self.feature_indices = indices
        
        # Update categorical feature indices after shuffling
        is_categorical = np.isin(np.arange(n_features), categorical_features)
        self.categorical_indices = np.where(is_categorical[self.feature_indices])[0].tolist()
        
        x_shuffled = x[:, self.feature_indices]
        return x_shuffled, self.categorical_indices
    
    def transform(self, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Transform data by shuffling features."""
        if self.feature_indices is None:
            raise RuntimeError("Must call fit_transform first")
        if len(self.feature_indices) != x.shape[1]:
            raise ValueError(f"Feature count mismatch: {len(self.feature_indices)} != {x.shape[1]}")
        return x[:, self.feature_indices], self.categorical_indices or []


class LimixPreprocessingPipeline:
    """
    Complete preprocessing pipeline matching official LimiX configuration.
    """
    
    def __init__(
        self,
        rebalance_config: dict = None,
        categorical_config: dict = None,
        shuffler_config: dict = None,
        filter_valid: bool = True,
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            rebalance_config: Config for RebalanceFeatureDistribution
            categorical_config: Config for CategoricalFeatureEncoder
            shuffler_config: Config for FeatureShuffler
            filter_valid: Whether to apply FilterValidFeatures (default: True)
        """
        self.rebalance_config = rebalance_config or {
            "worker_tags": ["quantile_uniform_all_data"],
            "original_flag": True,
            "svd_tag": "svd"
        }
        self.categorical_config = categorical_config or {
            "encoding_strategy": "ordinal_strict_feature_shuffled"
        }
        self.shuffler_config = shuffler_config or {
            "mode": "shuffle"
        }
        self.filter_valid = filter_valid
        
        self.filter = None
        self.rebalance = None
        self.categorical = None
        self.shuffler = None
        self.categorical_features = None
        
    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, y: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[int]]:
        """Apply full preprocessing pipeline."""
        # Step 0: FilterValidFeatures (if enabled)
        if self.filter_valid:
            self.filter = FilterValidFeatures()
            x, categorical_features = self.filter.fit_transform(x, categorical_features, seed)
        
        # Step 1: RebalanceFeatureDistribution
        self.rebalance = RebalanceFeatureDistribution(**self.rebalance_config)
        x, categorical_features = self.rebalance.fit_transform(x, categorical_features, seed, y=y)
        
        # Step 2: CategoricalFeatureEncoder
        self.categorical = CategoricalFeatureEncoder(**self.categorical_config)
        x, categorical_features = self.categorical.fit_transform(x, categorical_features, seed)
        
        # Step 3: FeatureShuffler
        self.shuffler = FeatureShuffler(**self.shuffler_config)
        x, categorical_features = self.shuffler.fit_transform(x, categorical_features, seed)
        
        self.categorical_features = categorical_features
        return x, categorical_features
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using fitted pipeline."""
        if self.rebalance is None:
            raise RuntimeError("Must call fit_transform first")
        
        # Apply filters in same order as fit_transform
        if self.filter_valid and self.filter is not None:
            x, _ = self.filter.transform(x)  # FilterValidFeatures returns (x, categorical_idx)
        
        x = self.rebalance.transform(x)
        x = self.categorical.transform(x)
        x, _ = self.shuffler.transform(x)
        return x
