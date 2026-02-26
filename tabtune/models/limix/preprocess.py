"""
Preprocessing pipeline components for LimiX ensemble inference.

This module provides preprocessing classes that match the official LimiX
preprocessing pipeline to enable ensemble-based inference.
"""
import numpy as np
from typing import Literal, Any
from sklearn.preprocessing import (
    QuantileTransformer,
    PowerTransformer,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import warnings

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state and return the seed and generator"""
    if random_state is None:
        np_rng = np.random.default_rng()
        return int(np_rng.integers(0, MAXINT_RANDOM_SEED)), np_rng
        
    if isinstance(random_state, (int, np.integer)):
        return int(random_state), np.random.default_rng(random_state)
        
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        return seed, np.random.default_rng(seed)
        
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, MAXINT_RANDOM_SEED)), random_state
        
    raise ValueError(f"Invalid random_state {random_state}")


class BasePreprocess:
    """Abstract base class for preprocessing classes"""

    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> list[int]:
        """Fit the preprocessing model to the data"""
        raise NotImplementedError
    
    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Transform the data using the fitted preprocessing model"""
        raise NotImplementedError
    
    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Fit the preprocessing model to the data and transform the data"""
        self.fit(x, categorical_features, seed, **kwargs)
        return self.transform(x, **kwargs)


class FilterValidFeatures(BasePreprocess):
    """Remove constant features and features with all NaN values."""
    
    def __init__(self):
        self.valid_features: np.ndarray | None = None
        self.categorical_idx: list[int] | None = None
        self.invalid_indices: np.ndarray | None = None
        self.invalid_features: np.ndarray | None = None

    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, y: np.ndarray | None = None, **kwargs) -> list[int]:
        self.categorical_idx = categorical_features
        
        # Check for constant features (all values the same)
        if x.shape[0] > 1:
            self.valid_features = ((x[0:1, :] == x).mean(axis=0) < 1.0)
        else:
            self.valid_features = np.ones(x.shape[1], dtype=bool)
        
        self.invalid_indices = ~self.valid_features

        # Check for all-NaN features
        if y is not None:
            eval_pos = len(y)
            if eval_pos < x.shape[0]:
                nan_train = np.isnan(x[:eval_pos, :])
                all_nan_train = np.all(nan_train, axis=0)
                nan_test = np.isnan(x[eval_pos:, :])
                all_nan_test = np.all(nan_test, axis=0)
                
                features_nan = all_nan_train | all_nan_test
                self.valid_features = self.valid_features & ~features_nan
                self.invalid_indices = self.invalid_indices | features_nan

        if not np.any(self.valid_features):
            raise ValueError("All features are constant! Please check your data.")

        # Update categorical indices to match filtered features
        valid_indices = np.where(self.valid_features)[0]
        self.categorical_idx = [
            idx
            for idx, orig_idx in enumerate(valid_indices)
            if orig_idx in categorical_features
        ]

        return self.categorical_idx
    
    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        assert self.valid_features is not None, "You must call fit first to get effective_features"
        self.invalid_features = x[:, self.invalid_indices] if np.any(self.invalid_indices) else None
        return x[:, self.valid_features], self.categorical_idx or []


class FeatureShuffler(BasePreprocess):
    """Feature column reordering preprocessor for ensemble diversity."""

    def __init__(
        self,
        mode: Literal['rotate', 'shuffle'] | None = "shuffle",
        offset: int = 0,
    ):
        self.mode = mode
        self.offset = offset
        self.random_seed = None
        self.feature_indices = None
        self.categorical_indices = None
    
    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> list[int]:
        n_features = x.shape[1]
        self.random_seed = seed
        
        indices = np.arange(n_features)
        
        if self.mode == "rotate":
            self.feature_indices = np.roll(indices, self.offset)
        elif self.mode == "shuffle":
            _, rng = infer_random_state(self.random_seed + self.offset)
            self.feature_indices = rng.permutation(indices)
        elif self.mode is None:
            self.feature_indices = np.arange(n_features)
        else:
            raise ValueError(f"Unsupported reordering mode: {self.mode}")

        # Update categorical indices after shuffling
        is_categorical = np.isin(np.arange(n_features), categorical_features)
        self.categorical_indices = np.where(is_categorical[self.feature_indices])[0].tolist()
        
        return self.categorical_indices

    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.feature_indices is None:
            raise RuntimeError("Please call the fit method first to initialize")
        if len(self.feature_indices) != x.shape[1]:
            raise ValueError(f"The number of features in the input data ({x.shape[1]}) does not match the training data ({len(self.feature_indices)})")
            
        return x[:, self.feature_indices], self.categorical_indices or []


class CategoricalFeatureEncoder(BasePreprocess):
    """Categorical feature encoder with multiple strategies."""

    def __init__(
        self,
        encoding_strategy: Literal['ordinal', 'ordinal_strict_feature_shuffled', 'ordinal_shuffled', 'onehot', 'numeric', 'none'] | None = "ordinal",
    ):
        self.encoding_strategy = encoding_strategy
        self.random_seed = None
        self.transformer = None
        self.category_mappings = None
        self.categorical_features = None
        self.feature_indices = None

    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> tuple[np.ndarray, list[int]]:
        self.random_seed = seed
        
        if self.encoding_strategy == "none" or len(categorical_features) == 0:
            self.transformer = None
            self.categorical_features = []
            return x, categorical_features
        
        if self.encoding_strategy == "numeric":
            # Treat categoricals as numeric (no encoding)
            self.transformer = None
            self.categorical_features = []
            return x, []
        
        # For ordinal strategies, we'll use a simplified approach
        # In practice, categoricals should already be encoded as integers
        # This is mainly for compatibility with the pipeline
        if self.encoding_strategy.startswith("ordinal"):
            # Assume categoricals are already ordinal encoded
            # Just apply shuffling if needed
            _, rng = infer_random_state(self.random_seed)
            
            if self.encoding_strategy.endswith("_shuffled"):
                self.category_mappings = {}
                for col_ix in categorical_features:
                    unique_vals = np.unique(x[:, col_ix][~np.isnan(x[:, col_ix])])
                    if len(unique_vals) > 0:
                        perm = rng.permutation(len(unique_vals))
                        self.category_mappings[col_ix] = perm
                        
                        col_data = x[:, col_ix].copy()
                        valid_mask = ~np.isnan(col_data)
                        if np.any(valid_mask):
                            # Map values through permutation
                            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                            for i in np.where(valid_mask)[0]:
                                orig_val = int(col_data[i])
                                if orig_val in val_to_idx:
                                    col_data[i] = perm[val_to_idx[orig_val]]
                            x[:, col_ix] = col_data
            
            self.transformer = None
            self.categorical_features = categorical_features
            return x, categorical_features
        
        elif self.encoding_strategy == "onehot":
            # One-hot encoding would expand features significantly
            # For now, we'll skip this and use ordinal
            # Full one-hot implementation would require sklearn OneHotEncoder
            self.transformer = None
            self.categorical_features = categorical_features
            return x, categorical_features
        
        return x, categorical_features


class RebalanceFeatureDistribution(BasePreprocess):
    """
    Rebalance feature distributions using various transformations.
    Simplified version supporting key transformations.
    """
    
    def __init__(
        self,
        *,
        worker_tags: list[str] | None = ["quantile_uniform_all_data"],
        discrete_flag: bool = False,
        original_flag: bool = False,
        svd_tag: Literal['svd'] | None = None,
        joined_svd_feature: bool = True,
        joined_log_normal: bool = True,
    ):
        self.worker_tags = worker_tags or []
        self.discrete_flag = discrete_flag
        self.original_flag = original_flag
        self.random_state = None
        self.svd_tag = svd_tag
        self.worker: Pipeline | ColumnTransformer | None = None
        self.joined_svd_feature = joined_svd_feature
        self.joined_log_normal = joined_log_normal
        self.feature_indices = None
        self.dis_ix = None
        self.svd_n_comp = 0

    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> list[int]:
        self.random_state = seed
        n_samples, n_features = x.shape
        worker, self.dis_ix, self.svd_n_comp = self._build_worker(
            n_samples, n_features, categorical_features
        )
        worker.fit(x)
        self.worker = worker
        return self.dis_ix

    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        assert self.worker is not None, "Must call fit first"
        x_transformed = self.worker.transform(x)
        return x_transformed, self.dis_ix

    def fit_transform(self, x: np.ndarray, categorical_features: list[int], seed: int, y: np.ndarray = None, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Fit the preprocessing model to the data and transform the data"""
        if y is not None:
            x_train_ = x[:len(y)]
            x_test_ = x[len(y):]
            if x_test_.shape[0] > 0 and x_train_.shape[1] != x_test_.shape[1]:
                x_test_ = x_test_[:, :x_train_.shape[1]]
            categorical_idx_ = self.fit(x_train_, categorical_features, seed, y=y)
            x_train_, categorical_idx_ = self.transform(x_train_)
            if x_test_.shape[0] > 0:
                x_test_, categorical_idx_ = self.transform(x_test_)
                x_ = np.concatenate([x_train_, x_test_], axis=0)
            else:
                x_ = x_train_
            return (x_, categorical_idx_)
        else:
            # No y provided, fit on all data
            categorical_idx_ = self.fit(x, categorical_features, seed)
            return self.transform(x)

    def _build_worker(self, n_samples: int, n_features: int, categorical_features: list[int]):
        """Build the transformation pipeline."""
        static_seed, rng = infer_random_state(self.random_state)
        all_ix = list(range(n_features))
        workers = []
        cont_ix = [i for i in all_ix if i not in categorical_features]
        svd_n_comp = 0
        
        if self.original_flag:
            trans_ixs = categorical_features + cont_ix if self.discrete_flag else cont_ix
            workers.append(("original", "passthrough", all_ix))
            dis_ix = categorical_features
        elif self.discrete_flag:
            trans_ixs = categorical_features + cont_ix
            self.feature_indices = categorical_features + cont_ix
            dis_ix = []
        else:
            workers.append(("discrete", "passthrough", categorical_features))
            trans_ixs, dis_ix = cont_ix, list(range(len(categorical_features)))
        
        # Build transformers for each worker tag
        for worker_tag in self.worker_tags:
            if worker_tag is None:
                continue
                
            if worker_tag == "quantile_uniform_10":
                sworker = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 10, 2),
                    random_state=static_seed,
                )
                workers.append(("quantile_uniform_10", sworker, trans_ixs))
                
            elif worker_tag == "quantile_uniform_5":
                sworker = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                )
                workers.append(("quantile_uniform_5", sworker, trans_ixs))
                
            elif worker_tag == "quantile_uniform_all_data":
                sworker = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                )
                workers.append(("quantile_uniform_all_data", sworker, trans_ixs))
                
            elif worker_tag == "power":
                # Simplified power transformer (Yeo-Johnson)
                nan_to_mean = SimpleImputer(
                    missing_values=np.nan,
                    strategy="mean",
                    keep_empty_features=True,
                )
                sworker = Pipeline(steps=[
                    ("power_transformer", PowerTransformer(method='yeo-johnson', standardize=False)),
                    ("inf_to_nan", FunctionTransformer(
                        func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                        inverse_func=lambda x: x,
                        check_inverse=False,
                    )),
                    ("nan_to_mean", nan_to_mean),
                    ("scaler", StandardScaler()),
                ])
                self.feature_indices = categorical_features + cont_ix
                workers.append(("power", sworker, trans_ixs))
        
        # Add SVD if requested
        if self.svd_tag == "svd" and len(trans_ixs) > 0:
            n_components = min(50, len(trans_ixs) // 2, n_samples // 2)
            if n_components > 0:
                svd_transformer = TruncatedSVD(n_components=n_components, random_state=static_seed)
                workers.append(("svd", svd_transformer, trans_ixs))
                svd_n_comp = n_components
        
        # Build ColumnTransformer
        if len(workers) == 0:
            # No transformations, just passthrough
            worker = ColumnTransformer(
                transformers=[("passthrough", "passthrough", all_ix)],
                remainder="drop",
                sparse_threshold=0.0,
                verbose_feature_names_out=False,
            )
        else:
            worker = ColumnTransformer(
                transformers=workers,
                remainder="drop",
                sparse_threshold=0.0,
                verbose_feature_names_out=False,
            )
        
        return worker, dis_ix, svd_n_comp