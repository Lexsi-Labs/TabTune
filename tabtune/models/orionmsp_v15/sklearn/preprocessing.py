from __future__ import annotations

import sys
import random
import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Literal, Callable, Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

# -----------------------------------------------------------------------------
# sklearn compatibility shim:
# Older scikit-learn versions don't have BaseEstimator._validate_data.
# We implement it using sklearn validators so your existing code works unchanged.
# -----------------------------------------------------------------------------
if not hasattr(BaseEstimator, "_validate_data"):
    def _validate_data(self, X, y=None, reset=True, **kwargs):
        """
        Backport of sklearn BaseEstimator._validate_data using sklearn validators.

        Supports the subset of parameters used in this codebase:
          - reset: bool
          - copy: bool
        """
        copy = bool(kwargs.get("copy", False))

        if y is None:
            Xc = check_array(
                X,
                accept_sparse=False,
                ensure_2d=True,
                dtype=None,
                force_all_finite="allow-nan",
            )
            if copy:
                Xc = Xc.copy()

            if reset:
                self.n_features_in_ = Xc.shape[1]
            else:
                if hasattr(self, "n_features_in_") and Xc.shape[1] != self.n_features_in_:
                    raise ValueError(
                        f"X has {Xc.shape[1]} features, but {self.__class__.__name__} "
                        f"is expecting {self.n_features_in_} features."
                    )
            return Xc

        Xc, yc = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=None,
            force_all_finite="allow-nan",
        )
        if copy:
            Xc = Xc.copy()

        if reset:
            self.n_features_in_ = Xc.shape[1]
        else:
            if hasattr(self, "n_features_in_") and Xc.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {Xc.shape[1]} features, but {self.__class__.__name__} "
                    f"is expecting {self.n_features_in_} features."
                )
        return Xc, yc

    BaseEstimator._validate_data = _validate_data


class RecursionLimitManager:
    """Context manager to temporarily set the recursion limit.

    Parameters
    ----------
    limit : int
        The recursion limit to set temporarily.

    Example
    -------
    >>> with RecursionLimitManager(4000):
    >>>     # Perform operations that require a higher recursion limit
    >>>     pass
    """

    def __init__(self, limit):
        self.limit = limit
        self.original_limit = None

    def __enter__(self):
        self.original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.original_limit)
        return False  # Return False to propagate exceptions


class TransformToNumerical(TransformerMixin, BaseEstimator):
    """Transforms non-numerical data in a DataFrame to numerical representations.

    This transformer automatically detects and converts categorical variables, text features,
    and boolean data types into numerical representations suitable for machine learning models.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print information about column classifications.

    Attributes
    ----------
    tfm_ : ColumnTransformer or FunctionTransformer
        The fitted transformer that handles the conversion of different column types.
         - If input is a DataFrame: A ColumnTransformer with OrdinalEncoder for categorical
           columns and SimpleImputer for numeric columns
         - If input is not a DataFrame: A FunctionTransformer that passes data through unchanged
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
        """Configure transformers for different column types in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data. If a DataFrame, column types are used to determine
            appropriate transformations.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        if not hasattr(X, "columns"):  # proxy way to check whether X is a dataframe without importing pandas
            # no dataframe
            self.tfm_ = FunctionTransformer()
            return self

        cat_cols = make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)
        cat_pos = [X.columns.get_loc(col) for col in cat_cols]

        numeric_cols = make_column_selector(dtype_include="number")(X)
        numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]

        self.tfm_ = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        dtype=np.int64,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    cat_pos,
                ),
                ("continuous", SimpleImputer(), numeric_pos),
            ]
        )
        self.tfm_.fit(X)

        if self.verbose:
            selected_cols = []
            for name, tfm, pos in self.tfm_.transformers_:
                if tfm != "drop":
                    cols = list(X.columns[pos])
                    selected_cols.extend(cols)
                    print(f"Columns classified as {name}: {cols}")

            dropped_cols = set(X.columns).difference(set(selected_cols))
            if len(dropped_cols) >= 1:
                print(f"The following columns are not used due to their data type: {list(dropped_cols)}")

        return self

    def transform(self, X):
        """Transform features using the fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with numerical representations.
        """
        return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Filter that removes features with only one unique value in the training set.

    Parameters
    ----------
    threshold : int, default=1
        Features with unique values less than or equal to this threshold will be removed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    n_features_out_ : int
        Number of features after filtering.

    features_to_keep_ : ndarray
        Boolean mask for features to keep.

    Notes
    -----
    1. Features with unique values <= threshold are removed.
    2. When the input dataset has very few samples (n_samples <= threshold), all features are preserved
       regardless of their unique value counts. This is a safety mechanism because:
       - With few samples, it's difficult to reliably assess feature variability
       - A feature might appear constant in few samples but vary in the complete dataset
    """

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn which features to keep based on unique value counts."""
        X = self._validate_data(X)

        # If there are very few samples, keep all features
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            # For each feature, check if it has more than threshold unique values
            self.features_to_keep_ = np.array(
                [len(np.unique(X[:, i])) > self.threshold for i in range(self.n_features_in_)]
            )

        self.n_features_out_ = np.sum(self.features_to_keep_)
        return self

    def transform(self, X):
        """Filter features according to unique value counts."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Transformer that clips extreme values based on training data distribution.

    This implementation uses a two-stage Z-score based approach to identify and clip outliers:
    1. First stage: Identify values beyond z standard deviations and mark as missing
    2. Second stage: Recompute statistics without outliers for more robust bounds
    3. Final stage: Apply log-based clipping to maintain data distribution
    """

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn clipping bounds from training data using two-stage Z-score method."""
        X = self._validate_data(X)

        # First stage: Identify outliers using initial statistics
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Create a clean copy with outliers replaced by NaN
        X_clean = X.copy()
        lower_bounds = self.means_ - self.threshold * self.stds_
        upper_bounds = self.means_ + self.threshold * self.stds_

        # Create masks for values outside bounds
        lower_mask = X < lower_bounds[np.newaxis, :]
        upper_mask = X > upper_bounds[np.newaxis, :]
        outlier_mask = np.logical_or(lower_mask, upper_mask)

        # Set outliers to NaN
        X_clean[outlier_mask] = np.nan

        # Second stage: Recompute statistics without outliers
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Compute final bounds
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_

        return self

    def transform(self, X):
        """Clip values based on learned bounds with log-based adjustments."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)

        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Custom implementation of standard scaling with clipping."""

    def __init__(self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the mean and std to be used for scaling."""
        X = self._validate_data(X)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon

        return self

    def transform(self, X):
        """Standardize features by removing the mean and scaling to unit variance."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        X_scaled = (X - self.mean_) / self.scale_
        X_clipped = np.clip(X_scaled, self.clip_min, self.clip_max)

        return X_clipped


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    """Quantile transformer adapted for tabular deep learning models."""

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: Optional[int] = None,
    ):
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the quantile transformer to training data with optional noise addition."""
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        X_modified = self._add_noise(X) if self.noise > 0 else X
        normalizer.fit(X_modified)
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        """Transform data using the fitted quantile transformer."""
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        """Add noise to the input data proportional to feature standard deviations."""
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        X_noisy = X + noise_std * rng.standard_normal(X.shape)
        return X_noisy


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Preprocessing pipeline for tabular data."""

    def __init__(
        self, normalization_method: str = "power", outlier_threshold: float = 4.0, random_state: Optional[int] = None
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline."""
        X = self._validate_data(X)

        # 1. Apply standard scaling
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        # 2. Apply normalization
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        ("quantile_rtdl", RTDLQuantileTransformer(output_distribution="normal", random_state=self.random_state)),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

        # 3. Handle outliers
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)

        return self

    def transform(self, X):
        """Apply the preprocessing pipeline."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)
        # Standard scaling
        X = self.standard_scaler_.transform(X)
        # Normalization
        if self.normalizer_ is not None:
            try:
                # this can fail in rare cases if there is an outlier in X that was not present in fit()
                X = self.normalizer_.transform(X)
            except ValueError:
                # clip values to train min/max
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
        # Outlier removal
        X = self.outlier_remover_.transform(X)

        return X


class FeatureShuffler:
    """Utility that generates feature permutations for ensemble creation."""

    def __init__(
        self,
        n_features: int,
        method: str = "latin",
        max_features_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ):
        self.n_features = n_features
        self.method = method
        self.max_features_for_latin = max_features_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[np.ndarray]:
        """Generate feature shuffling patterns for ensemble diversity."""
        self.rng_ = random.Random(self.random_state)
        feature_indices = list(range(self.n_features))

        # Use the random method if n_features exceeds the limit for Latin square
        if self.n_features > self.max_features_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method

        if method == "none" or n_estimators == 1:
            # No shuffling
            shuffle_patterns = [feature_indices]
        elif method == "shift":
            # All possible circular shifts
            shuffle_patterns = [feature_indices[-i:] + feature_indices[:-i] for i in range(self.n_features)]
        elif method == "random":
            # Random permutations
            if self.n_features <= 5:
                all_perms = [list(perm) for perm in itertools.permutations(feature_indices)]
                shuffle_patterns = self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
            else:
                shuffle_patterns = [self.rng_.sample(feature_indices, self.n_features) for _ in range(n_estimators)]
        elif method == "latin":
            # Latin square permutations
            with RecursionLimitManager(100000):  # Set a higher recursion limit to avoid recursion error
                shuffle_patterns = self._latin_squares()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'.")

        return shuffle_patterns

    def _latin_squares(self):
        """Generate Latin squares for feature shuffling."""

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            self.rng_.shuffle(square)
            trans = list(zip(*square))
            self.rng_.shuffle(trans)
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            else:
                sym = self.rng_.choice(symbols)
                symbols.remove(sym)
                square = _rls(symbols)
                square.append(square[0].copy())
                for i in range(n):
                    square[i].insert(i, sym)
                return square

        symbols = list(range(self.n_features))
        square = _rls(symbols)
        feature_shuffles = _shuffle_transpose_shuffle(square)

        return [list(shuffle) for shuffle in feature_shuffles]


class EnsembleGenerator(TransformerMixin, BaseEstimator):
    """Generate ensemble variants for robust tabular prediction with ORION-MSP."""

    def __init__(
        self,
        n_estimators: int,
        norm_methods: str | List[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shift: bool = True,
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shift = class_shift
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Create ensemble configurations and fit preprocessing pipelines."""
        self._validate_data(X, y)

        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        else:
            if isinstance(self.norm_methods, str):
                self.norm_methods_ = [self.norm_methods]
            else:
                self.norm_methods_ = self.norm_methods

        # Filter unique features
        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)

        self.X_ = X
        self.y_ = y

        # override n_features_in_ to account for unique feature filtering
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffle_patterns_, self.class_shift_offsets_ = self._generate_ensemble()

        self.preprocessors_ = {}
        for norm_method in self.ensemble_configs_:
            if norm_method not in self.preprocessors_:
                preprocessor = PreprocessingPipeline(
                    normalization_method=norm_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=self.random_state,
                )
                preprocessor.fit(X)
                self.preprocessors_[norm_method] = preprocessor

        return self

    def _generate_ensemble(self):
        """Create diverse ensemble configurations grouped by normalization method."""
        shuffler = FeatureShuffler(
            n_features=self.n_features_in_, method=self.feat_shuffle_method, random_state=self.random_state
        )
        shuffle_patterns = shuffler.shuffle(self.n_estimators)

        if self.class_shift and self.n_estimators > 1:
            shift_offsets = self.rng_.sample(range(self.n_classes_), self.n_classes_)
        else:
            shift_offsets = [0]

        shuffle_shift_configs = list(itertools.product(shuffle_patterns, shift_offsets))
        self.rng_.shuffle(shuffle_shift_configs)

        shuffle_shift_norm_configs = list(itertools.product(shuffle_shift_configs, self.norm_methods_))
        shuffle_shift_norm_configs = shuffle_shift_norm_configs[: self.n_estimators]

        # Reorganize configs so that those with the same normalization method are grouped together
        used_methods = list(set([config[1] for config in shuffle_shift_norm_configs]))

        ensemble_configs = OrderedDict()
        shuffle_patterns = OrderedDict()
        shift_offsets = OrderedDict()

        for method in used_methods:
            shuffle_shift_configs = [config[0] for config in shuffle_shift_norm_configs if config[1] == method]
            shuffle_patterns[method] = [config[0] for config in shuffle_shift_configs]
            shift_offsets[method] = [config[1] for config in shuffle_shift_configs]
            ensemble_configs[method] = shuffle_shift_configs

        return ensemble_configs, shuffle_patterns, shift_offsets

    def transform(self, X):
        """Combines training and test data to create different in-context learning prompts."""
        check_is_fitted(self, ["ensemble_configs_"])

        # Unique feature filtering
        X = self.unique_filter_.transform(X)
        y = self.y_

        data = OrderedDict()
        for norm_method, shuffle_shift_configs in self.ensemble_configs_.items():
            # Apply preprocessing
            preprocessor = self.preprocessors_[norm_method]
            X_variant = np.concatenate(
                [preprocessor.X_transformed_, preprocessor.transform(X)],
                axis=0,
            )
            # Shuffle features and shift class labels
            X_ensemble = []
            y_ensemble = []
            for shuffle_pattern, shift_offset in shuffle_shift_configs:
                X_ensemble.append(X_variant[:, shuffle_pattern])
                y_ensemble.append((y + shift_offset) % self.n_classes_)
            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))

        return data


# NaN handling
class EnhancedNanHandler(TransformerMixin, BaseEstimator):
    """
    NaN handling with indicator features.

    Creates separate indicator features for NaN, Inf, and -Inf values.
    """

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(self, keep_nans: bool = True):
        self.keep_nans = keep_nans
        self.feature_means_ = None

    def fit(self, X, y=None):
        """Compute feature means for replacing NaNs."""
        X = self._validate_data(X)

        # Ensure float dtype so np.finfo works even when upstream produced ints (e.g., all-categorical datasets)
        X = X.astype(np.float32, copy=False)

        self.feature_means_ = np.nanmean(X, axis=0, keepdims=True)
        return self

    def transform(self, X):
        """Handle NaNs with indicator features."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)

        # Ensure float dtype so np.finfo works even when upstream produced ints (e.g., all-categorical datasets)
        X = X.astype(np.float32, copy=False)

        # Create NaN indicators
        nan_mask = np.isnan(X)
        inf_mask = np.isinf(X)
        pos_inf_mask = np.logical_and(inf_mask, X > 0)
        neg_inf_mask = np.logical_and(inf_mask, X < 0)

        # Replace NaNs with feature means
        if nan_mask.any():
            X[nan_mask] = self.feature_means_[0][np.where(nan_mask)[1]]

        # Replace Inf with large finite values
        X[pos_inf_mask] = np.finfo(X.dtype).max
        X[neg_inf_mask] = np.finfo(X.dtype).min

        return X


# Normalization
class NormalizeByUsedFeatures(TransformerMixin, BaseEstimator):
    """
    Normalize by number of used features.

    Scales features to maintain unit variance when padding zeros.
    """

    def __init__(self, normalize_by_sqrt: bool = True):
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features_ = None

    def fit(self, X, y=None):
        """Compute number of used (non-constant) features."""
        X = self._validate_data(X)
        # Count non-constant features
        sel = (X[1:] == X[0]).sum(0) != (X.shape[0] - 1)
        self.number_of_used_features_ = np.clip(sel.sum(), a_min=1, a_max=None)
        return self

    def transform(self, X):
        """Normalize by used features."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)

        if self.normalize_by_sqrt:
            # Scale to maintain unit variance with appended zeros
            scale_factor = np.sqrt(X.shape[1] / self.number_of_used_features_)
        else:
            scale_factor = X.shape[1] / self.number_of_used_features_

        X = X * scale_factor
        return X


# PreprocessingPipeline (enhanced)
class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Preprocessing pipeline."""

    def __init__(
        self,
        normalization_method: str = "power",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
        enhanced_nan_handling: bool = True,
        normalize_by_used_features: bool = True,
        remove_empty_features: bool = True,
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.enhanced_nan_handling = enhanced_nan_handling
        self.normalize_by_used_features = normalize_by_used_features
        self.remove_empty_features = remove_empty_features

    def fit(self, X, y=None):
        """Fit the enhanced preprocessing pipeline."""
        X = self._validate_data(X)

        # Remove empty features
        if self.remove_empty_features:
            self.empty_feature_filter_ = UniqueFeatureFilter(threshold=1)
            X = self.empty_feature_filter_.fit_transform(X)

        # NaN handling
        if self.enhanced_nan_handling:
            self.nan_handler_ = EnhancedNanHandler(keep_nans=True)
            X = self.nan_handler_.fit_transform(X)

        # Normalize by used features
        if self.normalize_by_used_features:
            self.used_feature_normalizer_ = NormalizeByUsedFeatures(normalize_by_sqrt=True)
            X = self.used_feature_normalizer_.fit_transform(X)

        # Standard scaling
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        # Normalization
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        ("quantile_rtdl", RTDLQuantileTransformer(output_distribution="normal", random_state=self.random_state)),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

        # Outlier handling
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)

        return self

    def transform(self, X):
        """Apply the enhanced preprocessing pipeline."""
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)

        # Apply all transformations in order
        if self.remove_empty_features:
            X = self.empty_feature_filter_.transform(X)

        if self.enhanced_nan_handling:
            X = self.nan_handler_.transform(X)

        if self.normalize_by_used_features:
            X = self.used_feature_normalizer_.transform(X)

        X = self.standard_scaler_.transform(X)

        if self.normalizer_ is not None:
            try:
                X = self.normalizer_.transform(X)
            except ValueError:
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)

        X = self.outlier_remover_.transform(X)
        return X
