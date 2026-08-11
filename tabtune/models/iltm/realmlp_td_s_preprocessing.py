# Adapted rom: https://github.com/dholzmueller/realmlp-td-s_standalone/blob/main/preprocessing.py
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer


def to_numeric_coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Top-level function so the pipeline is picklable."""
    return df.apply(pd.to_numeric, errors='coerce')


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ordinal_enc_ = OrdinalEncoder(unknown_value=np.nan, encoded_missing_value=np.nan,
                                           handle_unknown='use_encoded_value',
                                           dtype=np.float32)
        self.ordinal_enc_.fit(X)
        self.cat_sizes_ = []
        for cat_arr in self.ordinal_enc_.categories_:
            has_nan = np.any([isinstance(val, (float, np.float32, np.float64)) and np.isnan(val) for val in cat_arr])
            self.cat_sizes_.append(len(cat_arr) - int(has_nan))
        # mark fitted for sklearn checks
        self.is_fitted_ = True
        try:
            self.n_features_in_ = X.shape[1]
        except Exception:
            pass
        return self

    def transform(self, X, y=None):
        x_enc = self.ordinal_enc_.transform(X)
        n_samples = x_enc.shape[0]
        out_arrs = []
        for i, cat_size in enumerate(self.cat_sizes_):
            column = x_enc[:, i]
            idxs = np.arange(n_samples)
            isnan = np.isnan(column)
            out_arr = np.zeros(shape=(n_samples, cat_size), dtype=np.int64)
            # do one-hot encoding, encode nan (missing or unknown) values to all zeros
            out_arr[idxs[~isnan], column[~isnan].astype(np.int64)] = 1#.
            if cat_size == 2:
                # binary: encode to single feature being -1, 1 or 0 (for missing or unknown values)
                out_arr = out_arr[:, 0:1] - out_arr[:, 1:2]
            out_arrs.append(out_arr)
        concatenated = np.concatenate(out_arrs, axis=-1)
        return concatenated


class CustomOneHotPipeline(BaseEstimator, TransformerMixin):
    """
    Apply CustomOneHotEncoder only to specified categorical features.
    """

    def __init__(self, cat_features: list[int], max_cat_size: int = 12):
        """
        Initialize the pipeline with specified categorical feature indices.

        Parameters:
        - cat_features: list of int
            List of column indices corresponding to categorical features.
        """
        self.cat_features = cat_features
        self.max_cat_size = max_cat_size

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # Map column indices to column names
        cat_column_names = X.columns[self.cat_features].tolist()
        # Fit an initial OrdinalEncoder to get category sizes
        initial_ordinal_enc = OrdinalEncoder(
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
            handle_unknown='use_encoded_value',
            dtype=np.float32
        )
        initial_ordinal_enc.fit(X[cat_column_names])
        # Determine the number of categories per feature
        cat_sizes = []
        for cat_arr in initial_ordinal_enc.categories_:
            has_nan = np.any([isinstance(val, (float, np.float32, np.float64)) and np.isnan(val) for val in cat_arr])
            cat_sizes.append(len(cat_arr) - int(has_nan))
        
        # Split the categorical columns based on max_cat_size
        one_hot_columns = [
            col for col, size in zip(cat_column_names, cat_sizes) if size <= self.max_cat_size
        ]
        other_cat_columns = [
            col for col, size in zip(cat_column_names, cat_sizes) if size > self.max_cat_size
        ]
        # Identify remaining numerical columns by excluding categorical columns
        remaining_columns = [c for c in X.columns if c not in cat_column_names]
        # Define a pipeline for numerical columns to convert them to numeric types
        numerical_pipeline = Pipeline([
            ('to_numeric', FunctionTransformer(to_numeric_coerce, validate=False))
        ])
        # Define transformers based on the split
        transformers = []
        
        if one_hot_columns:
            transformers.append(('one_hot', CustomOneHotEncoder(), one_hot_columns))
        
        if other_cat_columns:
            # You can choose to apply a different encoder here, e.g., OrdinalEncoder or embeddings
            # For simplicity, we'll use OrdinalEncoder
            transformers.append(('ordinal', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan,
                dtype=np.float32
            ), other_cat_columns))
        
        if remaining_columns:
            transformers.append(('numerical', numerical_pipeline, remaining_columns))
        
        # Initialize the ColumnTransformer with the specified transformers
        self.tfm_ = ColumnTransformer(transformers=transformers)
        self.tfm_.fit(X)
        # mark fitted for sklearn checks
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        transformed = self.tfm_.transform(X)
        return transformed


class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    """
    Robust center + IQR scale + smooth clip, with median imputation.
    Any NaN at transform time is filled with the training median so the output is 0.
    Columns that are constant or all-NaN produce all zeros.
    """
    def __init__(self, clip_divisor: float = 3.0):
        self.clip_divisor = float(clip_divisor)

    def fit(self, X, y=None):
        # ndarray only
        assert isinstance(X, np.ndarray) and X.ndim == 2

        # NaN-aware statistics across samples (axis -2 == 0 for 2D)
        med = np.nanmedian(X, axis=-2)
        q75 = np.nanquantile(X, 0.75, axis=-2)
        q25 = np.nanquantile(X, 0.25, axis=-2)
        iqr = q75 - q25

        # fallback width when IQR is zero or NaN
        width = 0.5 * (np.nanmax(X, axis=-2) - np.nanmin(X, axis=-2))

        # choose denominator: prefer IQR, otherwise width
        use_iqr = np.isfinite(iqr) & (iqr != 0.0)
        denom = np.where(use_iqr, iqr, width)

        # finalize stats
        med = np.where(np.isfinite(med), med, 0.0)          # all-NaN columns -> median 0
        factors = np.empty_like(denom, dtype=np.float64)

        good = np.isfinite(denom) & (denom != 0.0)
        factors[good] = 1.0 / (denom[good] + 1e-30)
        factors[~good] = 0.0                                 # constant or undefined -> zero out

        self._median = med.astype(np.float32)
        self._factors = factors.astype(np.float32)
        # mark fitted for sklearn checks
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None, chunk_size: int = 10000):
        assert isinstance(X, np.ndarray) and X.ndim == 2

        n_samples, n_features = X.shape
        out = np.empty((n_samples, n_features), dtype=np.float32)
        m = self._median
        f = self._factors
        d = np.float32(self.clip_divisor)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)

            # process in-place in float32
            chunk = np.asarray(X[start:end], dtype=np.float32)

            # impute NaNs to training medians so they map to 0 after scaling
            if np.isnan(chunk).any():
                r, c = np.where(np.isnan(chunk))
                if r.size:
                    chunk[r, c] = m[c]

            x_scaled = f[None, :] * (chunk - m[None, :])
            out[start:end] = x_scaled / np.sqrt(1.0 + (x_scaled / d) ** 2)

        return out


def get_realmlp_td_s_pipeline(cat_features):
    pipeline = sklearn.pipeline.Pipeline([
        ('one_hot', CustomOneHotPipeline(cat_features=cat_features)),
        ('rssc', RobustScaleSmoothClipTransform())
    ])
    
    return pipeline


class RealMLPTDSepPipeline(BaseEstimator, TransformerMixin):
    """
    Wrap the existing realmlp-td-s pipeline but at transform-time
    split its output into (x_num, x_cat).
    """
    def __init__(self, cat_features, max_cat_size: int = 12):
        self.cat_features = [] if cat_features is None else cat_features
        self.max_cat_size = max_cat_size
        # delegate to the original pipeline
        self._pipe = sklearn.pipeline.Pipeline([
            ('one_hot', CustomOneHotPipeline(cat_features=self.cat_features,
                                             max_cat_size=max_cat_size)),
            ('rssc',    RobustScaleSmoothClipTransform())
        ])

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # 1) fit the combined pipeline
        self._pipe.fit(X_df, y)
        # Explicitly mark inner pipeline as fitted so sklearn’s transform guard passes
        setattr(self._pipe, "is_fitted_", True)
        # also mirror common fit markers on the wrapper itself
        self.is_fitted_ = True
        self.n_features_in_ = X_df.shape[1]
        # 2) figure out how many *categorical* columns it is producing
        #    by inspecting the ColumnTransformer inside CustomOneHotPipeline
        ct = self._pipe.named_steps['one_hot'].tfm_
        cat_dim = 0
        if hasattr(ct, "output_indices_"):
            idxs = ct.output_indices_
            for key in ("one_hot", "ordinal"):
                if key in idxs:
                    sl = idxs[key]
                    cat_dim += (sl.stop - sl.start) if isinstance(sl, slice) else len(sl)
        else:
            # Fallback if sklearn version lacks output_indices_
            for name, transformer, cols in ct.transformers_:
                if name == "one_hot":
                    sizes = transformer.cat_sizes_
                    cat_dim += sum(1 if s == 2 else s for s in sizes)
                elif name == "ordinal":
                    cat_dim += len(cols)

        self._cat_dim = int(cat_dim)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        # self._pipe has is_fitted_ set in fit, so this is safe on sklearn>=1.7
        full = self._pipe.transform(X_df)
        # split!
        x_cat = full[:, :self._cat_dim]
        x_num = full[:, self._cat_dim:]
        return x_num, x_cat


def get_realmlp_td_s_pipeline_separated(cat_features, max_cat_size: int = 12):
    """
    Returns a transformer whose transform(X) gives (x_num, x_cat)
    instead of the single concatenated array.
    """
    return RealMLPTDSepPipeline(cat_features, max_cat_size)