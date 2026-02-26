# tabtune/sampling/context_sampling.py

"""
Context (train-set) sampling / resampling utilities.

Why this exists:
- TabTune already has "resampling_strategy" inside DataProcessor, but that is aimed at
  imbalance resampling (SMOTE/over/under-sampling) and is not consistently applied when
  model-specific preprocessors are used.
- The "resampling notebook" strategies (uniform, stratified, balanced, kmeans diversity,
  hybrid, etc.) are better modeled as a *context selection* problem: choose a fixed-size
  training subset (context) for foundational/tabular ICL-style models.

This module implements a unified API:
    sample_context(X, y, task_type, strategy, context_size, strat_set, hybrid_ratio, seed, **kwargs)

and supports:
- classification: uniform/stratified/balanced/oversample_minority/smote/diversity_kmeans/hybrid_balanced_diverse
- regression: uniform/stratified (by y-quantiles)/balanced (by y-bins)/diversity_kmeans/hybrid_stratified_diverse

Notes:
- "strat_set" for regression => number of quantile bins.
- Hybrid ratio => fraction of context_size allocated to the "core" sampler, remainder to diversity sampler.
- When context_size > len(X), padding is done with replacement (configurable via allow_replacement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# -------------------------
# Helpers / config
# -------------------------

StrategyName = str


@dataclass
class ContextSamplingConfig:
    """
    Configuration for context sampling.

    Attributes:
        strategy: sampling strategy name.
        context_size: number of rows desired in the sampled context.
        strat_set: number of strata/bins (for regression quantile stratification).
        hybrid_ratio: fraction allocated to the "core" strategy for hybrid strategies.
        seed: random seed for deterministic sampling.
        allow_replacement: if True, pad with replacement when context_size > len(X).
        kmeans_centers: number of clusters to use for kmeans diversity strategy.
        min_pos: for binary classification, minimum positive examples to keep if possible.
        oversample_weight: for oversample_minority, multiplicative weight for minority class.
    """

    strategy: StrategyName = "uniform"
    context_size: int = 10_000
    strat_set: int = 10
    hybrid_ratio: float = 0.7
    seed: int = 42
    allow_replacement: bool = True

    # diversity knobs
    kmeans_centers: int = 2000

    # minority knobs
    min_pos: int = 50
    oversample_weight: float = 5.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def normalize_sampling_strategy_name(name: Optional[str]) -> str:
    """
    Normalizes user-provided strategy names.
    Supports aliases (including notebook-style '*_10k').

    Examples:
        "uniform_10k" -> "uniform"
        "hybrid_balanced_diverse_10k" -> "hybrid_balanced_diverse"
    """
    if not name:
        return "uniform"

    n = str(name).strip().lower()

    # strip notebook suffixes
    if n.endswith("_10k"):
        n = n[:-4]

    # common aliasing
    aliases = {
        "random": "uniform",
        "rand": "uniform",
        "strat": "stratified",
        "balanced_diverse": "hybrid_balanced_diverse",
        "hybrid": "hybrid_balanced_diverse",
        "diverse": "diversity_kmeans",
        "kmeans": "diversity_kmeans",
        "kmeans_diversity": "diversity_kmeans",
        "oversample": "oversample_minority",
        "oversample_pos": "oversample_minority",
        "smote_then": "smote",
    }
    return aliases.get(n, n)


def _ensure_series(y: Union[pd.Series, np.ndarray, list]) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y)


def _subset_by_indices(
    X: pd.DataFrame, y: pd.Series, idx: np.ndarray
) -> Tuple[pd.DataFrame, pd.Series]:
    Xs = X.iloc[idx].copy()
    ys = y.iloc[idx].copy()
    return Xs, ys


def _pad_or_trim_indices(
    idx: np.ndarray, n_desired: int, rng: np.random.Generator, allow_replacement: bool
) -> np.ndarray:
    if len(idx) == n_desired:
        return idx

    if len(idx) > n_desired:
        return rng.choice(idx, size=n_desired, replace=False)

    # Need to pad
    if not allow_replacement:
        # can't pad: return as-is (caller may warn)
        return idx

    extra = rng.choice(idx, size=n_desired - len(idx), replace=True)
    return np.concatenate([idx, extra])


# ============================================================================
# CRITICAL FIX: NaN Handling Utilities
# ============================================================================

def _simple_impute_for_sampling(X: pd.DataFrame) -> pd.DataFrame:
    """
    Quick and dirty imputation for sampling methods that don't handle NaN.
    
    This is a lightweight imputation ONLY for the purpose of clustering/SMOTE.
    The actual data preprocessing happens later in the pipeline.
    
    Strategy:
    - Numerical: fill with median
    - Categorical: fill with mode (most frequent)
    """
    X_imputed = X.copy()
    
    for col in X_imputed.columns:
        if X_imputed[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_imputed[col]):
                # Numerical: use median
                fill_value = X_imputed[col].median()
                if pd.isna(fill_value):  # All NaN column
                    fill_value = 0
                X_imputed[col] = X_imputed[col].fillna(fill_value)
            else:
                # Categorical: use mode
                mode_values = X_imputed[col].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else "MISSING"
                X_imputed[col] = X_imputed[col].fillna(fill_value)
    
    return X_imputed


# -------------------------
# Core sampling strategies
# -------------------------

def _uniform_indices(n: int, k: int, rng: np.random.Generator, allow_replacement: bool) -> np.ndarray:
    if k <= n:
        return rng.choice(n, size=k, replace=False)
    # pad
    if not allow_replacement:
        return np.arange(n)
    return rng.choice(n, size=k, replace=True)


def _classification_stratified_indices(
    y: pd.Series,
    k: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> np.ndarray:
    """
    Stratified sampling preserving class proportions (as much as possible).
    """
    y = _ensure_series(y)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)

    if k <= n:
        # desired per class proportional allocation
        proportions = counts / counts.sum()
        desired = np.floor(proportions * k).astype(int)

        # fix rounding to sum to k
        diff = k - desired.sum()
        if diff > 0:
            # distribute remaining to largest remainders
            remainders = proportions * k - desired
            for c in classes[np.argsort(-remainders)][:diff]:
                desired[np.where(classes == c)[0][0]] += 1

        idx_parts = []
        for c, dc in zip(classes, desired):
            cls_idx = np.where(y.values == c)[0]
            if dc <= len(cls_idx):
                idx_parts.append(rng.choice(cls_idx, size=dc, replace=False))
            else:
                if not allow_replacement:
                    idx_parts.append(cls_idx)
                else:
                    idx_parts.append(rng.choice(cls_idx, size=dc, replace=True))

        idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
        # In rare rounding edge cases, ensure exact size
        idx = _pad_or_trim_indices(idx, k, rng, allow_replacement=True)
        rng.shuffle(idx)
        return idx

    # k > n
    if not allow_replacement:
        return np.arange(n)

    # sample with replacement, preserving class distribution
    return rng.choice(n, size=k, replace=True, p=counts / counts.sum())


def _classification_balanced_indices(
    y: pd.Series,
    k: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> np.ndarray:
    """
    Balanced sampling: equal number from each class.
    """
    y = _ensure_series(y)
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes == 0:
        return np.array([], dtype=int)

    per = k // n_classes
    rem = k - per * n_classes

    idx_parts = []
    for i, c in enumerate(classes):
        cls_idx = np.where(y.values == c)[0]
        take = per + (1 if i < rem else 0)

        if take <= len(cls_idx):
            idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
        else:
            if not allow_replacement:
                idx_parts.append(cls_idx)
            else:
                idx_parts.append(rng.choice(cls_idx, size=take, replace=True))

    idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
    idx = _pad_or_trim_indices(idx, k, rng, allow_replacement=True)
    rng.shuffle(idx)
    return idx


def _regression_bins(y: pd.Series, n_bins: int) -> pd.Series:
    """
    Convert continuous y to quantile bins for stratification.
    """
    y = _ensure_series(y)
    # handle constant y
    if y.nunique(dropna=False) <= 1:
        return pd.Series(np.zeros(len(y), dtype=int), index=y.index)

    # qcut may fail on duplicates; use duplicates='drop'
    try:
        b = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        if b.isna().any():
            # fall back to cut
            b = pd.cut(y, bins=n_bins, labels=False)
        return b.astype("Int64").fillna(0).astype(int)
    except Exception:
        b = pd.cut(y, bins=n_bins, labels=False)
        return b.astype("Int64").fillna(0).astype(int)


def _regression_stratified_indices(
    y: pd.Series,
    k: int,
    n_bins: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> np.ndarray:
    bins = _regression_bins(y, n_bins)
    return _classification_stratified_indices(bins, k, rng, allow_replacement)


def _regression_balanced_indices(
    y: pd.Series,
    k: int,
    n_bins: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> np.ndarray:
    bins = _regression_bins(y, n_bins)
    return _classification_balanced_indices(bins, k, rng, allow_replacement)


def _oversample_minority_indices(
    y: pd.Series,
    k: int,
    rng: np.random.Generator,
    allow_replacement: bool,
    oversample_weight: float = 5.0,
    min_pos: int = 50,
) -> np.ndarray:
    """
    Oversample minority class(es) using weighted sampling with replacement.
    For multiclass: inverse-frequency weighting.
    """
    y = _ensure_series(y)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) <= 1:
        return _uniform_indices(n, k, rng, allow_replacement)

    # build weights: inverse frequency
    inv = 1.0 / np.maximum(counts, 1)
    base_w = inv / inv.sum()
    cls_w = dict(zip(classes, base_w))

    # boost minority further if binary
    if len(classes) == 2:
        # define minority
        minority = classes[np.argmin(counts)]
        cls_w[minority] *= float(oversample_weight)

        # renormalize
        s = sum(cls_w.values())
        cls_w = {c: w / s for c, w in cls_w.items()}

    weights = np.array([cls_w[val] for val in y.values], dtype=float)
    weights = weights / weights.sum()

    # draw indices with replacement to reach k
    idx = rng.choice(n, size=k, replace=True, p=weights)

    # try to enforce min_pos for binary
    if len(classes) == 2 and min_pos is not None:
        minority = classes[np.argmin(counts)]
        cur_min = np.sum(y.iloc[idx].values == minority)
        if cur_min < min_pos:
            # force-add minority examples
            min_idx = np.where(y.values == minority)[0]
            need = min_pos - cur_min
            forced = rng.choice(min_idx, size=need, replace=True)
            # replace some majority picks
            maj_pos = np.where(y.iloc[idx].values != minority)[0]
            if len(maj_pos) > 0:
                replace_pos = rng.choice(maj_pos, size=min(need, len(maj_pos)), replace=False)
                idx[replace_pos] = forced[: len(replace_pos)]

    return idx


def _smote_resample_then_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE/SMOTENC if available, then trims/pads to k.
    Only meaningful for classification tasks.
    """
    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
    except Exception:
        # imblearn not installed; fall back
        idx = _classification_balanced_indices(y, k, rng, allow_replacement)
        return _subset_by_indices(X, y, idx)

    # detect categorical columns (object/category/bool)
    cat_mask = [
        pd.api.types.is_object_dtype(X[col])
        or pd.api.types.is_categorical_dtype(X[col])
        or pd.api.types.is_bool_dtype(X[col])
        for col in X.columns
    ]
    cat_indices = [i for i, is_cat in enumerate(cat_mask) if is_cat]

    # encode categoricals as integers for SMOTENC
    enc_maps: Dict[str, Dict[Any, int]] = {}
    
    # === FIX: Impute NaN values before SMOTE ===
    X_work = _simple_impute_for_sampling(X)
    # encode categoricals as integers for SMOTENC
    enc_maps: Dict[str, Dict[Any, int]] = {}
    for col in X.columns:
        if pd.api.types.is_object_dtype(X_work[col]) or pd.api.types.is_categorical_dtype(X_work[col]):
            vals = X_work[col].astype("category")
            enc_maps[col] = {cat: i for i, cat in enumerate(vals.cat.categories)}
            X_work[col] = vals.cat.codes

    y_work = _ensure_series(y)

    if len(cat_indices) > 0:
        sm = SMOTENC(categorical_features=cat_indices, random_state=int(rng.integers(0, 2**31 - 1)))
    else:
        sm = SMOTE(random_state=int(rng.integers(0, 2**31 - 1)))

    X_res, y_res = sm.fit_resample(X_work.values, y_work.values)

    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)

    # decode categoricals
    for col, mp in enc_maps.items():
        inv = {v: k for k, v in mp.items()}
        X_res[col] = X_res[col].round().astype(int).map(inv).astype(object)

    # now sample exactly k
    n = len(X_res)
    idx = _uniform_indices(n, k, rng, allow_replacement)
    return X_res.iloc[idx].reset_index(drop=True), y_res.iloc[idx].reset_index(drop=True)


def _diversity_kmeans_indices(
    X: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
    allow_replacement: bool,
    kmeans_centers: int = 2000,
) -> np.ndarray:
    """
    KMeans-based diversity sampling:
    - do a lightweight numeric encoding (one-hot for categoricals)
    - run kmeans with min(kmeans_centers, n, k) clusters
    - pick one representative from each cluster
    - pad/trim to k
    """
    from sklearn.cluster import MiniBatchKMeans

    n = len(X)
    if n == 0:
        return np.array([], dtype=int)

    # if dataset is small, uniform is fine
    if n <= k:
        return _uniform_indices(n, k, rng, allow_replacement)

    # === FIX: Impute NaN values before clustering ===
    X_clean = _simple_impute_for_sampling(X)

    # basic encoding for clustering
    X_enc = pd.get_dummies(X_clean, drop_first=False).astype(float).values

    n_clusters = int(min(kmeans_centers, n, k))
    if n_clusters <= 1:
        return _uniform_indices(n, k, rng, allow_replacement)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(rng.integers(0, 2**31 - 1)),
        batch_size=min(1024, n),
        n_init="auto",
        max_iter=100,
    )
    labels = km.fit_predict(X_enc)

    # pick a representative per cluster
    reps = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        reps.append(rng.choice(members, size=1, replace=False)[0])

    idx = np.array(reps, dtype=int)
    idx = _pad_or_trim_indices(idx, k, rng, allow_replacement)
    rng.shuffle(idx)
    return idx


# -------------------------
# Public API
# -------------------------

def sample_context(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, list],
    task_type: str,
    strategy: Optional[str] = None,
    context_size: Optional[int] = None,
    strat_set: int = 10,
    hybrid_ratio: float = 0.7,
    seed: int = 42,
    allow_replacement: bool = True,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select a fixed-size context subset (X_ctx, y_ctx) from (X, y).

    Args:
        X: training features
        y: training target
        task_type: "classification" or "regression"
        strategy: sampling strategy
        context_size: desired context size (defaults to len(X) if None)
        strat_set: # strata/bins (used in regression stratified/balanced)
        hybrid_ratio: for hybrid strategies
        seed: random seed
        allow_replacement: whether to pad when context_size > len(X)
        kwargs: optional strategy-specific knobs:
            - kmeans_centers
            - min_pos
            - oversample_weight

    Returns:
        (X_sampled, y_sampled)
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided for context sampling.")

    y = _ensure_series(y)
    if context_size is None:
        context_size = len(X)
    context_size = int(context_size)

    st = normalize_sampling_strategy_name(strategy)

    rng = _rng(seed)
    task = str(task_type).lower().strip()

    # knobs
    kmeans_centers = int(kwargs.get("kmeans_centers", 2000))
    min_pos = int(kwargs.get("min_pos", 50))
    oversample_weight = float(kwargs.get("oversample_weight", 5.0))

    if context_size <= 0:
        return X.iloc[[]].copy(), y.iloc[[]].copy()

    # --- Hybrid strategies ---
    # classification: hybrid_balanced_diverse => balanced core + diversity remainder
    # regression: hybrid_stratified_diverse => stratified core + diversity remainder
    if st in ("hybrid_balanced_diverse", "hybrid_stratified_diverse"):
        ratio = float(hybrid_ratio)
        ratio = max(0.0, min(1.0, ratio))
        k_core = int(round(context_size * ratio))
        k_div = context_size - k_core

        if task == "classification":
            core_idx = _classification_balanced_indices(y, k_core, rng, allow_replacement)
        else:
            core_idx = _regression_stratified_indices(y, k_core, strat_set, rng, allow_replacement)

        # diversity on remaining pool, but keep it simple:
        # run diversity on the full X then exclude core indices if possible.
        div_idx = _diversity_kmeans_indices(
            X, max(k_div, 0), rng, allow_replacement, kmeans_centers=kmeans_centers
        )

        # exclude overlaps if feasible
        if len(div_idx) > 0 and len(core_idx) > 0:
            core_set = set(core_idx.tolist())
            div_idx = np.array([i for i in div_idx if i not in core_set], dtype=int)
            div_idx = _pad_or_trim_indices(div_idx, k_div, rng, allow_replacement)

        idx = np.concatenate([core_idx, div_idx]) if k_div > 0 else core_idx
        idx = _pad_or_trim_indices(idx, context_size, rng, allow_replacement)
        rng.shuffle(idx)
        return _subset_by_indices(X, y, idx)

    # --- Plain strategies ---
    if st == "uniform":
        idx = _uniform_indices(len(X), context_size, rng, allow_replacement)
        return _subset_by_indices(X, y, idx)

    if st == "stratified":
        if task == "classification":
            idx = _classification_stratified_indices(y, context_size, rng, allow_replacement)
        else:
            idx = _regression_stratified_indices(y, context_size, strat_set, rng, allow_replacement)
        return _subset_by_indices(X, y, idx)

    if st == "balanced":
        if task == "classification":
            idx = _classification_balanced_indices(y, context_size, rng, allow_replacement)
        else:
            idx = _regression_balanced_indices(y, context_size, strat_set, rng, allow_replacement)
        return _subset_by_indices(X, y, idx)

    if st in ("oversample_minority", "oversample_minority_then"):
        # classification only (regression -> uniform)
        if task != "classification":
            idx = _uniform_indices(len(X), context_size, rng, allow_replacement)
            return _subset_by_indices(X, y, idx)

        idx = _oversample_minority_indices(
            y,
            context_size,
            rng,
            allow_replacement=True,
            oversample_weight=oversample_weight,
            min_pos=min_pos,
        )
        return _subset_by_indices(X, y, idx)

    if st == "smote":
        if task != "classification":
            idx = _uniform_indices(len(X), context_size, rng, allow_replacement)
            return _subset_by_indices(X, y, idx)
        return _smote_resample_then_subsample(X, y, context_size, rng, allow_replacement)

    if st in ("diversity_kmeans", "diversity"):
        idx = _diversity_kmeans_indices(
            X, context_size, rng, allow_replacement, kmeans_centers=kmeans_centers
        )
        return _subset_by_indices(X, y, idx)

    # fallback
    idx = _uniform_indices(len(X), context_size, rng, allow_replacement)
    return _subset_by_indices(X, y, idx)
