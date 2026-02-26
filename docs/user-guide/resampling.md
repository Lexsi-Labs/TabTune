## 4. Resampling

TabTune includes a unified **context sampling / resampling** module:

**File:** `tabtune/resampling/context_sampling.py`  
**Purpose:** choose a fixed-size **training subset (context)** for ICL-style / foundation models using strategies beyond simple random sampling.

### 4.1 Why this module exists

The module exists because:
- TabTune already has a generic `resampling_strategy` in some data processors (often aimed at imbalance resampling),
- but model-specific preprocessors and ICL-style pipelines may not consistently apply those,
- and for LimiX-like models the right abstraction is **context selection**, not just “rebalance the whole dataset”.

### 4.2 Public API

```python
sample_context(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, list],
    task_type: str,                    # "classification" or "regression"
    strategy: Optional[str] = None,    # e.g. "uniform", "balanced", "diversity_kmeans", ...
    context_size: Optional[int] = None,
    strat_set: int = 10,               # bins for regression stratification/balancing
    hybrid_ratio: float = 0.7,         # split between "core" and "diverse" parts
    seed: int = 42,
    allow_replacement: bool = True,
    **kwargs                           # kmeans_centers, min_pos, oversample_weight, ...
) -> Tuple[pd.DataFrame, pd.Series]
```

**Key behavior:**
- Returns `(X_ctx, y_ctx)` — a sampled context dataset.
- If `context_size > len(X)`:
  - by default it **pads with replacement** (`allow_replacement=True`).
- Normalizes strategy names (e.g. notebook-style suffixes like `_10k` are stripped).

### 4.3 Strategy Names and Behavior

#### Core strategies (classification)
- **`uniform`**: random subset of rows.
- **`stratified`**: preserve class proportions.
- **`balanced`**: equal rows per class (pads with replacement if needed).
- **`oversample_minority`**: weighted sampling to oversample minority classes.
- **`smote`**: uses SMOTE/SMOTENC (if `imblearn` installed), then subsamples to `context_size`.
- **`diversity_kmeans`**: diversity sampling by clustering in feature space (MiniBatchKMeans), then selecting representatives.
- **`hybrid_balanced_diverse`**: balanced core + kmeans-diverse remainder.

#### Core strategies (regression)
Regression uses binning of `y` into quantile bins (via `qcut` with safe fallbacks):
- **`uniform`**
- **`stratified`**: stratify by quantile bins of `y`.
- **`balanced`**: balance across quantile bins of `y`.
- **`diversity_kmeans`**
- **`hybrid_stratified_diverse`**: stratified core + kmeans-diverse remainder.

### 4.4 Strategy Aliases

`normalize_sampling_strategy_name()` supports common aliases:
- `random`, `rand` → `uniform`
- `strat` → `stratified`
- `diverse`, `kmeans` → `diversity_kmeans`
- `hybrid`, `balanced_diverse` → `hybrid_balanced_diverse`
- Notebook suffix stripping: `uniform_10k` → `uniform`, `hybrid_balanced_diverse_10k` → `hybrid_balanced_diverse`

### 4.5 Important implementation details

#### (A) NaN handling (critical fix)

Some strategies (SMOTE, KMeans clustering) do not handle NaNs.  
This module adds `_simple_impute_for_sampling(X)`:

- Numerical columns → fill with **median** (or 0 if all-NaN)
- Categorical columns → fill with **mode** (or `"MISSING"`)

This imputation is **only** for sampling/clustering. The real preprocessing still happens later in the TabTune pipeline.

#### (B) KMeans diversity sampling

`diversity_kmeans`:
- imputes NaNs (see above),
- one-hot encodes categoricals (`pd.get_dummies`),
- runs `MiniBatchKMeans` with `n_clusters = min(kmeans_centers, n, k)`,
- selects one representative per cluster,
- pads/trims to exact `k`.

Useful when:
- you want **coverage** of different regions of feature space,
- especially on large datasets where random sampling can miss minority modes.

#### (C) Hybrid strategies

Hybrid strategies split the context into:
- **core sampler**: balanced (classification) or stratified (regression)
- **diversity sampler**: kmeans-based diversity

Allocation:
- `k_core = round(context_size * hybrid_ratio)`
- `k_div  = context_size - k_core`

Overlaps are removed where possible; remaining slots are padded if needed.

---

## 5. How to use Resampling in TabTune

### 5.1 Recommended patterns

**If you just want a fast baseline:**
- `uniform` with a moderate `context_size`

**If you have class imbalance:**
- `balanced` or `oversample_minority`

**If you want broader coverage / robustness:**
- `diversity_kmeans` (or hybrid)

**If you want both calibration + coverage:**
- `hybrid_balanced_diverse` (classification)
- `hybrid_stratified_diverse` (regression)

### 5.2 Example: sample a context before fitting

```python
import pandas as pd
from tabtune.resampling.context_sampling import sample_context

# X_train: pd.DataFrame, y_train: pd.Series
X_ctx, y_ctx = sample_context(
    X_train,
    y_train,
    task_type="classification",
    strategy="hybrid_balanced_diverse",
    context_size=10_000,
    hybrid_ratio=0.7,
    seed=42,
    kmeans_centers=2000
)

# Then feed (X_ctx, y_ctx) to your TabTune pipeline/model
```

### 5.3 Example: regression stratification by y-quantiles

```python
X_ctx, y_ctx = sample_context(
    X_train,
    y_train,
    task_type="regression",
    strategy="stratified",
    context_size=8_000,
    strat_set=10,   # 10 quantile bins
    seed=7
)
```

## 7. Troubleshooting

### Issue: “KMeans sampling is slow”
**What to do:**
- Lower `kmeans_centers` (e.g. from 2000 → 500)
- Use `uniform` or `stratified` if diversity is not crucial

```python
X_ctx, y_ctx = sample_context(
    X_train, y_train,
    task_type="classification",
    strategy="diversity_kmeans",
    context_size=10_000,
    kmeans_centers=500
)
```

### Issue: “SMOTE fails or imblearn not installed”
**Behavior:**
- The module automatically falls back to **balanced** sampling if `imblearn` isn’t available.

### Issue: “NaNs break sampling”
**Expected:**
- SMOTE and KMeans paths already apply a lightweight imputation via `_simple_impute_for_sampling()`.

---

## 8. Quick Reference

| Goal | Task | Suggested Strategy |
|------|------|--------------------|
| Fast baseline | cls/reg | `uniform` |
| Preserve label shape | classification | `stratified` |
| Fix imbalance | classification | `balanced` / `oversample_minority` |
| Boost minority realism | classification | `smote` (if available) |
| Cover diverse modes | cls/reg | `diversity_kmeans` |
| Robust default | classification | `hybrid_balanced_diverse` |
| Robust default | regression | `hybrid_stratified_diverse` |

---

## 9. Next Steps

- Add context sampling as a configurable knob in your LimiX pipeline config (strategy + context_size).
- Benchmark `uniform` vs `hybrid_*` on your target datasets.
- For very large data, consider reducing `kmeans_centers` and using hybrid sampling for a strong speed/quality tradeoff.

---
