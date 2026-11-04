# TabTune Regression Data Processing

This document explains **how TabTune processes regression data** end-to-end: feature preprocessing, target scaling, and model-specific regression processors.

It focuses on the code paths used when `task_type="regression"`.

---

## 1. Overview

In regression mode, TabTune uses a two-layer processing stack:

1. **`DataProcessor`** (`tabtune/Dataprocess/data_processor.py`)
   - orchestrates feature transforms
   - selects and fits a **regression processor** for `y`
   - routes to model-specific preprocessors when needed

2. **Regression processors** (`tabtune/Dataprocess/regression/*`)
   - implement target scaling / transforms
   - implement model-specific regression target encoding where needed

---

## 2. Regression Target Scaling

### 2.1 `RegressionDataProcessor`

Base target scaling is implemented in:

- `tabtune/Dataprocess/regression/base_processor.py`

Supported `target_scaling_strategy`:

| Strategy | Description | When to use |
|---|---|---|
| `standard` | z-score scaling | strong default |
| `minmax` | maps y into [0, 1] | bounded targets |
| `robust` | robust scaling vs outliers | heavy-tailed y |
| `power_transform` | stabilize variance / normalize | skewed y |
| `none` | no scaling | already-normalized targets |

Example:

```python
pipeline = TabularPipeline(
    model_name="Mitra",
    task_type="regression",
    processor_params={
        "target_scaling_strategy": "robust"
    }
)
```

### 2.2 Inverse Transform (Predictions)

Regression processors support `inverse_transform()` so that:
- the model trains on scaled targets
- predictions are returned in the **original y space**

If you bypass pipeline APIs, ensure you inverse-transform predictions before reporting.

---

## 3. Model-Specific Regression Processors

`DataProcessor` selects the correct regression processor via a small factory:

- `tabtune/Dataprocess/data_processor.py` → `_get_regression_processor()`

### 3.1 Available Processor Implementations

| Model | Processor |
|---|---|
| TabPFN | `TabPFNRegressionProcessor` |
| ContextTab | `ContextTabRegressionProcessor` |
| TabDPT | `TabDPTRegressionProcessor` |
| Mitra | `MitraRegressionProcessor` |
| Limix | `LimixRegressionProcessor` |
| fallback | `RegressionDataProcessor` |

These live in `tabtune/Dataprocess/regression/`.

### 3.2 ContextTab Regression: Binning & Regression Type

ContextTab regression processor supports additional config:

- `regression_type` (default: `l2`)
- `num_regression_bins` (default: `16`)

These are forwarded from `model_params` and used in ContextTab preprocessor/processor construction.

Example:

```python
pipeline = TabularPipeline(
    model_name="ContextTab",
    task_type="regression",
    model_params={
        "regression_type": "l2",
        "num_regression_bins": 32
    }
)
```

---

## 4. Feature Processing in Regression Mode

Regression mode reuses much of TabTune’s standard feature preprocessing stack:

- numeric scaling
- categorical encoding (ordinal/one-hot depending on model/preprocessor)
- missing value handling

Key rule:
- If a **custom preprocessor** is selected for a model, TabTune delegates the feature transform to it.
- Otherwise, TabTune applies standard transforms inferred from column types.

---

## 5. Practical Recommendations

### 5.1 Choose a Target Scaling Strategy

- Start with `standard`
- Use `robust` for outlier-prone targets
- Use `power_transform` for strong skew / nonlinear target distributions
- Use `none` only if you know the model expects raw y

### 5.2 Keep y as 1D

Internally, TabTune reshapes y to fit sklearn scalers.  
If you pass a `DataFrame` with multiple regression targets, behavior may be undefined unless the model explicitly supports it.

---

## 6. Troubleshooting

### Issue: “y has wrong shape”
**Fix:** pass a 1D `pd.Series` or 1D numpy array for y.

### Issue: “Inverse transform fails”
**Cause:** regression processor not fitted.  
**Fix:** ensure `pipeline.fit(X_train, y_train)` ran successfully before calling `predict()`.

---

## 7. Next Steps

- For regression metrics & evaluation behavior: `regression.md`
- For benchmarking regression suites: `regression_benchmarking.md`
