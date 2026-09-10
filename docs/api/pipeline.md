# API: TabularPipeline

Complete API reference for the `TabularPipeline` class—the main entry point for TabTune.

::: tabtune.TabularPipeline.pipeline.TabularPipeline
    options:
      show_source: true

---

## Overview

`TabularPipeline` provides a scikit-learn-compatible interface for training and using tabular foundation models. It coordinates data preprocessing, model initialization, training, and inference.

---

## Constructor

### `TabularPipeline.__init__()`

```python
TabularPipeline(
    model_name: str,
    task_type: str = 'classification',
    tuning_strategy: str = 'inference',
    tuning_params: dict | None = None,
    processor_params: dict | None = None,
    model_params: dict | None = None,
    model_checkpoint_path: str | None = None,
    finetune_mode: str | None = None,
    *,
    cache: str | bool | None = None,        # new in 0.2.0
    envelope_mode: str = 'warn',            # new in 0.2.0
    license_mode: str = 'research',         # new in 0.2.0
    validate: bool = True,                  # new in 0.2.0
)
```

#### Parameters

**`model_name`** (str, required)
- Model name or alias. **16 models** are registered.
- Resolution ignores case, hyphens, underscores, dots and whitespace, so `'TabPFN-v2.6'` and `'tabpfnv26'` are equivalent.
- Canonical names: `'TabPFN'`, `'TabPFNv26'`, `'TabPFNv3'`, `'TabICL'`, `'TabICLv2'`, `'OrionMSP'`, `'OrionMSPv1.5'`, `'OrionBix'`, `'Mitra'`, `'ContextTab'`, `'TabDPT'`, `'Limix'`, `'TabFM'`, `'XRFM'`, `'ILTM'`, `'EXAONETabular'`
- Do not hardcode this list — call `tabtune.registry.list_model_names()`.
- Example: `model_name="TabICLv2"`

**`task_type`** (str, default: `'classification'`)
- Type of machine learning task.
- Supported: `'classification'`, `'regression'`
- Regression is supported for: `TabPFN`, `TabPFNv26`, `TabPFNv3`, `TabICLv2`, `Mitra`, `ContextTab`, `TabDPT`, `Limix`, `TabFM`, `XRFM`, `ILTM`, `EXAONETabular`
- Classification-only: `TabICL`, `OrionMSP`, `OrionMSPv1.5`, `OrionBix`
- Query it instead of memorising it: `[s.name for s in list_models(task="regression")]`
- Example: `task_type="regression"`

**`tuning_strategy`** (str, default: `'inference'`)
- Training/fine-tuning strategy to use.
- Options:
  - `'inference'`: Zero-shot predictions (no training)
  - `'finetune'`: Full fine-tuning of all parameters
  - `'peft'`: Parameter-efficient fine-tuning with LoRA adapters
- Example: `tuning_strategy="peft"`

**`tuning_params`** (dict, optional)
- Hyperparameters for training/inference.
- Common parameters for all models:
  - `device` (str): `'cuda'` or `'cpu'` (default: auto-detected)
  - `epochs` (int): Number of training epochs
  - `learning_rate` (float): Learning rate for optimizer
  - `batch_size` (int): Batch size for training
  - `show_progress` (bool): Show tqdm progress bar
  - `save_checkpoint_path` (str): Path to save fine-tuned weights
  - `checkpoint_dir` (str): Directory for automatic checkpoint saving
  - `weight_decay` (float): AdamW weight decay
  - `clip_grad_norm` (float): Gradient clipping norm
- Episodic training parameters (meta-learning / TBT):
  - `support_size` (int): Support set size per episode
  - `query_size` (int): Query set size per episode
  - `n_episodes` (int): Episodes per epoch (classification)
  - `steps_per_epoch` (int): Steps per epoch (regression TBT)
  - `context_size` (int): Alias for `support_size` (regression TBT)
- PEFT parameters:
  - `peft_config` (dict): LoRA configuration — `r`, `lora_alpha`, `lora_dropout`, `target_modules`

- Example:
  ```python
  tuning_params={
      "device": "cuda",
      "epochs": 5,
      "learning_rate": 2e-5,
      "batch_size": 8
  }
  ```

**`processor_params`** (dict, optional)
- Parameters for data preprocessing.
- Common parameters:
  - `imputation_strategy` (str): `'mean'`, `'median'`, `'mode'`, `'knn'`
  - `scaling_strategy` (str): `'standard'`, `'minmax'`, `'robust'`
  - `categorical_encoding` (str): Encoding method (auto-selected for model-specific)
  - `resampling_strategy` (str): `'smote'`, `'random_oversample'`, etc.
- Example:
  ```python
  processor_params={
      "imputation_strategy": "median",
      "scaling_strategy": "standard"
  }
  ```

**`model_params`** (dict, optional)
- Direct parameters passed to the model constructor.
- Model-specific (see individual model documentation).
- Example for TabICL:
  ```python
  model_params={"n_estimators": 16, "softmax_temperature": 0.9}
  ```

**`model_checkpoint_path`** (str, optional)
- Path to a pre-trained model checkpoint (`.pt` file).
- If provided, loads weights from checkpoint instead of default pre-trained weights.
- Example: `model_checkpoint_path="./checkpoints/tabicl_epoch5.pt"`

**`finetune_mode`** (str, default: `'meta-learning'`)
- Fine-tuning mode for models that support it.
- Options:
  - `'meta-learning'`: Episodic meta-learning (default)
  - `'sft'`: Standard supervised fine-tuning
- Example: `finetune_mode="sft"`

#### Keyword-only parameters (new in 0.2.0)

**`cache`** (str | bool | None, default: `None`)
- Prediction cache: `'memory'`, `'disk'`, `None`, or a `PredictionCache` instance.
- Enabling it collapses `evaluate()`'s three redundant forward passes into one.
- Entries are keyed on a fingerprint covering the fitted model *and* the input data, so refitting or changing the data invalidates automatically.
- Inspect with `pipeline.cache.stats`; clear with `pipeline.clear_cache()`.
- See [Prediction Caching](../user-guide/caching.md).

**`envelope_mode`** (str, default: `'warn'`)
- How to treat data outside the model's documented limits: `'error'`, `'warn'`, `'ignore'`.
- Architectural limits (`max_classes`, `min_rows`) always raise unless this is `'ignore'`.
- Resource limits (`max_rows`, `max_features`, `max_cells`) warn under `'warn'`.

**`license_mode`** (str, default: `'research'`)
- `'research'` — no licence enforcement.
- `'commercial'` — raise `LicenseError` on weights that forbid commercial use.
- `'ignore'` — skip the check entirely.
- Unverified licences warn under `'commercial'` rather than blocking.

**`validate`** (bool, default: `True`)
- Check model / task / strategy against the registry **before** loading weights.
- Set `False` to use a model TabTune does not know about.

```python
pipeline = TabularPipeline(
    model_name="TabICLv2",
    tuning_strategy="finetune",
    cache="disk",
    envelope_mode="error",
    license_mode="commercial",
)
print(pipeline.cache.stats)   # hits / misses / stores / hit_rate
```

See [Model Registry](../user-guide/registry.md).

#### Raises

| Exception | When |
|---|---|
| `ModelNotFoundError` | `model_name` does not resolve to a registered model |
| `UnsupportedTaskError` | Model has no head for `task_type` |
| `UnsupportedStrategyError` | Model does not implement `tuning_strategy` |
| `EnvelopeError` | Data violates a hard architectural limit |
| `LicenseError` | Weight licence forbids the intended use under `license_mode` |

All derive from `tabtune.registry.TabTuneError`.

#### Returns

Returns a `TabularPipeline` instance (not yet fitted).

---

## Core Methods

### `.fit(X, y)`

Train the pipeline on training data.

```python
pipeline.fit(X_train: pd.DataFrame, y_train: pd.Series) -> TabularPipeline
```

#### Parameters

- **`X`** (pd.DataFrame): Training features
- **`y`** (pd.Series): Training labels

#### Returns

Returns `self` (allows method chaining).

#### What it does

1. Fits the `DataProcessor` on training data (learns preprocessing transformations)
2. Applies preprocessing to training data
3. Initializes the model (if late initialization required)
4. Trains the model using `TuningManager` (if strategy != `'inference'`)

#### Example

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="finetune"
)
pipeline.fit(X_train, y_train)
```

---

### `.predict(X)`

Generate predictions on new data.

```python
predictions = pipeline.predict(X_test: pd.DataFrame) -> np.ndarray
```

#### Parameters

- **`X`** (pd.DataFrame): Features for prediction

#### Returns

- **`predictions`** (np.ndarray): Predicted class labels (shape: `(n_samples,)`)

#### Notes

- Automatically applies learned preprocessing
- Converts class indices back to original label format
- Must call `.fit()` before `.predict()`

#### Example

```python
predictions = pipeline.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
print(f"Unique classes: {np.unique(predictions)}")
```

---

### `.predict_proba(X)`

Get probability predictions for classification.

```python
probabilities = pipeline.predict_proba(X_test: pd.DataFrame) -> np.ndarray
```

#### Parameters

- **`X`** (pd.DataFrame): Features for prediction

#### Returns

- **`probabilities`** (np.ndarray): Class probabilities (shape: `(n_samples, n_classes)`)

#### Notes

- Each row sums to 1.0
- Column order matches label encoder classes
- Required for ROC AUC calculation

#### Example

```python
probabilities = pipeline.predict_proba(X_test)
print(f"Probabilities shape: {probabilities.shape}")
print(f"Row sums: {probabilities.sum(axis=1)}")  # Should be ~1.0
```

---

### `.evaluate(X, y, output_format='rich')`

Evaluate model performance on test data.

```python
metrics = pipeline.evaluate(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_format: str = 'rich'
) -> dict
```

#### Parameters

- **`X`** (pd.DataFrame): Test features
- **`y`** (pd.Series): True labels
- **`output_format`** (str): `'rich'` (formatted console output) or `'json'` (dict only)

#### Returns

- **`metrics`** (dict): Dictionary with evaluation metrics:
  - `accuracy` (float): Overall accuracy
  - `roc_auc_score` (float): ROC AUC (binary/multi-class)
  - `f1_score` (float): Weighted F1 score
  - `precision` (float): Weighted precision
  - `recall` (float): Weighted recall
  - `mcc` (float): Matthews Correlation Coefficient

#### Example

```python
metrics = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC AUC: {metrics['roc_auc_score']:.4f}")
```

---

### `.save(file_path)`

Save the entire pipeline to disk.

```python
pipeline.save(file_path: str) -> None
```

#### Parameters

- **`file_path`** (str): Path to save pipeline (typically `.joblib` extension)

#### What it saves

- DataProcessor state (preprocessing transformations)
- Model weights and state
- Configuration (model_name, strategy, params)
- Label encoders

#### Notes

- Must call `.fit()` before saving
- Uses `joblib` for serialization
- Large files (includes model weights)

#### Example

```python
pipeline.fit(X_train, y_train)
pipeline.save("my_pipeline.joblib")
```

---

### `.load(file_path)` (classmethod)

Load a saved pipeline from disk.

```python
loaded_pipeline = TabularPipeline.load(file_path: str) -> TabularPipeline
```

#### Parameters

- **`file_path`** (str): Path to saved pipeline file

#### Returns

- **`TabularPipeline`**: Loaded pipeline instance (already fitted)

#### Example

```python
loaded_pipeline = TabularPipeline.load("my_pipeline.joblib")
predictions = loaded_pipeline.predict(X_new)
```

---

## Additional Methods

### `.uncertainty_report(X_test, y_test, *, X_cal=None, y_cal=None, alpha=0.1, n_bins=15, method='lac')`

*New in 0.2.0.* One call for calibration **and** conformal coverage diagnostics.

```python
report = pipeline.uncertainty_report(
    X_test, y_test,
    X_cal=X_cal, y_cal=y_cal,
    alpha=0.1,          # 90% target coverage
    n_bins=15,          # calibration bins
    method='lac',       # 'lac' | 'aps'
) -> dict
```

Returns `ece`, `mce`, `brier`, `coverage`, `avg_set_size` and `sscs` (size-stratified
coverage — the worst-covered stratum). Omitting `X_cal` / `y_cal` gives the calibration
metrics only.

!!! danger "The calibration split must be disjoint from training"
    For an in-context model the training data *is* the support set. A re-used training frame
    is detected by fingerprint and **raises** rather than silently voiding the guarantee.

See [Uncertainty Quantification](../user-guide/uncertainty.md).

---

### `.predict_quantiles(X, quantiles=None)`

Regression only. Returns a dict of predicted quantiles. Available where the model has a
native quantile head — the TabPFN family and TabICLv2.

```python
pipeline.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

---

### `.predict_intervals(X, confidence=0.95)`

Regression only. Returns prediction intervals at the requested confidence level.

For a *distribution-free* guarantee, use
[`ConformalRegressor`](../user-guide/uncertainty.md) instead.

---

### `.clear_cache()`

*New in 0.2.0.* Drops this pipeline's cached predictions and returns the number of entries
removed. No-op when `cache=None`.

---

### `.evaluate_interval_calibration(X, y, ...)`

Regression counterpart to `evaluate_calibration`: checks whether predicted intervals achieve
their nominal coverage.

---

### `.get_residuals(X, y)` / `.analyze_residuals(X, y)` / `.plot_residuals(...)`

Regression diagnostics: raw residuals, a summary dict (bias, heteroscedasticity, normality),
and diagnostic plots.

---

### `.cross_validate(X, y, cv=5, ...)`

IID k-fold cross-validation. For **shift-aware** validation — temporal or grouped splits and
the IID-to-shift gap — use
[`ShiftEvaluator`](../user-guide/shift-evaluation.md) instead.

---

### `.distill(X_train, y_train, ...)`

Compress this fitted pipeline into a lightweight student model. See
[Distillation](../user-guide/distillation.md).

---

### `.get_feature_importance(X, y=None, ...)`

Permutation-based feature importance over the fitted pipeline.

---

### `.evaluate_checkpoints(X_test, y_test, checkpoint_dir, epochs, map_location=None)`

Evaluate every saved epoch checkpoint in a directory and return per-epoch metrics — useful
for picking the best epoch after a fine-tuning run.

---

### `.get_params(deep=True)`

scikit-learn-compatible parameter dict for the pipeline.

---


### `.evaluate_calibration(X, y, n_bins=15, output_format='rich')`

Evaluate model calibration (how well probabilities match actual outcomes).

```python
calibration_metrics = pipeline.evaluate_calibration(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 15,
    output_format: str = 'rich'
) -> dict
```

#### Returns

- **`dict`**: Contains:
  - `brier_score_loss` (float): Mean squared error of probabilities
  - `expected_calibration_error` (float): Average calibration error
  - `maximum_calibration_error` (float): Worst-case calibration error

---

### `.evaluate_fairness(X, y, sensitive_features, output_format='rich')`

Evaluate group fairness metrics.

```python
fairness_metrics = pipeline.evaluate_fairness(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_features: pd.Series,
    output_format: str = 'rich'
) -> dict
```

#### Returns

- **`dict`**: Contains:
  - `statistical_parity_difference` (float): Selection rate disparity
  - `equal_opportunity_difference` (float): True positive rate disparity
  - `equalized_odds_difference` (float): Overall error rate disparity

---

### `.show_processing_summary()`

Display a summary of data preprocessing steps applied.

```python
pipeline.show_processing_summary() -> None
```

#### Example Output

```
Data Processing Summary:
- Imputation: mean (numerical), mode (categorical)
- Scaling: standard
- Encoding: tabicl_special
- Features: 50 numerical, 10 categorical
```

---

### `.baseline(X_train, y_train, X_test, y_test, models=None, time_limit=60)`

Compare TabTune models against AutoGluon baselines.

```python
baseline_results = pipeline.baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: list | str | None = None,
    time_limit: int = 60
) -> dict
```

#### Returns

- **`dict`**: Contains AutoGluon baseline results and leaderboard

---

## Usage Patterns

### Pattern 1: Quick Inference Baseline

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="TabPFN",
    tuning_strategy="inference"
)
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### Pattern 2: Production Fine-Tuning

```python
pipeline = TabularPipeline(
    model_name="OrionBix",
    tuning_strategy="finetune",
    tuning_params={
        "device": "cuda",
        "epochs": 10,
        "learning_rate": 2e-5,
        "save_checkpoint_path": "./checkpoints/model.pt"
    }
)
pipeline.fit(X_train, y_train)
pipeline.save("production_model.joblib")
```

### Pattern 3: Memory-Efficient PEFT

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        "device": "cuda",
        "epochs": 5,
        "learning_rate": 2e-4,
        "peft_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05
        }
    }
)
pipeline.fit(X_train, y_train)
```

---

## Error Handling

### Common Exceptions

**`RuntimeError`**: "You must call fit() before predict()"
- **Cause**: Calling predict/evaluate before fitting
- **Solution**: Call `.fit()` first

**`ModelNotFoundError`**: "Unknown model 'X'"
- **Cause**: The name does not resolve to a registered model
- **Solution**: Check `tabtune.registry.list_model_names()`; the error suggests the closest match

**`UnsupportedTaskError`** / **`UnsupportedStrategyError`**
- **Cause**: The model has no head for that task, or does not implement that strategy
- **Solution**: `get_model_spec(name).tasks` and `.strategies_for(task)`

**`EnvelopeError`**: "X supports at most N classes (found M)"
- **Cause**: Data violates a hard architectural limit of the checkpoint
- **Solution**: Pick a model with a larger envelope, or `envelope_mode='ignore'` if you know what you are doing

**`RuntimeError`**: "CUDA out of memory"
- **Cause**: Insufficient GPU memory
- **Solution**: Use PEFT, reduce batch size, or use CPU

---

## See Also

- [Pipeline Overview](../user-guide/pipeline-overview.md): Detailed usage guide
- [Tuning Strategies](../user-guide/tuning-strategies.md): Strategy comparisons
- [Model Selection](../user-guide/model-selection.md): Choosing the right model
- [Model Registry](../user-guide/registry.md): Envelopes, licensing and `validate`
- [Typed Configuration](../user-guide/configuration.md): YAML configs and CI validation
- [Prediction Caching](../user-guide/caching.md): the `cache` parameter
- [Uncertainty Quantification](../user-guide/uncertainty.md): `uncertainty_report` and conformal wrappers
- [Troubleshooting](../user-guide/troubleshooting.md): Common issues and solutions