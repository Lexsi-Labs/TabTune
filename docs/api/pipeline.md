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
    finetune_mode: str = 'meta-learning'
)
```

#### Parameters

**`model_name`** (str, required)
- Name of the model to use.
- Supported values: `'TabPFN'`, `'TabICL'`, `'OrionMSP'`, `'OrionBix'`, `'TabDPT'`, `'Mitra'`, `'ContextTab'`
- Example: `model_name="TabICL"`

**`task_type`** (str, default: `'classification'`)
- Type of machine learning task.
- Currently supported: `'classification'`
- Planned: `'regression'`
- Example: `task_type="classification"`

**`tuning_strategy`** (str, default: `'inference'`)
- Training/fine-tuning strategy to use.
- Options:
  - `'inference'`: Zero-shot predictions (no training)
  - `'base-ft'`: Full fine-tuning of all parameters
  - `'peft'`: Parameter-efficient fine-tuning with LoRA adapters
- Example: `tuning_strategy="peft"`

**`tuning_params`** (dict, optional)
- Hyperparameters for training/inference.
- Common parameters:
  - `device` (str): `'cuda'` or `'cpu'` (default: auto-detected)
  - `epochs` (int): Number of training epochs
  - `learning_rate` (float): Learning rate for optimizer
  - `batch_size` (int): Batch size for training
  - `peft_config` (dict): LoRA configuration for PEFT strategy
  - `support_size` (int): Context size for episodic training
  - `query_size` (int): Query size for episodic training
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
    tuning_strategy="base-ft"
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
    tuning_strategy="base-ft",
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

**`ValueError`**: "Model 'X' not supported"
- **Cause**: Invalid model name
- **Solution**: Check supported models list

**`RuntimeError`**: "CUDA out of memory"
- **Cause**: Insufficient GPU memory
- **Solution**: Use PEFT, reduce batch size, or use CPU

---

## See Also

- [Pipeline Overview](../user-guide/pipeline-overview.md): Detailed usage guide
- [Tuning Strategies](../user-guide/tuning-strategies.md): Strategy comparisons
- [Model Selection](../user-guide/model-selection.md): Choosing the right model
- [Troubleshooting](../user-guide/troubleshooting.md): Common issues and solutions