# Troubleshooting Guide

This guide addresses common errors, issues, and solutions when using TabTune.

---

## Common Import Errors

### ModuleNotFoundError: No module named 'tabtune'

**Cause**: TabTune is not installed or not in your Python path.

**Solution**:
```bash
cd TabTune
pip install -e .
```

**Verify installation**:
```python
import tabtune
print(tabtune.__version__)
```

### ImportError: Cannot import TabularPipeline

**Cause**: Incorrect import path or installation issue.

**Solution**: Use the correct import:
```python
from tabtune import TabularPipeline  # ✅ Correct
# NOT: from TabularPipeline.pipeline import TabularPipeline  # ❌ Old path
```

### ImportError: No module named 'torch'

**Cause**: PyTorch is not installed.

**Solution**:
```bash
# For CPU only
pip install torch

# For GPU support (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## CUDA/GPU Issues

### CUDA out of memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Use PEFT strategy** (recommended):
   ```python
   pipeline = TabularPipeline(
       model_name="TabICL",
       tuning_strategy="peft",  # Uses 40-60% less memory
       tuning_params={"peft_config": {"r": 4}}  # Lower rank = less memory
   )
   ```

2. **Reduce batch size**:
   ```python
   tuning_params={"batch_size": 4}  # Default is often 8 or 16
   ```

3. **Use a smaller model**:
   - Switch from TabDPT to TabICL
   - Switch from OrionBix to TabICL

4. **Process in chunks**:
   ```python
   # Split dataset into smaller batches
   chunk_size = 10000
   for i in range(0, len(X_train), chunk_size):
       X_chunk = X_train[i:i+chunk_size]
       y_chunk = y_train[i:i+chunk_size]
       # Process chunk
   ```

5. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

6. **Use CPU instead** (slower but no memory limits):
   ```python
   tuning_params={"device": "cpu"}
   ```

### CUDA device not found

**Symptoms**: `RuntimeError: CUDA error: no kernel image is available`

**Cause**: PyTorch version doesn't match your GPU's CUDA version.

**Solution**:
1. Check your CUDA version: `nvidia-smi`
2. Install matching PyTorch:
   ```bash
   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### GPU not being used

**Symptoms**: Training runs on CPU despite having GPU.

**Check**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
```

**Solution**: Explicitly set device in `tuning_params`:
```python
tuning_params={"device": "cuda"}
```

---

## Memory Errors

### Out of memory during training

**Cause**: Dataset too large or batch size too high.

**Solutions**:
1. Reduce batch size: `tuning_params={"batch_size": 2}`
2. Use gradient accumulation:
   ```python
   tuning_params={
       "batch_size": 4,
       "gradient_accumulation_steps": 4  # Effective batch size = 16
   }
   ```
3. Use PEFT instead of base-ft
4. Reduce dataset size for testing

### Memory leak during training

**Symptoms**: Memory usage increases over epochs.

**Solution**:
```python
# Add periodic cleanup
import torch
import gc

for epoch in range(epochs):
    # Training code
    if epoch % 5 == 0:
        torch.cuda.empty_cache()
        gc.collect()
```

---

## Model Loading Failures

### Model checkpoint not found

**Symptoms**: `FileNotFoundError` when loading checkpoint.

**Solution**:
```python
# Check if checkpoint exists
import os
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
    # Use inference mode instead
    pipeline = TabularPipeline(model_name="TabPFN", tuning_strategy="inference")
```

### Model state dict mismatch

**Symptoms**: `RuntimeError: Error(s) in loading state_dict`

**Cause**: Checkpoint from different model version or architecture.

**Solution**:
- Use checkpoints saved from the same TabTune version
- Or start fresh training without loading checkpoint

---

## Preprocessing Errors

### ValueError: Found array with 0 sample(s)

**Cause**: Empty dataset after preprocessing/filtering.

**Solution**:
```python
# Check data before preprocessing
print(f"Data shape: {X.shape}")
print(f"Null values: {X.isnull().sum().sum()}")

# Ensure sufficient samples after train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train samples: {len(X_train)}")
```

### Categorical encoding mismatch

**Symptoms**: Errors during prediction about unseen categories.

**Cause**: Test set contains categories not seen during training.

**Solution**:
- Ensure train/test split preserves all categories
- Use `stratify=y` in train_test_split for classification
- Check for new categories in test set:
  ```python
  train_cats = set(X_train['column'].unique())
  test_cats = set(X_test['column'].unique())
  unseen = test_cats - train_cats
  if unseen:
      print(f"Unseen categories: {unseen}")
  ```

### Data type mismatches

**Symptoms**: Type errors during preprocessing.

**Solution**:
```python
# Ensure correct types
X = X.astype({'col1': 'float64', 'col2': 'category'})

# Or let DataProcessor handle it
pipeline = TabularPipeline(
    model_name="TabICL",
    processor_params={"override_types": None}  # Auto-detect
)
```

---

## Training Convergence Issues

### Model not converging / Loss not decreasing

**Symptoms**: Loss plateaus or increases.

**Solutions**:

1. **Lower learning rate**:
   ```python
   tuning_params={"learning_rate": 1e-5}  # Default might be too high
   ```

2. **Reduce epochs**:
   ```python
   tuning_params={"epochs": 3}  # Start small
   ```

3. **Check data quality**:
   - Ensure labels are correct
   - Check for data leakage
   - Verify train/test split

4. **Try different model**:
   - Some models work better for certain datasets
   - Use TabularLeaderboard to compare

5. **Warmup learning rate**:
   ```python
   tuning_params={
       "learning_rate": 2e-5,
       "warmup_steps": 100
   }
   ```

### Overfitting

**Symptoms**: High training accuracy, low validation accuracy.

**Solutions**:
1. **Reduce epochs**: `tuning_params={"epochs": 3}`
2. **Lower learning rate**: `tuning_params={"learning_rate": 1e-5}`
3. **Use PEFT**: Often generalizes better than base-ft
4. **More training data**: Collect or use data augmentation
5. **Early stopping**: Implement callback to stop when validation loss increases

### Underfitting

**Symptoms**: Both training and validation accuracy are low.

**Solutions**:
1. **Train longer**: Increase `epochs`
2. **Higher learning rate**: `tuning_params={"learning_rate": 5e-4}`
3. **Use base-ft**: Full fine-tuning may be needed
4. **Larger model**: Try TabDPT or OrionBix instead of TabICL
5. **Feature engineering**: Add relevant features

---

## PEFT-Specific Problems

### PEFT not working / Falling back to base-ft

**Symptoms**: Warning messages about PEFT compatibility.

**Cause**: Some models have experimental PEFT support.

**Solution**:
- TabPFN and ContextTab have experimental PEFT
- Use `base-ft` strategy for these models
- Or check PEFT configuration:
  ```python
  peft_config = {
      "r": 8,  # Rank (lower = less memory, but may affect performance)
      "lora_alpha": 16,
      "lora_dropout": 0.05,
      "target_modules": ["query", "value"]  # Model-specific
  }
  ```

### PEFT model size larger than expected

**Cause**: Incorrect target modules or high rank.

**Solution**:
```python
# Use lower rank
peft_config = {"r": 4}  # Instead of default 8

# Check actual model size
import torch
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable}")
```

---

## Data Format Errors

### ValueError: Input must be pandas DataFrame

**Symptoms**: Error when passing numpy arrays.

**Solution**: Convert to DataFrame:
```python
import pandas as pd
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y)
```

### Shape mismatch errors

**Symptoms**: Dimension errors during prediction.

**Cause**: Feature count differs between train and test.

**Solution**:
```python
# Ensure same features
assert X_train.columns.tolist() == X_test.columns.tolist()

# Or use pipeline's built-in handling
# TabTune automatically handles this
```

### Missing target column

**Symptoms**: KeyError when accessing target.

**Solution**:
```python
# Ensure target is separate Series
y = df['target']  # ✅ Correct
# NOT: y = df[['target']]  # ❌ DataFrame instead of Series
```

---

## Version Compatibility Issues

### PyTorch version conflicts

**Symptoms**: Errors about tensor operations or CUDA compatibility.

**Solution**: Check and match versions:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

Update if needed:
```bash
pip install --upgrade torch
```

### scikit-learn version issues

**Symptoms**: Deprecation warnings or API errors.

**Solution**: Use compatible version:
```bash
pip install scikit-learn==1.7
```

### Python version too old

**Symptoms**: Syntax errors or unsupported features.

**Solution**: TabTune requires Python 3.10+. Upgrade Python:
```bash
# Using conda
conda create -n tabtune python=3.10
conda activate tabtune
```

---

## Evaluation & Prediction Errors

### predict_proba returns incorrect shape

**Symptoms**: Shape mismatch or wrong number of classes.

**Solution**:
```python
# Check class count
n_classes = len(pipeline.processor.custom_preprocessor_.label_encoder_.classes_)
print(f"Expected {n_classes} classes")
probabilities = pipeline.predict_proba(X_test)
print(f"Got shape: {probabilities.shape}")  # Should be (n_samples, n_classes)
```

### All predictions are the same class

**Symptoms**: Model predicts only one class for all samples.

**Possible causes**:
1. Model not trained (using inference with poor weights)
2. Severe class imbalance
3. Data preprocessing issue

**Solutions**:
1. **Fine-tune the model**: Use `base-ft` or `peft`
2. **Check class distribution**:
   ```python
   print(y_train.value_counts())
   ```
3. **Use resampling**:
   ```python
   processor_params={"resampling_strategy": "smote"}
   ```
4. **Inspect predictions**:
   ```python
   predictions = pipeline.predict(X_test)
   print(f"Unique predictions: {np.unique(predictions)}")
   print(f"Prediction distribution: {pd.Series(predictions).value_counts()}")
   ```

---

## Performance Issues

### Training is very slow

**Solutions**:
1. **Use GPU**: `tuning_params={"device": "cuda"}`
2. **Reduce dataset size**: Test with subset first
3. **Use inference mode**: For quick baselines
4. **Optimize batch size**: Larger batches (if memory allows)
5. **Use PEFT**: Faster than base-ft

### Inference is slow

**Solutions**:
1. **Batch predictions**: Process multiple samples at once
2. **Use GPU**: `tuning_params={"device": "cuda"}`
3. **Reduce n_estimators** (for ensemble models):
   ```python
   model_params={"n_estimators": 8}  # Instead of default 16 or 32
   ```
4. **Cache preprocessing**: Save and load preprocessed data

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the logs**: TabTune provides detailed logging
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Reproduce with minimal example**: Create smallest code that reproduces the issue

3. **Check GitHub Issues**: Search [TabTune_Internal Issues](https://github.com/Lexsi-Labs/TabTune_Internal/issues)

4. **Open new issue**: Include:
   - TabTune version
   - Python version
   - Full error traceback
   - Minimal reproducible code
   - System information

5. **Review documentation**: Check relevant guides:
   - [Installation](../getting-started/installation.md)
   - [User Guide](pipeline-overview.md)
   - [FAQ](../about/faq.md)
