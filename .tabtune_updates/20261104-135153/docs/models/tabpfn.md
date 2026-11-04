# TabPFN: Prior-Fitted Network

TabPFN is a revolutionary tabular model that demonstrates strong zero-shot performance without any fine-tuning. This document provides an in-depth guide to using TabPFN with TabTune. TabTune supports both **TabPFN v2.5** and the latest **TabPFN v2.6**, each with distinct fine-tuning capabilities.

---

## 1. Introduction

**What is TabPFN?**

TabPFN (Prior-Fitted Network) is a neural network trained via in-context learning on thousands of synthetic datasets. It approximates Bayesian posterior inference, making it uniquely suited for:

- Quick baseline predictions
- Small dataset learning
- Uncertainty quantification
- Few-shot adaptation

**Key Innovation**: Rather than training on a specific task, TabPFN learns to solve tasks as a **sequence-to-sequence problem**, making it excel in in-context learning scenarios.


**v2.5 vs v2.6**: TabPFN v2.6 is the latest PriorLabs release. It introduces a dedicated native fine-tuning API (`FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor`) with bar distribution loss, cosine LR scheduling with warmup, mixed-precision (AMP), early stopping, and validation-based model selection — in addition to all the meta-learning and SFT modes available in v2.5.

---

## 2. Architecture

### 2.1 High-Level Design

```mermaid
flowchart LR
    A[Input Features] --> B[Feature Encoding]
    B --> C[Support Set Processing]
    C --> D[Transformer Stack]
    D --> E[Bayesian Inference]
    E --> F[Predictions + Uncertainty]
```

### 2.2 Core Components

1. **Feature Encoder**: Converts tabular features to embedding space
2. **Support Set Processor**: Handles training examples as context
3. **Transformer Stack**: Self-attention over support + query samples
4. **Bayesian Head**: Produces mean and variance estimates

### 2.3 Inference Process

```
1. Encode support set (training data)
2. Encode query point (test sample)
3. Process through transformer layers
4. Output Bayesian posterior (mean + variance)
5. Generate predictions with uncertainty
```

---

## 3. Inference Parameters

### 3.1 Complete Parameter Reference

```python
model_params = {
    'n_estimators': 16,                    # Ensemble size
    'softmax_temperature': 0.9,            # Prediction confidence
    'average_logits': True,                # Aggregation method
    'prior_strength': 1.0,                 # Bayesian prior weight
    'normalize_input': True,               # Feature normalization
    'seed': 42                             # Reproducibility
}
```

### 3.2 Parameter Descriptions

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_estimators` | int | 16 | 1-32 | Number of ensemble members; higher = more robust |
| `softmax_temperature` | float | 0.9 | 0.1-2.0 | Scaling of logits before softmax; lower = sharper predictions |
| `average_logits` | bool | True | True/False | Average logits vs probabilities across ensemble |
| `prior_strength` | float | 1.0 | 0.5-2.0 | Weight of Bayesian prior relative to data |
| `normalize_input` | bool | True | True/False | Apply input normalization |
| `seed` | int | 42 | 0+ | Random seed for reproducibility |

### 3.3 Parameter Tuning Guidelines

**Ensemble Size (`n_estimators`)**:
- `8-16`: Fast inference, good uncertainty
- `16-32`: Robust predictions, slower

**Temperature (`softmax_temperature`)**:
- `< 0.5`: Very confident predictions (may overfit)
- `0.5 - 1.0`: Default, balanced confidence
- `> 1.0`: Softer predictions, lower confidence

**Average Method (`average_logits`)**:
- `True`: Better for class imbalance
- `False`: Better for probability calibration

---

## 4. Fine-Tuning with TabPFN

Both v2.5 and v2.6 support `tuning_strategy='finetune'`. The `finetune_mode` parameter controls which algorithm is used.

### finetune_mode options

| `finetune_mode` | v2.5 | v2.6 | Description |
|-----------------|------|------|-------------|
| `'meta-learning'` | ✅ | ✅ | Episodic meta-learning with cosine LR, AMP, gradient accumulation |
| `'sft'` | ✅ | ✅ | SFT-style: single large (support, query) episode repeated for N epochs |
| `'native'` | ❌ | ✅ | PriorLabs `FinetunedTabPFN{Classifier,Regressor}` with bar distribution loss |
| `'turn_by_turn'` / `'tbt'` | ✅ | ✅ | Regression only — episodic turn-by-turn with bardist NLL loss |

If `finetune_mode` is not set, TabTune uses a smart default: `'turn_by_turn'` for regression tasks, `'meta-learning'` for classification.


### 4.1 Base Fine-Tuning Parameters

```python
tuning_params = {
    'device': 'cuda',
    'epochs': 3,
    'learning_rate': 1e-5,
    'batch_size': 512,
    'optimizer': 'adamw',
    'scheduler': 'linear',
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'show_progress': True
}
```

### 4.2 Fine-Tuning Best Practices

- **Learning Rate**: Start with 1e-5, increase if needed
- **Epochs**: 3-5 epochs typically sufficient
- **Batch Size**: 256-512 works well
- **Warmup**: Use 5-10% of total steps
- **Early Stopping**: Monitor validation metric

### 4.3 Fine-Tuning Example

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5,
        'batch_size': 256,
        'scheduler': 'cosine',
        'show_progress': True
    }
)

# Fine-tune on your data
pipeline.fit(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

## 5. TabPFN v2.6 — Fine-Tuning

TabPFN v2.6 adds a **native fine-tuning mode** on top of all v2.5 capabilities.

### 6.1 Classification — Native Fine-Tuning

```python
pipeline = TabularPipeline(
    model_name='TabPFNv26',
    task_type='classification',
    tuning_strategy='finetune',
    finetune_mode='native',      # Uses FinetunedTabPFNClassifier
    tuning_params={
        'device': 'cuda',
        'epochs': 30,
        'learning_rate': 1e-5,
        'early_stopping': True,
        'early_stopping_patience': 8,
        'show_progress': True
    }
)
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### 5.2 Regression — Native Fine-Tuning

```python
pipeline = TabularPipeline(
    model_name='TabPFNv26',
    task_type='regression',
    tuning_strategy='finetune',
    finetune_mode='native',      # Uses FinetunedTabPFNRegressor
    tuning_params={
        'device': 'cuda',
        'epochs': 30,
        'learning_rate': 1e-5,
        'early_stopping': True,
        'early_stopping_patience': 8,
    }
)
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### 5.3 Native Fine-Tuning Parameters (v2.6 only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 30 | Maximum training epochs |
| `learning_rate` | float | 1e-5 | Learning rate |
| `weight_decay` | float | 0.01 | AdamW weight decay |
| `early_stopping` | bool | True | Enable early stopping |
| `early_stopping_patience` | int | 8 | Patience before stopping |
| `validation_split_ratio` | float | 0.1 | Fraction of data for validation |
| `n_finetune_ctx_plus_query_samples` | int | 10000 | Context + query pool size |
| `finetune_ctx_query_split_ratio` | float | 0.2 | Fraction used as query |
| `n_estimators_finetune` | int | 2 | Ensemble size during FT |
| `n_estimators_validation` | int | 2 | Ensemble size during validation |
| `n_estimators_final_inference` | int | 8 | Ensemble size for final predict |
| `grad_clip_value` | float | 1.0 | Gradient clipping value |
| `use_lr_scheduler` | bool | True | Cosine LR with warmup |
| `use_activation_checkpointing` | bool | True | Save memory via gradient checkpointing |
| `random_state` | int | 0 | Reproducibility seed |

---


## 5. Inference-Only Usage

### 5.1 Zero-Shot Predictions

Use TabPFN's pre-trained weights for immediate predictions without training:

```python
from tabtune import TabularPipeline

# Create pipeline with inference strategy
pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='inference',
    model_params={
        'n_estimators': 16,
        'softmax_temperature': 0.9
    }
)

# No training needed - just preprocess and predict
pipeline.fit(X_train, y_train)  # Only does preprocessing
predictions = pipeline.predict(X_test)
uncertainty = pipeline.get_uncertainty(X_test)
```

### 5.2 Uncertainty Estimation

```python
# Get predictions with uncertainty
predictions, std_dev = pipeline.predict_with_uncertainty(X_test)

# Filter predictions by confidence
high_conf_idx = std_dev < np.percentile(std_dev, 25)
print(f"High confidence predictions: {high_conf_idx.sum()}/{len(predictions)}")
```

---

## 6. Usage Scenarios

### 6.1 Quick Baseline

```python
from tabtune import TabularPipeline

# Establish baseline in seconds
pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='inference'
)
pipeline.fit(X_train, y_train)
baseline_score = pipeline.evaluate(X_test, y_test)
print(f"Baseline accuracy: {baseline_score['accuracy']:.4f}")
```

### 6.2 Small Dataset Learning

```python
# For datasets < 10K rows, TabPFN excels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='finetune',
    tuning_params={'epochs': 3}
)
pipeline.fit(X_train, y_train)
```


---

## 7. Limitations and Constraints

### 7.1 Data Constraints

| Constraint | Limit | Impact |
|-----------|-------|--------|
| Max Rows | ~10K | Exceeding causes performance degradation |
| Max Features | ~100 | More features → longer processing time |
| Min Features | 2 | Single-feature prediction not supported |
| Max Classes | 10 | Binary/multi-class up to 10 classes |

### 7.2 Feature Type Constraints

- **Supported**: Numerical, categorical, mixed
- **Not Supported**: Text, images, time-series
- **Preprocessing**: One-hot encoding recommended for categoricals

### 7.3 Task Type Constraints

- ✅ Binary Classification
- ✅ Multi-class Classification
- ❌ Regression
- ❌ Multi-output
- ❌ Multi-label

---

## 8. PEFT (LoRA) Support

### 8.1 Current Status

**⚠️ Experimental**: LoRA support for TabPFN is experimental due to:
- Batched inference engine architecture
- Adapter state management conflicts
- Potential prediction inconsistencies

### 8.2 When to Use PEFT

**Not Recommended** for TabPFN. Use `finetune` strategy instead:

```python
# ❌ Not recommended
pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='peft'  # May have issues - will override 
)

# ✅ Recommended
pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='finetune'  # Fully supported
)
```

---

<!-- ## 9. Performance Characteristics

### 9.1 Speed

| Operation | Time | Notes |
|-----------|------|-------|
| Zero-shot inference | 0.5-2s | Per batch of 1000 samples |
| Fine-tuning | 5-15 min | For 5 epochs on ~50K samples |
| Prediction latency | 10-50ms | Per sample |

### 9.2 Memory Usage

| Scenario | Memory | GPU |
|----------|--------|-----|
| Inference only | 2-3 GB | 2GB VRAM minimum |
| Fine-tuning | 4-6 GB | 4GB VRAM recommended |
| Large batch | Up to 8 GB | 8GB VRAM for batch_size=512 |

### 9.3 Accuracy Profile

| Dataset Size | Baseline Acc | After Fine-Tune |
|--------------|-------------|-----------------|
| <5K rows | 80-85% | 85-90% |
| 5-10K rows | 82-87% | 87-92% |
| >10K rows | Degradation | Comparable to TabICL |

 -->
 ---

## 10. Troubleshooting

### Issue: "Dataset too large for TabPFN"
**Solution**: Use TabICL for datasets >10K rows

```python
if len(X_train) > 10000:
    model = 'TabICL'
else:
    model = 'TabPFN'
```

### Issue: "Out of memory during inference"
**Solution**: Reduce batch size

```python
tuning_params = {
    'batch_size': 128  # Instead of 512
}
```

### Issue: "Predictions too confident (low uncertainty)"
**Solution**: Increase temperature

```python
model_params = {
    'softmax_temperature': 1.5  # Instead of 0.9
}
```

### Issue: "PEFT causing prediction errors"
**Solution**: Use finetuning strategy instead

```python
pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='finetune'  # Not peft
)
```

---

## 11. Complete Example Workflow

```python
from tabtune import TabularPipeline, TabularLeaderboard
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load data
df = pd.read_csv('small_dataset.csv')  # <10K rows ideal
X = df.drop('target', axis=1)
y = df['target']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Strategy 1: Zero-shot baseline
print("=== Zero-Shot Baseline ===")
baseline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='inference'
)
baseline.fit(X_train, y_train)
baseline_metrics = baseline.evaluate(X_test, y_test)
print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")

# 4. Strategy 2: Fine-tuned
print("\n=== Fine-Tuned ===")
finetuned = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5,
        'show_progress': True
    }
)
finetuned.fit(X_train, y_train)
finetuned_metrics = finetuned.evaluate(X_test, y_test)
print(f"Fine-tuned Accuracy: {finetuned_metrics['accuracy']:.4f}")

# 5. Compare with other models
print("\n=== Model Comparison ===")
lb = TabularLeaderboard(X_train, X_test, y_train, y_test)
lb.add_model('TabPFN', 'inference', name='TabPFN-Inference')
lb.add_model('TabPFN', 'finetune', name='TabPFN-FineTune', tuning_params={'epochs': 5})
lb.add_model('TabICL', 'peft', name='TabICL-PEFT')

lb.run(rank_by='accuracy')
 

```


---

## 12. Quick Reference

| Task | Strategy | Time | Accuracy |
|------|----------|------|----------|
| Instant baseline | inference | <1s | Medium |
| Rapid prototyping | finetune + 3 epochs | 5m | Good |
| Production model | finetune + 5 epochs | 15m | High |
| Uncertainty estimation | inference | <1s | With uncertainty |

---

## 13. Next Steps

- [Model Selection](../user-guide/model-selection.md) - Compare with other models
- [Tuning Strategies](../user-guide/tuning-strategies.md) - Fine-tuning details
- [TabularLeaderboard](../user-guide/leaderboard.md) - Benchmark TabPFN vs other models
- [API Reference](../api/pipeline.md) - Complete API docs

---

TabPFN excels at quick learning on small datasets. Use it for rapid experimentation and as a strong baseline!