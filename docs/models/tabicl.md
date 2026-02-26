# TabICL: In-Context Learning for Tabular Data

TabICL is a scalable, ensemble-based in-context learning model designed for general-purpose tabular classification. This document provides comprehensive guidance for using TabICL with TabTune.

---

## 1. Introduction

**What is TabICL?**

TabICL (Tabular In-Context Learning) is a neural model that leverages in-context learning principles adapted for tabular data. Unlike traditional models, TabICL learns to:

- Process feature relationships dynamically
- Adapt to task-specific patterns via fine-tuning
- Generate robust predictions via ensemble methods
- Handle mixed data types naturally

**Key Innovation**: Two-stage attention mechanism (column → row) enabling efficient feature processing and interaction modeling.

---

## 2. Architecture

### 2.1 High-Level Design

```mermaid
flowchart LR
    A[Input Features] --> B[Column Embedder]
    B --> C[Feature-wise Embeddings]
    C --> D[Row Interactor]
    D --> E[Feature Interactions]
    E --> F[ICL Predictor]
    F --> G[Predictions]
    G --> H[Ensemble Aggregation]
    H --> I[Final Output]
```
<!-- 
### 2.2 Core Components

1. **Column Embedder** (`col_embedder`)
   - Processes individual features independently
   - Learns feature-specific representations
   - Generates per-feature embeddings

2. **Row Interactor** (`row_interactor`)
   - Models relationships between features
   - Cross-feature attention mechanisms
   - Captures feature interactions

3. **ICL Predictor** (`icl_predictor`)
   - Context-aware prediction head
   - Leverages support/query sets
   - Generates class logits

4. **Ensemble Aggregation**
   - Multiple feature permutations
   - Voting across permutations
   - Robustness via diversity
 -->
### 2.3 Two-Stage Attention

```
Stage 1: Column Attention (Feature-wise)
  ↓
  Per-feature processing
  Feature extraction

Stage 2: Row Attention (Sample-wise)
  ↓
  Feature interaction
  Context modeling
```

---

## 3. Inference Parameters

### 3.1 Complete Parameter Reference

```python
model_params = {
    'n_estimators': 32,                    # Ensemble size (views)
    'softmax_temperature': 0.9,            # Prediction confidence
    'average_logits': True,                # Aggregation method
    'norm_methods': ['none', 'power'],     # Feature normalization
    'feat_shuffle_method': 'latin',        # Feature permutation strategy
    'batch_size': 8,                       # Ensemble batch size
    'seed': 42                             # Reproducibility
}
```

### 3.2 Parameter Descriptions

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_estimators` | int | 32 | 4-128 | Number of ensemble members; more = robust but slower |
| `softmax_temperature` | float | 0.9 | 0.1-2.0 | Scaling before softmax; lower = sharper predictions |
| `average_logits` | bool | True | True/False | Average logits vs probabilities |
| `norm_methods` | list | ['none', 'power'] | Varies | Feature normalization techniques |
| `feat_shuffle_method` | str | 'latin' | 'random', 'latin', 'sequential' | Feature permutation strategy |
| `batch_size` | int | 8 | 1-32 | Ensemble members per batch |
| `seed` | int | 42 | 0+ | Random seed for reproducibility |

### 3.3 Ensemble Size Effects

| n_estimators | Speed | Robustness | Memory |
|-------------|-------|-----------|--------|
| 4-8 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| 16-32 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 64-128 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 3.4 Feature Normalization Methods

```python
norm_methods = [
    'none',          # No normalization
    'power',         # Power transformation
    'quantile',      # Quantile normalization
    'minmax',        # Min-max scaling
    'standard'       # Standardization
]
```

### 3.5 Feature Shuffle Methods

```python
feat_shuffle_methods = {
    'random': 'Random permutation each time',
    'latin': 'Latin square design (balanced)',
    'sequential': 'Fixed sequential order'
}
```

---

## 4. Fine-Tuning with TabICL

TabICL supports **episodic fine-tuning** where training occurs via task-like episodes.

### 4.1 Episodic Training Parameters

```python
tuning_params = {
    'device': 'cuda',
    'epochs': 5,                          # Training epochs
    'learning_rate': 2e-5,                # Optimizer learning rate
    'optimizer': 'adamw',                 # Optimizer type
    # Episodic parameters
    'support_size': 48,                   # Support set samples
    'query_size': 32,                     # Query set samples
    'n_episodes': 1000,                   # Episodes per epoch
    'batch_size': 8,                      # Episodes per batch
    'show_progress': True                 # Progress bar
}
```

### 4.2 Parameter Descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `support_size` | int | 48 | Number of samples per support set |
| `query_size` | int | 32 | Number of samples per query set |
| `n_episodes` | int | 1000 | Total episodes for training |
| `batch_size` | int | 8 | Episodes per batch gradient update |

### 4.3 Episodic Training Concept

```
One Episode:
  ├─ Support Set (48 samples)
  │  └─ Used as training context
  ├─ Query Set (32 samples)
  │  └─ Used for evaluation
  └─ Loss computed on Query

Epoch = 1000 episodes with gradient updates
```

### 4.4 Fine-Tuning Guidelines

**Support/Query Size Balance**:
- Larger support → more context but slower
- Larger query → better gradient signal
- Typical: support:query = 3:2

**Number of Episodes**:
- 500-1000: Good for medium datasets
- 1000-5000: Better for large datasets
- Adjust based on dataset size: \(n_{episodes} = \frac{\text{dataset_size}}{100}\)

**Learning Rate**:
- 1e-5: Conservative, safe
- 2e-5: Balanced (default)
- 5e-5: Aggressive, higher variance

---

## 5. Usage Patterns

### 5.1 Inference Only

```python
from tabtune import TabularPipeline

# Zero-shot with pre-trained weights
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='inference',
    model_params={'n_estimators': 32}
)

pipeline.fit(X_train, y_train)  # Only preprocesses
predictions = pipeline.predict(X_test)
```

### 5.2 Base Fine-Tuning (Full Parameters)

```python
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'support_size': 48,
        'query_size': 32,
        'n_episodes': 1000
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### 5.3 PEFT Fine-Tuning (LoRA Adapters)

```python
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,  # Higher for PEFT
        'support_size': 24,     # Smaller for memory
        'query_size': 16,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        }
    }
)

pipeline.fit(X_train, y_train)
```

### 5.4 Ensemble Configuration

```python
# Increase ensemble for robustness
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='inference',
    model_params={
        'n_estimators': 128,  # Large ensemble
        'batch_size': 16      # Parallel processing
    }
)
```

---

## 6. LoRA Target Modules

When using PEFT, TabICL automatically targets these modules:

```python
target_modules = [
    'col_embedder.tf_col',          # Column transformer
    'col_embedder.in_linear',       # Input projection
    'row_interactor',               # Interaction layers
    'icl_predictor.tf_icl',         # Prediction transformer
    'icl_predictor.decoder'         # Decoder head
]
```

**Default PEFT Config**:
```python
peft_config = {
    'r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'target_modules': None  # Uses defaults above
}
```

---

## 7. Complete Examples

### 7.1 Basic Workflow

```python
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5,
        'n_episodes': 1000
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### 7.2 PEFT for Memory-Constrained Environments

```python
# Fit large model in limited memory with PEFT
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'support_size': 24,    # Reduced
        'query_size': 16,      # Reduced
        'batch_size': 4,       # Smaller batches
        'peft_config': {
            'r': 4,            # Lower rank
            'lora_alpha': 8,
            'lora_dropout': 0.1
        }
    }
)

pipeline.fit(X_train, y_train)
```

### 7.3 Hyperparameter Tuning

```python
from tabtune import TabularLeaderboard

# Compare different configurations
lb = TabularLeaderboard(X_train, X_test, y_train, y_test)

# Ensemble size comparison
for n_est in [16, 32, 64]:
    lb.add_model(
        'TabICL',
        'inference',
        name=f'TabICL-n{n_est}',
        model_params={'n_estimators': n_est}
    )

# LoRA rank comparison
for r in [4, 8, 16]:
    lb.add_model(
        'TabICL',
        'peft',
        name=f'TabICL-PEFT-r{r}',
        tuning_params={
            'epochs': 3,
            'peft_config': {'r': r, 'lora_alpha': 2*r}
        }
    )

lb.run(rank_by='accuracy')

```

---

<!-- ## 8. Performance Characteristics

### 8.1 Speed Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Inference (32 ensemble) | 1-2s | Per 1000 samples |
| Base FT (5 epochs) | 20-30m | For 100K samples |
| PEFT (5 epochs) | 10-15m | For 100K samples |
| Prediction latency | 10-50ms | Per sample |

### 8.2 Memory Usage

| Scenario | Memory | GPU VRAM |
|----------|--------|---------|
| Inference | 5-7 GB | 4GB minimum |
| Base FT | 12-16 GB | 8GB recommended |
| PEFT | 6-8 GB | 4GB sufficient |
| Large batch | Up to 20 GB | 12GB+ needed |

### 8.3 Accuracy Profile

| Dataset Size | Inference | Base-FT | PEFT |
|--------------|-----------|---------|------|
| 10K | 82% | 88% | 86% |
| 100K | 85% | 90% | 89% |
| 1M | 87% | 92% | 91% |

---
 -->
## 9. Troubleshooting

### Issue: "Out of memory during training"
**Solution 1**: Reduce support/query sizes
```python
tuning_params = {
    'support_size': 24,  # Instead of 48
    'query_size': 16     # Instead of 32
}
```

**Solution 2**: Use PEFT instead of base-ft
```python
tuning_strategy = 'peft'  # Lower memory
```

### Issue: "Model not converging"
**Solution**: Adjust learning rate and epochs
```python
tuning_params = {
    'learning_rate': 5e-5,  # Increase
    'epochs': 10,           # More epochs
    'n_episodes': 2000      # More training
}
```

### Issue: "Inference too slow"
**Solution**: Reduce ensemble size
```python
model_params = {
    'n_estimators': 8   # Instead of 32
}
```

### Issue: "Low accuracy on small datasets"
**Solution**: Use larger support set
```python
tuning_params = {
    'support_size': 96,  # Larger context
    'query_size': 64
}
```

---

## 10. Advanced Topics

<!-- ### 10.1 Custom Feature Normalization

```python
# Experiment with normalization methods
for norm in ['none', 'power', 'quantile']:
    pipeline = TabularPipeline(
        model_name='TabICL',
        model_params={'norm_methods': [norm]}
    )
    # Evaluate...
``` -->

<!-- ### 10.2 Feature Importance

```python
from sklearn.inspection import permutation_importance

# Get feature importance with TabICL
result = permutation_importance(
    pipeline.model,
    X_test,
    y_test,
    n_repeats=10
)

importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)
```
 -->
<!-- ### 10.3 Cross-Validation

```python
from sklearn.model_selection import cross_validate

# 5-fold cross-validation
scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=['accuracy', 'f1_weighted']
)

print(f"Mean Accuracy: {scores['test_accuracy'].mean():.4f}")
print(f"Std Accuracy: {scores['test_accuracy'].std():.4f}")
``` -->

---

## 11. Quick Reference

| Use Case | Strategy | Config | Time |
|----------|----------|--------|------|
| Quick test | inference | n_est=16 | <1s |
| Rapid proto | peft | r=8, epochs=3 | 5-10m |
| Production | base-ft | epochs=5 | 20-30m |
| Max accuracy | base-ft | epochs=10 | 40-60m |
| Memory limited | peft | r=4 | 5-10m |


<!-- ## 12. Comparison with Other Models

| Aspect | TabICL | TabPFN | TabDPT | Mitra |
|--------|--------|--------|--------|-------|
| Small data | Good | Excellent | Okay | Good |
| Large data | Excellent | Poor | Excellent | Okay |
| Speed | Fast | Fastest | Slow | Slowest |
| Memory | Moderate | Low | High | Very High |
| PEFT | ✅ Full | ⚠️ Exp | ✅ Full | ✅ Full |
 -->
---

## 13. Next Steps

- [Model Selection](../user-guide/model-selection.md) - Compare with other models
- [Tuning Strategies](../user-guide/tuning-strategies.md) - Deep dive into strategies
- [Advanced PEFT](../advanced/peft-lora.md) - LoRA deep dive
- [TabularLeaderboard](../user-guide/leaderboard.md) - Benchmark TabICL

---

TabICL offers an excellent balance of speed, accuracy, and scalability. Use it for most tabular classification tasks!