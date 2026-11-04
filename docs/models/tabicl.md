# TabICL

TabICL (Inria Soda) is the default answer to "which model do I try first": column-then-row attention, fast, BSD-licensed, and strong on tables up to hundreds of thousands of rows. This page covers the original, classification-only TabICL. [TabICL v2](tabiclv2.md) adds regression and has its own page.

## Introduction

**TabICL**

In-context learning adapted to tables. TabICL learns to:

- Process feature relationships dynamically
- Adapt to task-specific patterns via fine-tuning
- Generate reliable predictions via ensemble methods
- Handle mixed data types naturally

The idea: two-stage attention, columns first, then rows, so feature interactions are modelled before sample interactions.

### TabICLv2

TabICLv2 is the second generation of TabICL with an improved architecture. It extends support to **both classification and regression** with inference and episodic fine-tuning for both tasks.

## Architecture

### High-level design

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
### Core components

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
### Two-stage attention

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

## Inference parameters

### Complete parameter reference

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

### Parameter descriptions

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_estimators` | int | 32 | 4-128 | Number of ensemble members; more = reliable but slower |
| `softmax_temperature` | float | 0.9 | 0.1-2.0 | Scaling before softmax; lower = sharper predictions |
| `average_logits` | bool | True | True/False | Average logits vs probabilities |
| `norm_methods` | list | ['none', 'power'] | Varies | Feature normalization techniques |
| `feat_shuffle_method` | str | 'latin' | 'random', 'latin', 'sequential' | Feature permutation strategy |
| `batch_size` | int | 8 | 1-32 | Ensemble members per batch |
| `seed` | int | 42 | 0+ | Random seed for reproducibility |

### Ensemble size effects

| n_estimators | Speed | Robustness | Memory |
|-------------|-------|-----------|--------|
| 4-8 | | | |
| 16-32 | | | |
| 64-128 | | | |

### Feature normalization methods

```python
norm_methods = [
    'none',          # No normalization
    'power',         # Power transformation
    'quantile',      # Quantile normalization
    'minmax',        # Min-max scaling
    'standard'       # Standardization
]
```

### Feature shuffle methods

```python
feat_shuffle_methods = {
    'random': 'Random permutation each time',
    'latin': 'Latin square design (balanced)',
    'sequential': 'Fixed sequential order'
}
```

## Fine-tuning with TabICL

TabICL supports **episodic fine-tuning** where training occurs via task-like episodes.

### Episodic training parameters

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

### Parameter descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `support_size` | int | 48 | Number of samples per support set |
| `query_size` | int | 32 | Number of samples per query set |
| `n_episodes` | int | 1000 | Total episodes for training |
| `batch_size` | int | 8 | Episodes per batch gradient update |

### Episodic training concept

```
One Episode:
  ├─ Support Set (48 samples)
  │  └─ Used as training context
  ├─ Query Set (32 samples)
  │  └─ Used for evaluation
  └─ Loss computed on Query

Epoch = 1000 episodes with gradient updates
```

### Fine-tuning guidelines

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

## Fine-Tuning, TabICLv2 (regression)

TabICLv2 supports **episodic regression fine-tuning** using turn-by-turn MSE training. It samples random (support, query) episodes, feeds them through the transformer, and backpropagates MSE loss on query predictions.

### Regression inference

```python
pipeline = TabularPipeline(
    model_name='TabICLv2',
    task_type='regression',
    tuning_strategy='inference',
)
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
# Returns: mse, rmse, mae, r2_score, medae, explained_variance, ...
```

### Regression fine-tuning

```python
pipeline = TabularPipeline(
    model_name='TabICLv2',
    task_type='regression',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-6,
        'support_size': 128,
        'query_size': 64,
        'steps_per_epoch': 200,
        'grad_clip': 1.0,
        'show_progress': True
    }
)
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### Regression fine-tuning parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 5 | Training epochs |
| `learning_rate` | float | 2e-6 | Optimizer learning rate (lower than classification) |
| `support_size` | int | 128 | Context (support) set size per episode |
| `query_size` | int | 64 | Query set size per episode |
| `steps_per_epoch` | int | 200 | Random episodes sampled per epoch |
| `grad_clip` | float | 1.0 | Gradient clipping norm |
| `weight_decay` | float | 0.01 | AdamW weight decay |
| `show_progress` | bool | True | Show tqdm progress bar |

### How TabICLv2 regression fine-tuning works

```
One step:
  ├─ Sample random support indices (128 rows)
  ├─ Sample random query indices (64 rows)
  ├─ Concatenate: X_episode = [X_support | X_query]  (1, 192, F)
  ├─ Forward: model_(X_episode, y_support) → predictions for query
  ├─ If output has multiple bins: take weighted mean (distribution output)
  └─ MSE loss on query predictions vs ground truth

Post-finetune: model.fit(X_train, y_train) re-caches for inference
```

## Usage patterns

### Inference only

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

### Base fine-tuning (full parameters)

```python
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='finetune',
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

### PEFT fine-tuning (LoRA adapters)

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

### Ensemble configuration

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

## LoRA target modules

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

## Complete examples

### Basic workflow

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
    tuning_strategy='finetune',
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

### PEFT for memory-constrained environments

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

### Hyperparameter tuning

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

<!-- ## 8. Performance Characteristics

### Speed benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Inference (32 ensemble) | 1-2s | Per 1000 samples |
| Base FT (5 epochs) | 20-30m | For 100K samples |
| PEFT (5 epochs) | 10-15m | For 100K samples |
| Prediction latency | 10-50ms | Per sample |

### Memory usage

| Scenario | Memory | GPU VRAM |
|----------|--------|---------|
| Inference | 5-7 GB | 4GB minimum |
| Base FT | 12-16 GB | 8GB recommended |
| PEFT | 6-8 GB | 4GB sufficient |
| Large batch | Up to 20 GB | 12GB+ needed |

### Accuracy profile

| Dataset Size | Inference | Base-FT | PEFT |
|--------------|-----------|---------|------|
| 10K | 82% | 88% | 86% |
| 100K | 85% | 90% | 89% |
| 1M | 87% | 92% | 91% |

 -->
## Troubleshooting

### "Out of memory during training"
**Solution 1**: Reduce support/query sizes
```python
tuning_params = {
    'support_size': 24,  # Instead of 48
    'query_size': 16     # Instead of 32
}
```

**Solution 2**: Use PEFT instead of finetune
```python
tuning_strategy = 'peft'  # Lower memory
```

### "Model not converging"
**Solution**: Adjust learning rate and epochs
```python
tuning_params = {
    'learning_rate': 5e-5,  # Increase
    'epochs': 10,           # More epochs
    'n_episodes': 2000      # More training
}
```

### "Inference too slow"
**Solution**: Reduce ensemble size
```python
model_params = {
    'n_estimators': 8   # Instead of 32
}
```

### "Low accuracy on small datasets"
**Solution**: Use larger support set
```python
tuning_params = {
    'support_size': 96,  # Larger context
    'query_size': 64
}
```

## Quick reference

| Use Case | Model | Strategy | Config | Notes |
|----------|-------|----------|--------|-------|
| Quick classification | TabICL | inference | n_est=16 | <1s |
| Classification FT | TabICL | finetune | epochs=5 | Episodic |
| Classification PEFT | TabICL | peft | r=8, epochs=5 | Memory efficient |
| Regression inference | TabICLv2 | inference |, | |
| Regression FT | TabICLv2 | finetune | lr=2e-6, epochs=5 | Turn-by-turn MSE |

<!-- ## 11. Comparison with Other Models

| Aspect | TabICL | TabPFN | TabDPT | Mitra |
|--------|--------|--------|--------|-------|
| Small data | Good | Excellent | Okay | Good |
| Large data | Excellent | Poor | Excellent | Okay |
| Speed | Fast | Fastest | Slow | Slowest |
| Memory | Moderate | Low | High | Very High |
| PEFT | Full | Exp | Full | Full |
 -->
