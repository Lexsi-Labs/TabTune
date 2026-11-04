# TabDPT

TabDPT (Layer 6) was pretrained with a denoising objective on real datasets rather than synthetic ones, and at inference it retrieves the nearest training rows as context instead of reading the whole table. That retrieval is why it scales to millions of rows.

## Introduction

**TabDPT**

It is built for:

- Large-Scale Learning: Scales to datasets with millions of samples
- Reliable Feature Learning: Pre-trained on denoising objectives
- Noise Resilience: Handles missing and corrupted features
- Context-Aware Predictions: k-NN based context selection
- Strong Generalization: Pre-trained on diverse tabular corpora

The idea: pretraining by masked feature prediction on real tables, which gives feature representations that carry over to new tasks.

## Architecture

### High-level design

```mermaid
flowchart LR
    A[Input Features] --> B[Masking Layer]
    B --> C[Noisy Features]
    C --> D[Transformer Encoder]
    D --> E[Hidden Representations]
    E --> F[k-NN Context Retrieval]
    F --> G[Context Features]
    G --> H[Transformer Decoder]
    H --> I[Reconstructed Features]
    I --> J[Prediction Head]
    J --> K[Output]
```

<!-- ### 2.2 Core Components

1. **Masking/Noising Layer**
 - Masks random features during training
 - Simulates missing data
 - Improves reliability

2. **Transformer Encoder** (`transformer_encoder`)
 - Encodes input features
 - Multi-head self-attention
 - Position-wise feedforward

3. **Context Retrieval** (k-NN)
 - Finds k nearest neighbors in training data
 - Provides contextual information
 - Improves predictions

4. **Transformer Decoder** (`decoder`)
 - Reconstructs features from context
 - Attention over context
 - Feature-level modeling

5. **Prediction Head** (`head`)
 - Aggregates representations
 - Outputs class logits
 - Task-specific predictions -->

### Pre-training strategy

```
Pre-training Phase (on diverse data):
  1. Mask random features (30-50%)
  2. Encode remaining features
  3. Retrieve k-NN context
  4. Predict masked features
  5. Loss = MSE(predicted, actual)

Fine-tuning Phase (on your task):
  1. Replace prediction head
  2. Fine-tune on task labels
  3. Use pre-trained encoder
```

## Inference parameters

### Complete parameter reference

```python
model_params = {
    # Architecture
    'd_model': 256,                        # Embedding dimension
    'num_heads': 8,                        # Attention heads
    'num_layers': 4,                       # Transformer layers
    'hidden_size': 512,                    # Feedforward hidden size
    'dropout': 0.1,                        # Dropout probability
    
    # Context retrieval
    'k_neighbors': 5,                      # Number of neighbors for context
    'context_mode': 'mixed',               # 'mixed' or 'features_only'
    
    # Inference behavior
    'n_ensembles': 8,                      # Multiple runs
    'temperature': 0.3,                    # Output scaling
    'mask_ratio': 0.3,                     # Feature masking during inference
    
    # Training
    'use_pretrain': True,                  # Use pre-trained weights
    'seed': 42                             # Reproducibility
}
```

### Parameter descriptions

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `d_model` | int | 256 | 128-512 | Transformer embedding dimension |
| `num_heads` | int | 8 | 4-16 | Number of attention heads |
| `num_layers` | int | 4 | 2-8 | Number of transformer layers |
| `hidden_size` | int | 512 | 256-1024 | Feedforward hidden dimension |
| `dropout` | float | 0.1 | 0.0-0.3 | Dropout probability |
| `k_neighbors` | int | 5 | 1-50 | k-NN context neighbors |
| `context_mode` | str | 'mixed' | 'mixed', 'features_only' | How to use context |
| `n_ensembles` | int | 8 | 1-16 | Number of ensemble runs |
| `temperature` | float | 0.3 | 0.1-1.0 | Output temperature |
| `use_pretrain` | bool | True | True/False | Use pre-trained weights |

### Architecture tuning

| Config | Speed | Accuracy | Memory | Best For |
|--------|-------|----------|--------|----------|
| Small: d=128, layers=2 | | | | Quick baseline |
| Medium: d=256, layers=4 | | | | Balanced |
| Large: d=512, layers=8 | | | | Max accuracy |

### Context modes

```python
context_modes = {
    'mixed': 'Use both context features and their representations',
    'features_only': 'Use only context features, not representations'
}

# Typically 'mixed' is better
model_params = {'context_mode': 'mixed'}
```

## Fine-tuning with TabDPT

TabDPT uses **episodic fine-tuning** with large context windows.

### Fine-tuning parameters

```python
tuning_params = {
    'device': 'cuda',
    'epochs': 3,                          # Few epochs needed (pre-trained)
    'learning_rate': 2e-5,                # Conservative learning rate
    'optimizer': 'adamw',                 # Optimizer type
    'scheduler': 'linear',                # Learning rate scheduler
    'warmup_steps': 500,                  # Extended warmup
    'weight_decay': 0.01,                 # L2 regularization
    'gradient_clip_value': 1.0,           # Gradient clipping
    
    # Large context for TabDPT
    'support_size': 1024,                 # Large context
    'query_size': 256,                    # Prediction samples
    'steps_per_epoch': 15,                # Gradient steps
    'batch_size': 32,                     # Standard batch
    
    'show_progress': True                 # Progress bar
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `support_size` | int | 1024 | Large context for k-NN |
| `query_size` | int | 256 | Query samples per episode |
| `steps_per_epoch` | int | 15 | Optimization steps |
| `batch_size` | int | 32 | Samples per batch |

### Fine-tuning guidelines

**Large Context Windows**:
```python
# TabDPT benefits from large context
tuning_params = {
    'support_size': 1024,  # Large context (TabDPT strength)
    'query_size': 256,     # Balance for gradients
    'batch_size': 32       # Process in parallel
}
```

**Learning Rate Strategy**:
- 1e-5: Conservative, safe
- 2e-5: Balanced (default)
- 5e-5: Aggressive

**Pre-training Advantage**:
- TabDPT needs fewer epochs due to pre-training
- Typically 3-5 epochs sufficient
- Convergence faster than TabICL

### Dataset recommendations

```python
# TabDPT shines with large datasets
dataset_sizes = {
    '10K': 'Acceptable, TabICL better',
    '100K': 'Good fit for TabDPT',
    '1M': 'Excellent for TabDPT',
    '5M+': 'Perfect use case'
}
```

## LoRA target modules

When using PEFT, TabDPT targets these modules:

```python
target_modules = [
    'transformer_encoder',     # Main encoder
    'encoder',                 # Additional encoder
    'y_encoder',               # Label encoder
    'head'                     # Prediction head
]
```

### Default PEFT configuration

```python
peft_config = {
    'r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'target_modules': None  # Uses defaults
}
```

### PEFT for large models

```python
# PEFT works well with TabDPT's large architecture
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-4,
        'support_size': 512,  # Still large
        'peft_config': {'r': 16}  # Higher rank acceptable
    }
)
```

## Usage patterns

### Inference only

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='inference',
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Base fine-tuning on large Dataset

```python
# TabDPT excels with large datasets
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-5,
        'support_size': 1024,  # Large context
        'query_size': 256,
        'steps_per_epoch': 15,
        'batch_size': 32,
        'show_progress': True
    }
)

pipeline.fit(X_train, y_train)  # 100K+ samples ideal
metrics = pipeline.evaluate(X_test, y_test)
```

### PEFT fine-tuning

```python
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-4,
        'support_size': 512,
        'query_size': 256,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        }
    }
)

pipeline.fit(X_train, y_train)
```

## Complete examples

### Large Dataset workflow

```python
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load large dataset (1M+ rows)
df = pd.read_csv('large_dataset.csv')  # 1M+ rows
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Train TabDPT
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-5,
        'support_size': 1024,
        'query_size': 256,
        'batch_size': 32,
        'show_progress': True
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Training completed on {len(X_train)} samples")
```

### Production model with PEFT - saving using joblib

```python
# PEFT for production deployment
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-4,
        'support_size': 512,
        'peft_config': {'r': 8}
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

# Save for deployment
pipeline.save('tabdpt_production.joblib')
```

### Architecture comparison

```python
from tabtune import TabularLeaderboard

# Compare architectures
lb = TabularLeaderboard(X_train, X_test, y_train, y_test)

# Small
lb.add_model(
    'TabDPT',
    'finetune',
    name='TabDPT-Small',
    model_params={'d_model': 128, 'num_layers': 2},
    tuning_params={'epochs': 3}
)

# Medium
lb.add_model(
    'TabDPT',
    'finetune',
    name='TabDPT-Medium',
    model_params={'d_model': 256, 'num_layers': 4},
    tuning_params={'epochs': 3}
)

# Large
lb.add_model(
    'TabDPT',
    'finetune',
    name='TabDPT-Large',
    model_params={'d_model': 512, 'num_layers': 8},
    tuning_params={'epochs': 3}
)

lb.run(rank_by='accuracy')
```

<!-- ## 8. Performance Characteristics -->

<!-- ### 8.1 Speed Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Inference (100K samples) | 10-15s | Context retrieval overhead |
| Fine-tuning (3 epochs, 1M) | 60-90m | Good scaling |
| Fine-tuning (PEFT, 1M) | 30-45m | Better efficiency |
| Prediction latency | 50-200ms | Per sample with context |

### Memory usage

| Scenario | Memory | GPU VRAM |
|----------|--------|---------|
| Inference | 8-12 GB | 6GB minimum |
| Base FT | 16-24 GB | 12GB recommended |
| PEFT | 10-14 GB | 8GB sufficient |
| Large model | Up to 32 GB | 16GB+ needed |

### Scalability

| Dataset Size | Training Time | Memory |
|--------------|---------------|--------|
| 100K | 10-15m | 12-16GB |
| 1M | 60-90m | 16-24GB |
| 5M | 200-300m | 24-32GB |
| 10M+ | 500m+ | 32GB+ |

### Accuracy profile

| Dataset Size | Accuracy |
|--------------|----------|
| Small (10K) | 82% (not optimal) |
| Medium (100K) | 90% (good) |
| Large (1M) | 92% |
| Very Large (5M+) | 93%+ (best) |

--- -->

## Troubleshooting

### "Context retrieval slow"
**Solution**: Reduce k_neighbors or use approximate k-NN
```python
model_params = {
    'k_neighbors': 3,  # Instead of 5
    'context_mode': 'features_only'
}
```

### "Out of memory with large support_size"
**Solution**: Use PEFT or reduce support size
```python
tuning_params = {
    'support_size': 512,  # Instead of 1024
    'batch_size': 16     # Smaller batch
}
```

### "Accuracy plateauing"
**Solution**: Increase training budget
```python
tuning_params = {
    'epochs': 5,        # More epochs
    'steps_per_epoch': 20,  # More steps
    'warmup_steps': 1000    # Longer warmup
}
```

### "Prediction latency too high"
**Solution**: Use smaller ensemble
```python
model_params = {
    'n_ensembles': 2,  # Instead of 8
    'k_neighbors': 3   # Fewer neighbors
}
```

## Comparison with other models

| Aspect | TabDPT | TabICL | Mitra | TabPFN |
|--------|--------|--------|-------|--------|
| Small data (10K) | Poor | Good | Good | Excellent |
| Large data (1M) | Excellent | Good | Okay | N/A |
| Accuracy | Excellent | Good | Excellent | Medium |
| Speed | Slow | Fast | Slow | Fastest |
| Memory | High | Moderate | Very High | Low |
| Pre-training | Yes | No | No | Yes |
| PEFT | Full | Full | Full | Exp |

## When to use TabDPT

**Use TabDPT when**:
- Dataset has 100K+ samples
- Maximum accuracy is priority
- Data has missing/noisy values
- You have sufficient memory
- Training time is not critical
- Deployment can handle context retrieval

**Don't use TabDPT for**:
- Small datasets (<50K)
- When prediction speed critical
- Very memory-constrained systems
- When training time is limited

<!-- ---

## Advanced topics

### k-NN optimization

```python
# Experiment with k_neighbors
for k in [1, 3, 5, 10]:
    pipeline = TabularPipeline(
        model_name='TabDPT',
        model_params={'k_neighbors': k}
    )
    # Evaluate...
```

### Masking strategy tuning

```python
# Adjust masking ratio for robustness
model_params = {
    'mask_ratio': 0.5,  # Higher = more robustness
    'use_pretrain': True
}
```

### Feature importance

```python
# Use attention weights for feature importance
attention = pipeline.model.get_attention_weights(X_test)
feature_importance = attention.mean(axis=0)

top_features = np.argsort(feature_importance)[-10:]
```

--- -->

## Quick reference

| Use Case | Strategy | Config | Support |
|----------|----------|--------|---------|
| Baseline | inference | default | 1024 |
| Production (1M data) | finetune | default | 1024 |
| Memory limited | peft | r=8 | 512 |
| Max accuracy | finetune | large model | 2048 |
| Fast inference | peft | r=4 | 256 |
