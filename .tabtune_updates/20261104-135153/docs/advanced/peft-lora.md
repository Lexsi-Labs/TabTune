# PEFT & LoRA: Parameter-Efficient Fine-Tuning for Tabular Models

This document provides an in-depth guide to Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) for TabTune models. Learn the theory, implementation, and best practices for memory-efficient model adaptation.

---

## 1. Introduction to PEFT and LoRA

### 1.1 What is PEFT?

**Parameter-Efficient Fine-Tuning (PEFT)** is a set of techniques to adapt large pre-trained models using only a small fraction of the total parameters, dramatically reducing:

- Memory consumption (90% reduction)
- Training time (2-3x speedup)
- Storage requirements (only store small adapters)

### 1.2 What is LoRA?

**Low-Rank Adaptation (LoRA)** is a specific PEFT technique that:

- Freezes pre-trained model weights
- Adds small trainable "adapter" layers
- Uses low-rank decomposition for efficiency
- Trains only 1-10% of parameters

### 1.3 Key Innovation

Instead of updating all weights, LoRA learns a low-rank approximation of weight updates:

\[
W' = W_0 + \Delta W = W_0 + BA
\]

Where:
- \(W_0\): Original frozen weights (large)
- \(\Delta W = BA\): Low-rank decomposition
- \(A\): Input projection (small)
- \(B\): Output projection (small)
- \(r\): Rank (typically 4-16, much smaller than weight dimensions)

---

## 2. Mathematical Foundation

### 2.1 Low-Rank Decomposition

For a weight matrix \(W \in \mathbb{R}^{d_{out} \times d_{in}}\), LoRA represents updates as:

\[
\Delta W = BA, \quad B \in \mathbb{R}^{d_{out} \times r}, A \in \mathbb{R}^{r \times d_{in}}
\]

**Complexity Reduction**:
- Full weights: \(d_{out} \times d_{in}\) parameters
- LoRA: \(r(d_{out} + d_{in})\) parameters
- Compression ratio: \(\frac{r(d_{out} + d_{in})}{d_{out} \times d_{in}}\)

**Example**: For a 768×768 weight matrix:
- Full: 589,824 parameters
- LoRA (r=8): 12,288 parameters
- Compression: 98% reduction

### 2.2 Scaling

To balance adaptation magnitude, LoRA scales the output:

\[
h = W_0 x + \alpha \frac{1}{r} B(Ax)
\]

Where \(\alpha\) (lora_alpha) controls the scaling factor \(\frac{\alpha}{r}\).

**Effect of Alpha**:
- Higher alpha: Larger adaptation magnitude
- Default: \(\alpha = 2r\) (empirically optimal)

### 2.3 Dropout

Dropout is applied to the input before LoRA projection for regularization:

\[
h = W_0 x + \alpha \frac{1}{r} B(\text{dropout}(Ax))
\]

---

## 3. LoRA in TabTune

### 3.1 LoRA Configuration

```python
peft_config = {
    'r': 8,                    # Rank (main hyperparameter)
    'lora_alpha': 16,          # Scaling factor
    'lora_dropout': 0.05,      # Dropout probability
    'target_modules': None,    # Modules to adapt (model default)
    'bias': 'none'             # Bias handling
}
```

### 3.2 LoRA Linear Layer

TabTune implements `LoRALinear` that wraps standard PyTorch linear layers:

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear          # Frozen base layer
        self.lora_A = nn.Linear(..., r)  # Adapter A
        self.lora_B = nn.Linear(r, ...)  # Adapter B
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        
    def forward(self, x):
        # Base forward (no gradients)
        base_out = self.base(x)
        # LoRA forward
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out
```

### 3.3 Weight Freezing

- Base model weights: `requires_grad=False`
- LoRA adapters: `requires_grad=True`
- Enables gradient computation only on adapters

---

## 4. LoRA Hyperparameter Tuning

### 4.1 Rank Selection

The rank `r` is the most critical hyperparameter:

| Rank | Parameters | Memory | Accuracy | Speed |
|------|-----------|--------|----------|-------|
| r=2 | Minimal | Very Low | Lower | ⭐⭐⭐⭐⭐ |
| r=4 | Low | Low | Good | ⭐⭐⭐⭐⭐ |
| r=8 | Moderate | Moderate | Better | ⭐⭐⭐⭐ |
| r=16 | High | High | Best | ⭐⭐⭐ |
| r=32 | Very High | Very High | Optimal | ⭐⭐ |

**Guidelines**:
- **Small data (10K)**: r=4 (enough for adaptation)
- **Medium data (100K)**: r=8 (balanced)
- **Large data (1M)**: r=16 (more expressive)
- **Very constrained**: r=2 (minimum viable)

**Rule of Thumb**:
\[
r = \max(4, \frac{\text{dataset\_size}}{50000})
\]

### 4.2 Alpha Selection

Alpha controls the magnitude of LoRA contribution:

```python
# Recommended: alpha = 2 * rank
peft_config = {
    'r': 8,
    'lora_alpha': 16,  # = 2 * 8
    'lora_dropout': 0.05
}
```

**Effects**:
- **Alpha too low**: Adaptation weak, training slow
- **Alpha = 2r**: Empirically optimal
- **Alpha too high**: Training unstable, may diverge

### 4.3 Dropout Probability

LoRA dropout acts as regularization:

```python
lora_dropout_values = {
    0.0: 'No regularization (may overfit)',
    0.05: 'Light regularization (default)',
    0.1: 'Moderate regularization',
    0.2: 'Strong regularization (for small data)'
}
```

**Selection**:
- **Large data**: 0.05 (default)
- **Small data**: 0.1-0.2 (prevent overfitting)
- **Already regularized model**: 0.0-0.05

---

## 5. Target Module Selection

### 5.1 Module Hierarchy

TabTune identifies target modules via pattern matching:

```python
# LoRA targets linear transformation layers
target_modules = {
    'column_embeddings': 'Column feature processing',
    'row_attention': 'Row-wise interactions',
    'prediction_head': 'Final prediction',
    'decoder': 'Feature reconstruction'
}
```

### 5.2 Model-Specific Defaults

Each model has pre-configured target modules optimized for LoRA:

**TabICL**:
```python
target_modules = [
    'col_embedder.tf_col',
    'row_interactor',
    'icl_predictor.tf_icl',
    'icl_predictor.decoder'
]
```

**TabDPT**:
```python
target_modules = [
    'transformer_encoder',
    'encoder',
    'y_encoder',
    'head'
]
```

**Mitra**:
```python
target_modules = [
    'x_embedding',
    'layers',
    'final_layer'
]
```

### 5.3 Custom Target Selection

Override defaults for specific needs:

```python
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'target_modules': [
                'col_embedder.tf_col',  # Only column embedder
                'icl_predictor.decoder'   # Plus decoder
            ]
        }
    }
)
```

---

## 6. LoRA Training

### 6.1 Typical Training Loop

```python
from tabtune import TabularPipeline

# Create pipeline with PEFT
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,  # Typically higher than base-ft
        'peft_config': {'r': 8, 'lora_alpha': 16}
    }
)

# Training (only adapters are updated)
pipeline.fit(X_train, y_train)

# Inference
predictions = pipeline.predict(X_test)
```

### 6.2 Learning Rate Strategy

LoRA uses different learning rates than base fine-tuning:

```python
# Base fine-tuning (all parameters)
learning_rate_base_ft = 2e-5

# LoRA fine-tuning (small parameters)
learning_rate_peft = 2e-4  # 10x higher typical

# Rationale: Smaller parameter updates need larger learning rates
```

### 6.3 Optimizer Configuration

```python
# LoRA-specific optimizer settings
optimizer_config = {
    'optimizer': 'adamw',
    'learning_rate': 2e-4,
    'weight_decay': 0.01,
    'eps': 1e-8,
    'betas': (0.9, 0.999)
}
```

---

## 7. Memory Analysis

### 7.1 Memory Breakdown

**Base Fine-Tuning** (full model):
```
Model weights:     500 MB
Optimizer states:  1 GB  (Adam: 2x weights)
Gradients:         500 MB
Activations:       200 MB
─────────────────────────
Total:             ~2.2 GB per forward/backward
```

**LoRA Fine-Tuning** (adapters only):
```
Model weights:     500 MB (frozen, no gradients)
LoRA adapters:     5 MB
Optimizer states:  10 MB (only adapters)
Gradients:         5 MB
Activations:       200 MB
─────────────────────────
Total:             ~700 MB per forward/backward
~70% reduction
```

### 7.2 Practical Memory Savings

| Model | Base-FT | LoRA | Savings |
|-------|---------|------|---------|
| TabICL | 12 GB | 4 GB | 66% |
| TabDPT | 24 GB | 8 GB | 66% |
| Mitra | 20 GB | 6 GB | 70% |

<!-- 
## 8. Performance Impact

### 8.1 Accuracy Trade-off

LoRA typically shows minor accuracy loss compared to base fine-tuning:

```python
# Typical results on TabICL
base_ft_accuracy = 90.5%
lora_accuracy = 89.8%
loss = -0.7%  # Acceptable trade-off for 66% memory savings
```

**Accuracy vs Memory**: Empirical findings
- r=4: 91% accuracy, 50% memory
- r=8: 89.8% accuracy, 35% memory
- r=16: 90.2% accuracy, 45% memory

### 8.2 Training Speed

| Strategy | Time | Notes |
|----------|------|-------|
| Base-FT | 100% | Baseline |
| LoRA | 40-60% | Faster due to fewer parameters |
| PEFT+Mixed Precision | 25-35% | Optimal speed |

### 8.3 Inference Speed

- **Base-FT**: Baseline
- **LoRA**: +5-10% overhead (LoRA forward pass)
- **With caching**: Negligible difference
 -->
---

## 9. Complete Example

### 9.1 Memory-Constrained Scenario

```python
from tabtune import TabularPipeline
import torch

# Check available GPU memory
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# LoRA for 4GB GPU
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'batch_size': 16,
        'peft_config': {
            'r': 4,              # Lower rank for less memory
            'lora_alpha': 8,
            'lora_dropout': 0.1
        }
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
print(f"LoRA Accuracy: {metrics['accuracy']:.4f}")
```

### 9.2 Rank Exploration

```python
from tabtune import TabularLeaderboard

# Compare different LoRA ranks
lb = TabularLeaderboard(X_train, X_test, y_train, y_test)

for r in [2, 4, 8, 16]:
    lb.add_model(
        'TabICL',
        'peft',
        name=f'LoRA-r{r}',
        tuning_params={
            'epochs': 5,
            'peft_config': {'r': r, 'lora_alpha': 2*r}
        }
    )

results = lb.run(rank_by='accuracy')
print(lb.get_ranking())
```

### 9.3 Memory-Speed Trade-off

```python
# Explore memory-speed-accuracy trade-off
configs = [
    {'r': 2, 'name': 'Ultra-Light'},
    {'r': 4, 'name': 'Light'},
    {'r': 8, 'name': 'Medium'},
    {'r': 16, 'name': 'Heavy'}
]

for config in configs:
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 3,
            'peft_config': {'r': config['r']}
        }
    )
    
    # Time training
    import time
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    
    # Get memory usage
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    # Evaluate
    metrics = pipeline.evaluate(X_test, y_test)
    
    print(f"{config['name']:12} | Rank: {config['r']:2} | "
          f"Time: {elapsed:6.1f}s | Memory: {mem:5.1f}GB | "
          f"Accuracy: {metrics['accuracy']:.4f}")
```

---

## 10. Saving and Loading LoRA Models

### 10.1 Save LoRA Adapters Only

```python
import torch

# Save only LoRA adapter weights (minimal storage)
lora_state = {
    'rank': 8,
    'alpha': 16,
    'lora_a': pipeline.model.lora_A.state_dict(),
    'lora_b': pipeline.model.lora_B.state_dict()
}

torch.save(lora_state, 'lora_adapters.pt')  # ~1-5 MB
```

### 10.2 Load and Merge

```python
# Load adapters and merge with base model
lora_state = torch.load('lora_adapters.pt')

# Merge LoRA into base weights (optional, for inference optimization)
merged_weights = base_weights + (lora_B @ lora_A) * alpha / r
```

### 10.3 Full Pipeline Serialization

```python
# Save complete pipeline with LoRA adapters
pipeline.save('pipeline_with_lora.joblib')

# Load and use
loaded = TabularPipeline.load('pipeline_with_lora.joblib')
predictions = loaded.predict(X_test)
```

---

## 11. Troubleshooting

### Issue: "LoRA accuracy much lower than base-FT"
**Solution**: Increase rank
```python
peft_config = {
    'r': 16,  # Instead of 8
    'lora_alpha': 32
}
```

### Issue: "Training diverging with LoRA"
**Solution**: Reduce learning rate
```python
tuning_params = {
    'learning_rate': 1e-4,  # Instead of 2e-4
    'warmup_steps': 500
}
```

### Issue: "Still out of memory with LoRA"
**Solution**: Further reduce parameters
```python
peft_config = {
    'r': 2,  # Minimum viable
    'lora_dropout': 0.2  # Stronger regularization
}

tuning_params = {
    'batch_size': 4  # Smaller batches
}
```

### Issue: "LoRA inference slow"
**Solution**: Use merged weights
```python
# Merge LoRA weights into base after training
merged_model = merge_lora_weights(pipeline.model)
```

---

<!-- ## 12. Advanced Topics

### 12.1 LoRA+ Variant

Enhanced LoRA with different learning rates per layer:

```python
# LoRA+ uses higher LR for B matrix
lora_plus_config = {
    'lr_ratio': 10,  # B matrix gets 10x higher LR
    'lora_alpha': 16,
    'r': 8
}
```

### 12.2 QLoRA (Quantized LoRA)

Combine LoRA with quantization for extreme compression:

```python
# 4-bit quantized model with LoRA
quantization_config = {
    'load_in_4bit': True,
    'bnb_4bit_compute_dtype': 'float16'
}
# Reduces model size by 4x more
```

### 12.3 Multi-LoRA (Mixture of Adapters)

Multiple task-specific LoRA adapters:

```python
# Switch between different LoRA adapters
task_loras = {
    'classification': 'lora_classification.pt',
    'regression': 'lora_regression.pt',
    'clustering': 'lora_clustering.pt'
}

# Load task-specific adapter
selected_adapter = load_lora_adapter(task_loras['classification'])
```
 -->
---

## 13. Best Practices

### ✅ Do's

- ✅ Start with r=8 (good default)
- ✅ Use 2x learning rate for LoRA vs base-FT
- ✅ Include warmup phase (prevent instability)
- ✅ Monitor gradient norms
- ✅ Use gradient clipping
- ✅ Save adapter weights separately
- ✅ Test rank selection with leaderboard

### ❌ Don'ts

- ❌ Don't use same learning rate as base-FT
- ❌ Don't train very low ranks (r<2) without good reason
- ❌ Don't skip regularization on small data
- ❌ Don't forget to freeze base model
- ❌ Don't use LoRA on tiny models (overhead not worth it)

---

## 14. Comparison: Base-FT vs LoRA

| Aspect | Base-FT | LoRA | Winner |
|--------|---------|------|--------|
| Accuracy | High | Medium-High | Base-FT (~1% better) |
| Memory | Very High | Low | LoRA (70% savings) |
| Speed | Slow | Fast | LoRA (2-3x faster) |
| Storage | Huge | Tiny | LoRA (100x smaller) |
| Scalability | Limited | Excellent | LoRA |
| Production | Complex | Simple | LoRA |
| Learning Curve | Medium | Low | LoRA |

---

## 15. Quick Reference

| Task | r | Alpha | Dropout | LR |
|------|---|-------|---------|-----|
| Small data (10K) | 4 | 8 | 0.1 | 2e-4 |
| Medium data (100K) | 8 | 16 | 0.05 | 2e-4 |
| Large data (1M) | 16 | 32 | 0.02 | 1e-4 |
| Memory limited | 2 | 4 | 0.2 | 1e-4 |
| Max accuracy | 16 | 32 | 0.05 | 5e-5 |

---

## 16. Next Steps

- [Tuning Strategies](../user-guide/tuning-strategies.md) - Compare strategies
- [Hyperparameter Tuning](hyperparameter-tuning.md) - Full optimization guide
- [Models Overview](../models/overview.md) - PEFT support per model
- [TabularLeaderboard](../user-guide/leaderboard.md) - Compare configurations

---

LoRA enables efficient fine-tuning of large tabular models. Use it for memory-constrained environments while maintaining strong performance!