# Tuning Strategies

TabTune provides three distinct tuning strategies to accommodate different use cases, computational budgets, and performance requirements. This guide explains each strategy in detail, including when to use them and their tradeoffs.

---

## 1. Overview

| Strategy | Training | Use Case | Memory | Speed | Accuracy |
|----------|----------|----------|--------|-------|----------|
| **inference** | None | Baseline, zero-shot | Minimal | Fast | Baseline |
| **base-ft** | Full params | High accuracy, ample resources | High | Slow | Highest |
| **peft** | LoRA adapters | Memory-constrained, iteration | Low | Medium | High |

---

## 2. Inference Strategy

### Definition
**Zero-shot inference** using pre-trained model weights without any training on your data.

### Use Cases
- Quick baseline comparisons
- Evaluating out-of-the-box model performance
- Time-constrained scenarios
- Testing data preprocessing pipeline

### Workflow

```mermaid
flowchart LR
    A[Raw Data] --> B[DataProcessor]
    B --> C[Load Pre-trained Model]
    C --> D[Forward Pass Only]
    D --> E[Predictions]
```

### Implementation

```python
from tabtune import TabularPipeline

# No training occurs
pipeline = TabularPipeline(
    model_name='TabPFN',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda'}
)

# fit() only applies preprocessing; no model training
pipeline.fit(X_train, y_train)

# Direct prediction on test data
predictions = pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test)
```

### Advantages
- ✅ No training time needed
- ✅ Minimal memory footprint
- ✅ Immediate results
- ✅ Good for baseline comparisons

### Disadvantages
- ❌ Generic pre-trained weights may not fit your data
- ❌ Typically lower accuracy than fine-tuned models
- ❌ Cannot adapt to task-specific patterns

### Performance Profile
- **Training time**: 0 seconds
- **Memory usage**: 2-4 GB (model + data)
- **Inference latency**: 10-50 ms per batch

### Example with All Models

```python
from tabtune import TabularPipeline

models = ['TabPFN', 'TabICL', 'TabDPT', 'Mitra', 'ContextTab', 'OrionMSP','OrionBix']

for model in models:
    pipeline = TabularPipeline(
        model_name=model,
        tuning_strategy='inference'
    )
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_test, y_test)
    print(f"{model} - Accuracy: {metrics['accuracy']:.4f}")
```

---

## 3. Base Fine-Tuning Strategy (`base-ft`)

### Definition
**Full-parameter fine-tuning** where all model weights are updated during training.

### Use Cases
- Maximum accuracy is priority
- Abundant computational resources (GPU, RAM)
- Large training datasets (>100K samples)
- Production models requiring best performance
- Transfer learning from related domains

### Workflow

```mermaid
flowchart LR
    A[Raw Data] --> B[DataProcessor]
    B --> C[Load Pre-trained Model]
    C --> D[Update ALL Parameters]
    D --> E[Training Loop]
    E --> F[Fine-tuned Model]
    F --> G[Predictions]
```

### Implementation

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='OrionMSP',
    task_type='classification',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5,
        'batch_size': 32,
        'show_progress': True,
        'gradient_accumulation_steps': 2,  # optional
        'mixed_precision': 'fp16'  # optional
    }
)

# Full training occurs during fit()
pipeline.fit(X_train, y_train)

# Use fine-tuned model for predictions
predictions = pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test)
```

### Supported Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | 'cpu' | 'cuda' or 'cpu' |
| `epochs` | int | 3 | Number of training epochs |
| `learning_rate` | float | 2e-5 | Optimizer learning rate |
| `batch_size` | int | 32 | Samples per batch |
| `optimizer` | str | 'adamw' | 'adamw' or 'sgd' |
| `show_progress` | bool | True | Display training progress bar |

### Advantages
- ✅ Highest accuracy potential
- ✅ Fully adapts to task-specific patterns
- ✅ Works with any dataset size
- ✅ Best for production models
- ✅ Supports all hyperparameter tuning

### Disadvantages
- ❌ High memory consumption (8-16GB+)
- ❌ Long training time (hours for large models)
- ❌ Risk of overfitting on small datasets
- ❌ Requires careful hyperparameter tuning
- ❌ GPU memory can become bottleneck

### Performance Profile
- **Training time**: 30 minutes - 2 hours (depending on dataset)
- **Memory usage**: 12-24 GB (full model + gradients + optimizer states)
- **Inference latency**: 10-50 ms per batch

### Training Loop Details

The training process follows this pattern:

1. **Initialize optimizer** (AdamW with weight decay)
2. **For each epoch**:
   - Shuffle training data
   - **For each batch**:
     - Forward pass through model
     - Compute loss
     - Backward pass (compute gradients)
     - Clip gradients if specified
     - Update weights
     - Update learning rate scheduler
   - Validate on development set (if available)
3. **Save best checkpoint** based on validation metric
4. **Return fine-tuned model**

### Example: Full Training Pipeline

```python
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split

# Load and split data
X, y = load_your_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Configure base fine-tuning
pipeline = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 10,
        'learning_rate': 1e-5,
        'batch_size': 64,
        'scheduler': 'cosine',
        'warmup_steps': 500,
        'mixed_precision': 'fp16',
        'show_progress': True,
        'save_checkpoint_path': 'best_model.pt'
    }
)

# Train on training data
pipeline.fit(X_train, y_train)

# Evaluate on validation set
val_metrics = pipeline.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")

# Save for later use
pipeline.save('fintuned_pipeline.joblib')
```

---

## 4. PEFT Fine-Tuning Strategy (`peft`)

### Definition
**Parameter-Efficient Fine-Tuning using LoRA (Low-Rank Adaptation)** where only small adapter weights are trained while base model is frozen.

### How LoRA Works

<!-- Instead of updating all weights, LoRA adds trainable low-rank matrices:

\[
\text{output} = W_0 \cdot x + \frac{\alpha}{r} \cdot B(A \cdot x)
\]

Where:
- \(W_0\) is the frozen pre-trained weight
- \(A \in \mathbb{R}^{d_{in} \times r}\) and \(B \in \mathbb{R}^{r \times d_{out}}\) are trainable
- \(r \ll \min(d_{in}, d_{out})\) is the rank

This reduces trainable parameters from millions to thousands. -->

### Use Cases
- Limited GPU memory (< 8 GB)
- Quick iteration cycles
- Fine-tuning multiple models simultaneously
- Rapid experimentation
- Deployment with minimal storage

### Workflow

```mermaid
flowchart LR
    A[Raw Data] --> B[DataProcessor]
    B --> C[Load Pre-trained Model]
    C --> D[Inject LoRA Adapters]
    D --> E[Update ONLY Adapters]
    E --> F[Training Loop]
    F --> G[Model + LoRA Weights]
    G --> H[Predictions]
```

### Implementation

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='Mitra',
    task_type='classification',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'target_modules': None  # Uses model defaults if None
        },
        'show_progress': True
    }
)

# Training with LoRA adapters
pipeline.fit(X_train, y_train)

# Predictions with adapted model
predictions = pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test)
```

### PEFT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | LoRA rank (lower = more compression) |
| `lora_alpha` | int | 16 | Scaling factor for LoRA output |
| `lora_dropout` | float | 0.05 | Dropout applied to LoRA input |
| `target_modules` | list | None | Which linear layers to adapt (None = use defaults) |

### Model-Specific LoRA Targets

TabTune pre-configures optimal target modules per model:

**TabICL/OrionBix**:
```
col_embedder.tf_col, row_interactor, icl_predictor.tf_icl, icl_predictor.decoder
```

**TabDPT**:
```
transformer_encoder, encoder, y_encoder, head
```

**Mitra**:
```
x_embedding, layers, final_layer
```

**ContextTab**:
```
in_context_encoder, dense, output_head, embeddings
```

**TabPFN** (⚠️ Experimental):
```
encoder.5.layer, y_encoder.2.layer, transformer_encoder.layers, decoder_dict.standard
```

### Advantages
- ✅ 90% memory reduction vs base-ft
- ✅ 2-3x faster training
- ✅ Only stores small adapter weights
- ✅ Can run on 4GB GPUs
- ✅ Fast iteration for experimentation

### Disadvantages
- ❌ Slightly lower accuracy than base-ft (~2-5% in practice)
- ❌ Not all model layers adapted (frozen backbone limits flexibility)
- ❌ May struggle with very different tasks
- ❌ Experimental support on TabPFN and ContextTab

### Performance Profile
- **Training time**: 10-30 minutes
- **Memory usage**: 3-6 GB (adapters + activations only)
- **Inference latency**: 10-50 ms per batch
- **Model size**: Original model size + 1-2% (adapters)

### Parameter Tuning Guidelines

**Rank Selection**:
```
r = 4   → Highest compression, faster, lower accuracy
r = 8   → Good balance (default)
r = 16  → More expressive, slower, higher accuracy
r = 32  → Close to base-ft, but still compressed
```

**Alpha Selection**:
```
lora_alpha should typically be 2x the rank
r=8 → lora_alpha=16
r=16 → lora_alpha=32
```

**Dropout Selection**:
```
lora_dropout=0.0  → No regularization
lora_dropout=0.05 → Light regularization (default)
lora_dropout=0.1  → Strong regularization
```

### Example: PEFT Training with Hyperparameter Tuning

```python
from tabtune import TabularPipeline

# Experiment with different LoRA ranks
for r in [4, 8, 16]:
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 5,
            'learning_rate': 2e-4,
            'peft_config': {
                'r': r,
                'lora_alpha': 2 * r,
                'lora_dropout': 0.05
            }
        }
    )
    
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_test, y_test)
    print(f"Rank {r}: Accuracy = {metrics['accuracy']:.4f}")
```

---

## 5. Strategy Comparison & Decision Tree

### Quick Comparison Table

| Aspect | inference | base-ft | peft |
|--------|-----------|---------|------|
| Training | No | Yes, all params | Yes, adapters only |
| Memory | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Cost | Free | High GPU cost | Low GPU cost |
| Production | ❌ | ✅ | ✅ |

### Decision Tree

```
Start: Which strategy?
│
├─ "I want instant results, no training" → inference
│  └─ Best for: Baseline, quick exploration
│
├─ "I have limited resources (<8GB GPU)" → peft
│  └─ Best for: Rapid iteration, memory-constrained
│
└─ "I need best accuracy, have resources" → base-ft
   └─ Best for: Production, large datasets, high accuracy
```

---



## 6. Best Practices

1. **Start with inference** to establish baseline
2. **Use PEFT for exploration** when resources are limited
3. **Switch to base-ft for production** models
4. **Monitor for overfitting** on small datasets
5. **Save checkpoints** for long training runs
6. **Use validation set** to track progress
8. **Start with default hyperparameters** then tune

---

## 7. Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use PEFT strategy

### Issue: "Accuracy decreasing during training"
**Solution**: Lower learning rate, reduce epochs, use regularization

### Issue: "Model not improving after training"
**Solution**: Increase learning rate, use different scheduler, increase epochs

### Issue: "PEFT not significantly faster"
**Solution**: Use lower rank (r=4), verify LoRA is actually applied

---

## 8. Next Steps

- [PEFT & LoRA Details](../advanced/peft-lora.md) - Deep dive into LoRA theory
- [Hyperparameter Tuning](../advanced/hyperparameter-tuning.md) - Optimize model performance
- [Model Selection](model-selection.md) - Choose right model for your task

---

Choose the right strategy for your use case and resource constraints!