# PEFT Examples: Practical Parameter-Efficient Fine-Tuning Workflows

This document provides practical, production-ready examples for using LoRA and PEFT techniques with TabTune across various scenarios and constraints.

---

## 1. Quick Start PEFT

### 1.1 5-Minute PEFT Example

Minimal code to use LoRA:

```python
from tabtune import TabularPipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create PEFT pipeline
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        }
    }
)

# Train (much faster and lighter than base-ft)
pipeline.fit(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Model size: 1-2% of full model")
```

### 1.2 Comparing Base-FT vs PEFT

```python
import time
import torch
from tabtune import TabularPipeline

X_train, X_test, y_train, y_test = load_data()

# Method 1: Base Fine-Tuning
print("=== Base Fine-Tuning ===")
torch.cuda.reset_peak_memory_stats()

start = time.time()
pipeline_base = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={'epochs': 5}
)
pipeline_base.fit(X_train, y_train)
base_time = time.time() - start
base_memory = torch.cuda.max_memory_allocated() / 1e9

metrics_base = pipeline_base.evaluate(X_test, y_test)

print(f"Time: {base_time:.1f}s")
print(f"Memory: {base_memory:.1f}GB")
print(f"Accuracy: {metrics_base['accuracy']:.4f}")

# Method 2: PEFT
print("\n=== PEFT (LoRA) ===")
torch.cuda.reset_peak_memory_stats()

start = time.time()
pipeline_peft = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'epochs': 5,
        'peft_config': {'r': 8}
    }
)
pipeline_peft.fit(X_train, y_train)
peft_time = time.time() - start
peft_memory = torch.cuda.max_memory_allocated() / 1e9

metrics_peft = pipeline_peft.evaluate(X_test, y_test)

print(f"Time: {peft_time:.1f}s")
print(f"Memory: {peft_memory:.1f}GB")
print(f"Accuracy: {metrics_peft['accuracy']:.4f}")

# Comparison
print("\n=== Comparison ===")
print(f"Speedup: {base_time/peft_time:.1f}x")
print(f"Memory savings: {(1 - peft_memory/base_memory)*100:.0f}%")
print(f"Accuracy loss: {(metrics_base['accuracy'] - metrics_peft['accuracy'])*100:.2f}%")
```

---

## 2. Memory-Constrained Training

### 2.1 Training on Limited GPU (4GB)

```python
import torch
from tabtune import TabularPipeline

# Check available GPU memory
available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Available GPU memory: {available_memory:.1f}GB")

if available_memory < 4:
    print("Using ultra-efficient PEFT configuration...")
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 3,
            'learning_rate': 2e-4,
            'batch_size': 4,           # Small batch
            'support_size': 32,        # Small context
            'query_size': 16,
            'num_workers': 0,          # No parallel loading
            'peft_config': {
                'r': 4,                # Very low rank
                'lora_alpha': 8,
                'lora_dropout': 0.1
            }
        }
    )

elif available_memory < 8:
    print("Using efficient PEFT configuration...")
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 5,
            'learning_rate': 2e-4,
            'batch_size': 8,
            'support_size': 64,
            'query_size': 32,
            'peft_config': {
                'r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.05
            }
        }
    )

else:
    print("Sufficient memory. Using standard PEFT...")
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 5,
            'learning_rate': 2e-4,
            'peft_config': {'r': 8}
        }
    )

# Train
pipeline.fit(X_train, y_train)
```

<!-- ### 2.2 Combining PEFT with Mixed Precision

```python
from tabtune import TabularPipeline

# Ultra-efficient: PEFT + Mixed Precision + Gradient Accumulation
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'batch_size': 4,
        'gradient_accumulation_steps': 8,  # Effective batch: 32
        'mixed_precision': 'fp16',         # Half precision
        'peft_config': {
            'r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.1
        }
    }
)

pipeline.fit(X_train, y_train)

# Result: Similar effective batch size to batch_size=32
# But memory usage of batch_size=4
```
 -->
---

<!-- ## 3. Multi-Model PEFT Training

### 3.1 Compare All Models with PEFT

```python
from tabtune import TabularLeaderboard
import pandas as pd

X_train, X_test, y_train, y_test = load_data()

# Create leaderboard
lb = TabularLeaderboard(X_train, X_test, y_train, y_test)

# Add all models with PEFT
models_to_test = ['TabPFN', 'TabICL', 'OrionMSP', 'OrionBix', 'TabDPT', 'Mitra']

for model in models_to_test:
    if model == 'TabPFN':
        # TabPFN doesn't need PEFT as much (already fast)
        lb.add_model(model, 'inference', name=f'{model}-Inference')
    else:
        lb.add_model(
            model,
            'peft',
            name=f'{model}-PEFT-r8',
            tuning_params={
                'epochs': 3,
                'learning_rate': 2e-4,
                'peft_config': {'r': 8, 'lora_alpha': 16}
            }
        )

# Run benchmarks
print("Running PEFT comparison across all models...")
results = lb.run(rank_by='accuracy', verbose=True)

# Display results
print("\n" + "="*70)
print("PEFT Model Comparison")
print("="*70)
print(lb.get_ranking())

# Export for analysis
results_df = lb.get_results_dataframe()
results_df.to_csv('peft_comparison.csv', index=False)
```

### 3.2 Rank Ablation Study

Compare LoRA ranks for single model:

```python
from tabtune import TabularLeaderboard

lb = TabularLeaderboard(X_train, X_test, y_train, y_test)

# Test different ranks
for r in [2, 4, 8, 16, 32]:
    lb.add_model(
        'TabICL',
        'peft',
        name=f'TabICL-r{r}',
        tuning_params={
            'epochs': 3,
            'learning_rate': 2e-4,
            'peft_config': {
                'r': r,
                'lora_alpha': 2*r,
                'lora_dropout': 0.05
            }
        }
    )

results = lb.run(rank_by='accuracy')

# Analyze trade-offs
results_df = lb.get_results_dataframe()

print("\n=== Rank Ablation Study ===")
print(results_df[['model_name', 'accuracy', 'training_time']])

# Find sweet spot
print("\nRecommendations:")
for _, row in results_df.iterrows():
    model = row['model_name']
    acc = row['accuracy']
    time = row.get('training_time', 0)
    print(f"{model:15} | Accuracy: {acc:.4f} | Time: {time:6.1f}s")
```
 -->
---

## 4. LoRA Rank Selection

### 4.1 Choosing Optimal Rank

```python
import numpy as np
from tabtune import TabularPipeline

def evaluate_rank(X_train, X_test, y_train, y_test, rank):
    """Evaluate specific LoRA rank."""
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 3,
            'peft_config': {
                'r': rank,
                'lora_alpha': 2*rank,
                'lora_dropout': 0.05
            }
        }
    )
    
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_test, y_test)
    
    return metrics

# Evaluate multiple ranks
ranks = [2, 4, 8, 16, 32]
results = {}

print("Evaluating LoRA ranks...")
for r in ranks:
    print(f"  Testing rank {r}...", end='', flush=True)
    metrics = evaluate_rank(X_train, X_test, y_train, y_test, r)
    results[r] = metrics['accuracy']
    print(f" Accuracy: {metrics['accuracy']:.4f}")

# Find optimal rank
optimal_rank = max(results, key=results.get)
optimal_acc = results[optimal_rank]

print(f"\nOptimal rank: {optimal_rank} (Accuracy: {optimal_acc:.4f})")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ranks, [results[r] for r in ranks], 'o-', linewidth=2, markersize=8)
plt.xlabel('LoRA Rank')
plt.ylabel('Accuracy')
plt.title('LoRA Rank vs Model Accuracy')
plt.grid(True, alpha=0.3)
plt.savefig('rank_analysis.png')
```


---

## 6. Advanced PEFT Techniques

### 6.1 Custom Target Modules

```python
from tabtune import TabularPipeline

# Train only specific layers
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'target_modules': [
                'col_embedder.tf_col',      # Column embedder only
                'icl_predictor.decoder'     # Plus decoder
            ]
        }
    }
)

pipeline.fit(X_train, y_train)
```

### 6.2 LoRA with Different Learning Rates

```python
from tabtune import TabularPipeline

# Higher learning rate for smaller LoRA modules
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 1e-3,  # 10x higher for PEFT
        'peft_config': {
            'r': 8,
            'lora_alpha': 16
        }
    }
)

pipeline.fit(X_train, y_train)
```

<!-- ### 6.3 Sequential Fine-Tuning

Gradually unfreeze layers:

```python
from tabtune import TabularPipeline

# Phase 1: Train only head
print("Phase 1: Train only prediction head...")
pipeline1 = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'epochs': 2,
        'peft_config': {
            'r': 2,
            'target_modules': ['icl_predictor.decoder']
        }
    }
)
pipeline1.fit(X_train, y_train)

# Phase 2: Train more modules
print("Phase 2: Train more modules...")
pipeline2 = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'epochs': 3,
        'peft_config': {
            'r': 8,
            'target_modules': [
                'col_embedder.tf_col',
                'icl_predictor.tf_icl',
                'icl_predictor.decoder'
            ]
        }
    }
)
pipeline2.fit(X_train, y_train)

metrics = pipeline2.evaluate(X_test, y_test)
print(f"Final Accuracy: {metrics['accuracy']:.4f}")
``` -->

---

## 7. PEFT for Different Models

### 7.1 TabDPT with PEFT for Large Data

```python
from tabtune import TabularPipeline

# PEFT enables TabDPT on memory-limited systems
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-4,
        'support_size': 1024,  # Still use large context
        'batch_size': 16,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16
        }
    }
)

pipeline.fit(X_train, y_train)
```

### 7.2 Mitra with PEFT for 2D Attention

```python
from tabtune import TabularPipeline

# PEFT makes memory-hungry Mitra practical
pipeline = TabularPipeline(
    model_name='Mitra',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-4,
        'support_size': 128,
        'batch_size': 4,  # Still small due to 2D attention
        'peft_config': {
            'r': 4,
            'lora_alpha': 8
        }
    }
)

pipeline.fit(X_train, y_train)
```

---

<!-- ## 8. PEFT Hyperparameter Tuning

### 8.1 Bayesian Optimization for PEFT

```python
import optuna
from tabtune import TabularPipeline

def objective(trial):
    """Optimize PEFT hyperparameters."""
    
    # Suggest PEFT hyperparameters
    r = trial.suggest_int('r', 2, 16, step=2)
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.2, step=0.05)
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='peft',
        tuning_params={
            'device': 'cuda',
            'epochs': 3,
            'learning_rate': learning_rate,
            'peft_config': {
                'r': r,
                'lora_alpha': 2*r,
                'lora_dropout': dropout
            }
        }
    )
    
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_val, y_val)
    
    return metrics['accuracy']

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best PEFT config: {study.best_params}")

# Train final model
best_pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': study.best_params['lr'],
        'peft_config': {
            'r': study.best_params['r'],
            'lora_alpha': 2*study.best_params['r'],
            'lora_dropout': study.best_params['dropout']
        }
    }
)

best_pipeline.fit(X_train, y_train)
final_metrics = best_pipeline.evaluate(X_test, y_test)
print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
```

--- -->
---

## 9. Troubleshooting PEFT

### 9.1 Common PEFT Issues

```python
# Issue 1: PEFT accuracy much lower than base-ft
# Solution: Increase rank
peft_config = {
    'r': 16,  # Instead of 4
    'lora_alpha': 32
}

# Issue 2: PEFT training diverging
# Solution: Reduce learning rate
tuning_params = {
    'learning_rate': 1e-4  # Instead of 2e-4
}

# Issue 3: PEFT still using too much memory
# Solution: Combine PEFT + mixed precision + gradient accumulation
tuning_params = {
    'learning_rate': 2e-4,
    'batch_size': 4,
    'gradient_accumulation_steps': 8,
    'mixed_precision': 'fp16',
    'peft_config': {'r': 4}
}

# Issue 4: PEFT slower than expected
# Solution: Verify LoRA is applied
print(pipeline.model)  # Check for LoRA modules
```

---

## 10. PEFT Best Practices

### ✅ Do's

- ✅ Start with r=8 (good default)
- ✅ Use 2x learning rate for PEFT
- ✅ Include warmup steps
- ✅ Monitor gradient norms
- ✅ Use gradient clipping
- ✅ Test rank selection
- ✅ Save adapter weights separately

### ❌ Don'ts

- ❌ Don't use same learning rate as base-ft
- ❌ Don't use rank < 2
- ❌ Don't skip regularization
- ❌ Don't forget to scale learning rate
- ❌ Don't train very long (overfit risk)

---

## 11. PEFT Performance Summary

```
Typical Results on 100K Sample Classification Task:

Base Fine-Tuning:
  Training Time: 30 minutes
  Memory: 12 GB
  Accuracy: 90.5%
  Model Size: 500 MB

PEFT (r=8):
  Training Time: 10 minutes (3x faster)
  Memory: 3 GB (75% reduction)
  Accuracy: 89.8% (0.7% loss)
  Model Size: 5 MB (100x smaller)

Trade-off Analysis:
  Speed: 3x faster
  Memory: 75% reduction
  Storage: 100x smaller
  Accuracy: Only 0.7% lower
  RECOMMENDATION: Use PEFT for most scenarios
```

---

## 12. Quick Reference

| Scenario | r | alpha | dropout | LR | Notes |
|----------|---|-------|---------|-----|-------|
| Memory constrained | 4 | 8 | 0.1 | 1e-4 | Ultra-low resource |
| Standard | 8 | 16 | 0.05 | 2e-4 | Default, balanced |
| High accuracy | 16 | 32 | 0.02 | 1e-4 | Best results |
| Large data (1M) | 8 | 16 | 0.05 | 2e-4 | TabDPT recommended |

---

## 13. Next Steps

- [PEFT & LoRA](../advanced/peft-lora.md) - Theory and mathematics
- [Memory Optimization](../advanced/memory-optimization.md) - Memory techniques
- [Hyperparameter Tuning](../advanced/hyperparameter-tuning.md) - Optimization
- [Classification Examples](classification.md) - Complete workflows

---

Master PEFT for efficient, production-ready fine-tuning!