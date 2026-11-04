# TabPFN v3

TabPFN v3 (`TabPFNv3`) re-architects the PFN: a column distribution embedding, row aggregation, then in-context learning over compressed rows. The limits move from 10 classes, 500 features and 10,000 rows to 160 classes, 20,000 features and a million rows under a 200M-cell budget. Inference and every fine-tuning mode, both tasks.

## Introduction

**TabPFN v3**

Like earlier versions it approximates Bayesian posterior inference in a single forward pass. v3 introduces a new backbone (`architectures/tabpfn_v3.py`, registered as `"tabpfn_v3"`) and new default checkpoints.

It is well suited for:

- Strong zero-shot baselines on small-to-medium datasets
- Uncertainty-aware classification and regression
- Task adaptation through native, meta-learning, SFT, and PEFT fine-tuning
- Drop-in replacement for TabPFN / TabPFN v2.6 in existing TabTune workflows

**Relationship to TabPFN v2.6**: v3 is a separate model entry in TabTune. The existing `TabPFNv26` integration is left untouched; `TabPFNv3` is added alongside it with its own vendored model tree, so you can compare the two directly.

!!! note "Model name"
 Use `model_name="TabPFNv3"` in `TabularPipeline`. The v3 default classifier and regressor checkpoints are loaded automatically, no manual version flag is required.

## Architecture

### High-level design

```mermaid
flowchart LR
    A[Input Features] --> B[Feature Encoding]
    B --> C[Support Set Processing]
    C --> D[ICL Transformer Blocks]
    D --> E[Bayesian / Bar-Distribution Head]
    E --> F[Predictions + Uncertainty]
```

### Core components

1. **Feature Encoder** (`x_embed`): projects tabular features into the embedding space
2. **Target Encoders** (`col_y_encoder` / `icl_y_encoder`): encode context labels for in-context conditioning
3. **ICL Transformer Blocks** (`icl_blocks`): stacked attention layers with `q/k/v/out_projection` over support + query samples
4. Heads: classification logits, or a bar-distribution head for regression (mean + full predictive distribution)

### Inference process

```
1. Encode support set (training data) as context
2. Encode query point (test sample)
3. Process through the ICL transformer blocks
4. Output posterior (class probabilities, or bar-distribution for regression)
5. Generate predictions with uncertainty
```

## Inference parameters

### Complete parameter reference

```python
model_params = {
    'n_estimators': 8,                     # Ensemble size
    'softmax_temperature': 0.9,            # Prediction confidence (classification)
    'average_logits': True,                # Aggregation method
    'ignore_pretraining_limits': True,     # Allow larger inputs than pretraining limits
    'device': 'cuda',                      # 'cuda' or 'cpu'
    'random_state': 42                     # Reproducibility
}
```

### Parameter descriptions

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_estimators` | int | 8 | 1-32 | Number of ensemble members; higher = more reliable |
| `softmax_temperature` | float | 0.9 | 0.1-2.0 | Scaling of logits before softmax; lower = sharper predictions |
| `average_logits` | bool | True | True/False | Average logits vs probabilities across ensemble |
| `ignore_pretraining_limits` | bool | True | True/False | Permit inputs above pretraining size limits |
| `device` | str | 'cuda' | cuda/cpu | Compute device |
| `random_state` | int | 42 | 0+ | Random seed for reproducibility |

### Parameter tuning guidelines

**Ensemble Size (`n_estimators`)**:
- `4-8`: Fast inference, good uncertainty
- `8-16`: More reliable predictions, slower

**Temperature (`softmax_temperature`)**:
- `< 0.5`: Very confident predictions (may overfit)
- `0.5 - 1.0`: Default, balanced confidence
- `> 1.0`: Softer predictions, lower confidence

**Average Method (`average_logits`)**:
- `True`: Better for class imbalance
- `False`: Better for probability calibration

## Fine-tuning with TabPFN v3

Unlike the experimental status of fine-tuning on the original TabPFN, **TabPFN v3 supports the full TabTune fine-tuning surface**: native (PriorLabs finetuner), meta-learning, SFT, and PEFT/LoRA for classification, and native + turn-by-turn for regression.

| Task | Strategy | Modes |
|------|----------|-------|
| Classification | `finetune` | `meta-learning` (default), `sft`, `native` |
| Classification | `peft` | LoRA via meta-learning / SFT loops |
| Regression | `finetune` | `native` (default), `turn_by_turn` |

!!! warning "Native fine-tuning trains the v3 checkpoint"
 The upstream PriorLabs `FinetunedTabPFN*` classes default to an older checkpoint internally. TabTune's v3 integration pins `ModelVersion.V3` in native mode so fine-tuning updates the **v3** weights, not an earlier version.

### Fine-tuning parameters

```python
tuning_params = {
    'device': 'cuda',
    'epochs': 30,                 # native; meta/sft default to fewer
    'learning_rate': 1e-5,
    'batch_size': 256,            # meta-learning / sft episode size
    'weight_decay': 0.01,
    'early_stopping': True,       # native mode
    'early_stopping_patience': 8, # native mode
    'finetune_mode': 'meta-learning',  # 'native' | 'sft' | 'meta-learning'
    'show_progress': True
}
```

### Fine-tuning best practices

- Learning Rate: Start with `1e-5`; raise cautiously
- Epochs: `3-5` for meta-learning/SFT; up to `30` for native with early stopping
- Batch Size: `64-256` episodes work well for meta-learning
- Mode choice: `native` for the strongest single-task adaptation; `meta-learning` to preserve in-context generalization; `sft` for fast task specialization

### Native fine-tuning (classification)

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'finetune_mode': 'native',     # uses PriorLabs FinetunedTabPFNClassifier (V3-pinned)
        'device': 'cuda',
        'epochs': 30,
        'learning_rate': 1e-5,
        'early_stopping': True,
        'early_stopping_patience': 8,
    }
)
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Meta-learning fine-tuning (default)

```python
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'finetune_mode': 'meta-learning',  # default
        'device': 'cuda',
        'epochs': 5,
        'batch_size': 256,
        'learning_rate': 1e-5,
    }
)
pipeline.fit(X_train, y_train)
```

### SFT (supervised fine-tuning)

```python
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'finetune_mode': 'sft',
        'device': 'cuda',
        'epochs': 25,
        'learning_rate': 1e-5,
    }
)
pipeline.fit(X_train, y_train)
```

### Regression fine-tuning

```python
# Native (default) - PriorLabs FinetunedTabPFNRegressor, bar-distribution loss
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='regression',
    tuning_strategy='finetune',
    tuning_params={
        'finetune_mode': 'native',
        'device': 'cuda',
        'epochs': 30,
        'early_stopping': True,
    }
)
pipeline.fit(X_train, y_train)

# Turn-by-turn - lightweight episodic regression loop
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='regression',
    tuning_strategy='finetune',
    tuning_params={
        'finetune_mode': 'turn_by_turn',
        'device': 'cuda',
        'epochs': 5,
        'batch_size': 256,
    }
)
pipeline.fit(X_train, y_train)
```

## Inference-only usage

### Zero-shot predictions

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='inference',
    model_params={'n_estimators': 8, 'device': 'cuda'}
)

pipeline.fit(X_train, y_train)        # preprocessing + context setup only
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)
```

### Regression inference

```python
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='regression',
    tuning_strategy='inference',
    model_params={'device': 'cuda'}
)
pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)
```

## PEFT (LoRA) support

### Status

**Supported.** TabPFN v3 ships a dedicated LoRA target configuration (`MODEL_LORA_TARGETS["TabPFNv3"]`) covering the attention projections (`q/k/v/out_projection`), the input embedding (`x_embed`), and the ICL blocks. The dynamic-dimension target encoders (`col_y_encoder` / `icl_y_encoder`) are intentionally excluded from adapter wrapping for stability.

### Usage

```python
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='peft',
    tuning_params={
        'finetune_mode': 'meta-learning',   # PEFT routes through meta/sft loops
        'device': 'cuda',
        'epochs': 10,
        'learning_rate': 5e-5,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        }
    }
)
pipeline.fit(X_train, y_train)
```

!!! note "Native + PEFT"
 PEFT applies to the meta-learning and SFT loops. Native mode trains full weights via the PriorLabs finetuner; if a `peft_config` is supplied in native mode it is ignored with a warning.

## Usage scenarios

### Quick baseline

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='inference'
)
pipeline.fit(X_train, y_train)
baseline = pipeline.evaluate(X_test, y_test)
print(f"Baseline accuracy: {baseline['accuracy']:.4f}")
```

### Small/Medium Dataset adaptation

```python
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={'finetune_mode': 'native', 'epochs': 30, 'early_stopping': True}
)
pipeline.fit(X_train, y_train)
```

## Limitations and constraints

### Data constraints

| Constraint | Guidance | Impact |
|-----------|----------|--------|
| Rows | Best on small-to-medium data | Very large datasets increase memory/latency |
| Features | Moderate feature counts | More features → longer processing |
| Min Features | 2 | Single-feature prediction not supported |
| Classes | Multi-class supported | Very high class counts may need more estimators |

Use `ignore_pretraining_limits=True` (default in TabTune) to permit inputs above the pretraining limits at the cost of speed/memory.

### Feature type constraints

- Supported: Numerical, categorical, mixed (handled by the `tabpfn_special` preprocessor)
- Not Supported: Raw text, images, native time-series

### Task type constraints

- Binary Classification
- Multi-class Classification
- Regression
- Multi-output / Multi-label

## Troubleshooting

### "Out of memory during fine-tuning"
**Solution**: Lower `batch_size`, reduce `n_estimators`, or use PEFT.

```python
tuning_params = {'batch_size': 64, 'finetune_mode': 'meta-learning'}
```

### "Native fine-tuning seems to use an old checkpoint"
**Solution**: TabTune pins `ModelVersion.V3` automatically. Make sure you are on the `TabPFNv3` model entry (not `TabPFNv26`) and that the v3 default checkpoint downloaded from `Prior-Labs/tabpfn_3`.

### "PEFT has no effect in native mode"
**Solution**: Native mode trains full weights. Use `finetune_mode='meta-learning'` (or `'sft'`) together with `tuning_strategy='peft'`.

### "Predictions too confident (low uncertainty)"
**Solution**: Increase temperature.

```python
model_params = {'softmax_temperature': 1.5}
```

## Complete example workflow

```python
from tabtune import TabularPipeline, TabularLeaderboard
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load data
df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Zero-shot baseline
print("=== Zero-Shot Baseline ===")
baseline = TabularPipeline(model_name='TabPFNv3', tuning_strategy='inference')
baseline.fit(X_train, y_train)
print(f"Baseline Accuracy: {baseline.evaluate(X_test, y_test)['accuracy']:.4f}")

# 3. Native fine-tuning
print("\n=== Native Fine-Tuned ===")
finetuned = TabularPipeline(
    model_name='TabPFNv3',
    tuning_strategy='finetune',
    tuning_params={'finetune_mode': 'native', 'epochs': 30,
                   'early_stopping': True, 'device': 'cuda'}
)
finetuned.fit(X_train, y_train)
print(f"Fine-tuned Accuracy: {finetuned.evaluate(X_test, y_test)['accuracy']:.4f}")

# 4. Compare v3 vs v2.6 vs other models
print("\n=== Model Comparison ===")
lb = TabularLeaderboard(X_train, X_test, y_train, y_test)
lb.add_model('TabPFNv3', 'inference', name='TabPFNv3-Inference')
lb.add_model('TabPFNv3', 'finetune', name='TabPFNv3-Native',
             tuning_params={'finetune_mode': 'native', 'epochs': 5})
lb.add_model('TabPFNv26', 'inference', name='TabPFNv26-Inference')
lb.run(rank_by='accuracy')
```

## Quick reference

| Task | Strategy | Mode | Notes |
|------|----------|------|-------|
| Instant baseline | inference |, | Zero-shot, with uncertainty |
| Strongest single-task | finetune | native | V3-pinned PriorLabs finetuner + early stopping |
| Preserve generalization | finetune | meta-learning | Episodic (default) |
| Fast specialization | finetune | sft | Single-episode supervised |
| Memory-efficient | peft | meta-learning/sft | LoRA adapters |
| Regression | finetune | native / turn_by_turn | Bar-distribution loss |
