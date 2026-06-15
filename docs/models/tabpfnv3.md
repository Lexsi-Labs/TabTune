# TabPFN v3: Latest Prior-Fitted Network

TabPFN v3 is the newest release of PriorLabs' Prior-Fitted Network family, integrated into TabTune as the model name **`TabPFNv3`**. It builds on the in-context learning paradigm of earlier TabPFN versions with an updated architecture and checkpoints, and ships with **full inference and fine-tuning support** for both classification and regression. This document is an in-depth guide to using TabPFN v3 with TabTune.

---

## 1. Introduction

**What is TabPFN v3?**

TabPFN v3 is a transformer trained via in-context learning on large collections of synthetic tabular datasets. Like its predecessors it approximates Bayesian posterior inference in a single forward pass, but v3 introduces an updated backbone (`architectures/tabpfn_v3.py`, registered as `"tabpfn_v3"`) and new default checkpoints (`v3_default`) hosted on the `Prior-Labs/tabpfn_3` repository.

It is well suited for:

- Strong zero-shot baselines on small-to-medium datasets
- Uncertainty-aware classification and regression
- Task adaptation through native, meta-learning, SFT, and PEFT fine-tuning
- Drop-in replacement for TabPFN / TabPFN v2.6 in existing TabTune workflows

**Relationship to TabPFN v2.6**: v3 is a separate model entry in TabTune. The existing `TabPFNv26` integration is left untouched; `TabPFNv3` is added alongside it with its own vendored model tree, so you can compare the two directly.

!!! note "Model name"
    Use `model_name="TabPFNv3"` in `TabularPipeline`. The v3 default classifier and regressor checkpoints are loaded automatically — no manual version flag is required.

---

## 2. Architecture

### 2.1 High-Level Design

```mermaid
flowchart LR
    A[Input Features] --> B[Feature Encoding]
    B --> C[Support Set Processing]
    C --> D[ICL Transformer Blocks]
    D --> E[Bayesian / Bar-Distribution Head]
    E --> F[Predictions + Uncertainty]
```

### 2.2 Core Components

1. **Feature Encoder** (`x_embed`): projects tabular features into the embedding space
2. **Target Encoders** (`col_y_encoder` / `icl_y_encoder`): encode context labels for in-context conditioning
3. **ICL Transformer Blocks** (`icl_blocks`): stacked attention layers with `q/k/v/out_projection` over support + query samples
4. **Heads**: classification logits, or a bar-distribution head for regression (mean + full predictive distribution)

### 2.3 Inference Process

```
1. Encode support set (training data) as context
2. Encode query point (test sample)
3. Process through the ICL transformer blocks
4. Output posterior (class probabilities, or bar-distribution for regression)
5. Generate predictions with uncertainty
```

---

## 3. Inference Parameters

### 3.1 Complete Parameter Reference

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

### 3.2 Parameter Descriptions

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_estimators` | int | 8 | 1-32 | Number of ensemble members; higher = more robust |
| `softmax_temperature` | float | 0.9 | 0.1-2.0 | Scaling of logits before softmax; lower = sharper predictions |
| `average_logits` | bool | True | True/False | Average logits vs probabilities across ensemble |
| `ignore_pretraining_limits` | bool | True | True/False | Permit inputs above pretraining size limits |
| `device` | str | 'cuda' | cuda/cpu | Compute device |
| `random_state` | int | 42 | 0+ | Random seed for reproducibility |

### 3.3 Parameter Tuning Guidelines

**Ensemble Size (`n_estimators`)**:
- `4-8`: Fast inference, good uncertainty
- `8-16`: More robust predictions, slower

**Temperature (`softmax_temperature`)**:
- `< 0.5`: Very confident predictions (may overfit)
- `0.5 - 1.0`: Default, balanced confidence
- `> 1.0`: Softer predictions, lower confidence

**Average Method (`average_logits`)**:
- `True`: Better for class imbalance
- `False`: Better for probability calibration

---

## 4. Fine-Tuning with TabPFN v3

Unlike the experimental status of fine-tuning on the original TabPFN, **TabPFN v3 supports the full TabTune fine-tuning surface**: native (PriorLabs finetuner), meta-learning, SFT, and PEFT/LoRA for classification, and native + turn-by-turn for regression.

| Task | Strategy | Modes |
|------|----------|-------|
| Classification | `finetune` | `meta-learning` (default), `sft`, `native` |
| Classification | `peft` | LoRA via meta-learning / SFT loops |
| Regression | `finetune` | `native` (default), `turn_by_turn` |

!!! warning "Native fine-tuning trains the v3 checkpoint"
    The upstream PriorLabs `FinetunedTabPFN*` classes default to an older checkpoint internally. TabTune's v3 integration pins `ModelVersion.V3` in native mode so fine-tuning updates the **v3** weights, not an earlier version.

### 4.1 Fine-Tuning Parameters

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

### 4.2 Fine-Tuning Best Practices

- **Learning Rate**: Start with `1e-5`; raise cautiously
- **Epochs**: `3-5` for meta-learning/SFT; up to `30` for native with early stopping
- **Batch Size**: `64-256` episodes work well for meta-learning
- **Mode choice**: `native` for the strongest single-task adaptation; `meta-learning` to preserve in-context generalization; `sft` for fast task specialization

### 4.3 Native Fine-Tuning (Classification)

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

### 4.4 Meta-Learning Fine-Tuning (default)

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

### 4.5 SFT (Supervised Fine-Tuning)

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

### 4.6 Regression Fine-Tuning

```python
# Native (default) — PriorLabs FinetunedTabPFNRegressor, bar-distribution loss
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

# Turn-by-turn — lightweight episodic regression loop
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

---

## 5. Inference-Only Usage

### 5.1 Zero-Shot Predictions

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

### 5.2 Regression Inference

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

---

## 6. PEFT (LoRA) Support

### 6.1 Status

**✅ Supported.** TabPFN v3 ships a dedicated LoRA target configuration (`MODEL_LORA_TARGETS["TabPFNv3"]`) covering the attention projections (`q/k/v/out_projection`), the input embedding (`x_embed`), and the ICL blocks. The dynamic-dimension target encoders (`col_y_encoder` / `icl_y_encoder`) are intentionally excluded from adapter wrapping for stability.

### 6.2 Usage

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

---

## 7. Usage Scenarios

### 7.1 Quick Baseline

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

### 7.2 Small/Medium Dataset Adaptation

```python
pipeline = TabularPipeline(
    model_name='TabPFNv3',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={'finetune_mode': 'native', 'epochs': 30, 'early_stopping': True}
)
pipeline.fit(X_train, y_train)
```

---

## 8. Limitations and Constraints

### 8.1 Data Constraints

| Constraint | Guidance | Impact |
|-----------|----------|--------|
| Rows | Best on small-to-medium data | Very large datasets increase memory/latency |
| Features | Moderate feature counts | More features → longer processing |
| Min Features | 2 | Single-feature prediction not supported |
| Classes | Multi-class supported | Very high class counts may need more estimators |

Use `ignore_pretraining_limits=True` (default in TabTune) to permit inputs above the pretraining limits at the cost of speed/memory.

### 8.2 Feature Type Constraints

- **Supported**: Numerical, categorical, mixed (handled by the `tabpfn_special` preprocessor)
- **Not Supported**: Raw text, images, native time-series

### 8.3 Task Type Constraints

- ✅ Binary Classification
- ✅ Multi-class Classification
- ✅ Regression
- ❌ Multi-output / Multi-label

---

## 9. Troubleshooting

### Issue: "Out of memory during fine-tuning"
**Solution**: Lower `batch_size`, reduce `n_estimators`, or use PEFT.

```python
tuning_params = {'batch_size': 64, 'finetune_mode': 'meta-learning'}
```

### Issue: "Native fine-tuning seems to use an old checkpoint"
**Solution**: TabTune pins `ModelVersion.V3` automatically. Ensure you are on the `TabPFNv3` model entry (not `TabPFNv26`) and that the v3 default checkpoint downloaded from `Prior-Labs/tabpfn_3`.

### Issue: "PEFT has no effect in native mode"
**Solution**: Native mode trains full weights. Use `finetune_mode='meta-learning'` (or `'sft'`) together with `tuning_strategy='peft'`.

### Issue: "Predictions too confident (low uncertainty)"
**Solution**: Increase temperature.

```python
model_params = {'softmax_temperature': 1.5}
```

---

## 10. Complete Example Workflow

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

---

## 11. Quick Reference

| Task | Strategy | Mode | Notes |
|------|----------|------|-------|
| Instant baseline | inference | — | Zero-shot, with uncertainty |
| Strongest single-task | finetune | native | V3-pinned PriorLabs finetuner + early stopping |
| Preserve generalization | finetune | meta-learning | Episodic (default) |
| Fast specialization | finetune | sft | Single-episode supervised |
| Memory-efficient | peft | meta-learning/sft | LoRA adapters |
| Regression | finetune | native / turn_by_turn | Bar-distribution loss |

---

## 12. Next Steps

- [Model Selection](../user-guide/model-selection.md) - Compare with other models
- [Tuning Strategies](../user-guide/tuning-strategies.md) - Fine-tuning details
- [PEFT & LoRA](../advanced/peft-lora.md) - Parameter-efficient fine-tuning
- [TabPFN](tabpfn.md) - The original Prior-Fitted Network
- [API Reference](../api/pipeline.md) - Complete API docs

---

TabPFN v3 brings the latest Prior-Fitted Network into TabTune with first-class fine-tuning. Use it as a strong baseline and as a fully adaptable model for classification and regression.
