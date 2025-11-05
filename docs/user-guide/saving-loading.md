# Saving and Loading Pipelines

This guide explains how to persist TabTune pipelines for production deployment, reproducibility, and continued training.

---

## 1. Overview

TabTune supports multiple serialization formats for different use cases:

| Format | Extension | Use Case | Size | Compatibility |
|--------|-----------|----------|------|---------------|
| **joblib** | `.joblib` | Production, complete pipeline | ~100-500MB | Python only |
| **PyTorch** | `.pt` | Model weights only | ~50-200MB | PyTorch ecosystems |
| **Checkpoint** | `.pt` | Training resume, intermediate states | ~50-200MB | Development |

---

## 2. Pipeline Serialization (joblib)

### 2.1 Saving a Pipeline

Save the complete fitted pipeline including model, preprocessor, and configuration:

```python
from tabtune import TabularPipeline

# Train pipeline
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={'device': 'cuda', 'epochs': 5}
)
pipeline.fit(X_train, y_train)

# Save entire pipeline
pipeline.save('my_pipeline.joblib')
```

**What Gets Saved**:
- ✅ Trained model weights
- ✅ DataProcessor state (imputer, encoder, scaler)
- ✅ Model configuration
- ✅ Tuning parameters
- ✅ Label encoder for target variable
- ✅ PEFT adapter weights (if applicable)


### 2.2 Loading a Pipeline

Load a saved pipeline for inference or continued training:

```python
from tabtune import TabularPipeline

# Load pipeline
loaded_pipeline = TabularPipeline.load('my_pipeline.joblib')

# Make predictions immediately
predictions = loaded_pipeline.predict(X_test)

# Or continue training with new data
loaded_pipeline.fit(X_new_train, y_new_train)
```

### 2.3 Size Optimization

If pipeline file is too large, compress it:

```python
import joblib

# Save with compression
joblib.dump(pipeline, 'my_pipeline.joblib', compress=3)  # compression=0-9

# Load compressed pipeline
pipeline = joblib.load('my_pipeline.joblib')
```

---

## 3. Model-Only Serialization (PyTorch)

For minimal storage or deployment, save only model weights:

### 3.1 Saving Model Weights

```python
import torch

# Save model weights only
torch.save(pipeline.model.state_dict(), 'model_weights.pt')
```

### 3.2 Loading Model Weights

```python
import torch
from tabtune import TabularPipeline

# Create new pipeline with same config
new_pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='inference'
)

# Load weights into model
state_dict = torch.load('model_weights.pt')
new_pipeline.model.load_state_dict(state_dict)

# Ready for inference
predictions = new_pipeline.predict(X_test)
```

**Advantages**:
- ✅ Smaller file size (50-200MB)
- ✅ Faster loading
- ✅ Cross-platform compatible

**Disadvantages**:
- ❌ Requires manual DataProcessor setup
- ❌ Must know original configuration
- ❌ No automatic version compatibility

---

## 4. Checkpoint Management

For long training runs, save intermediate checkpoints:

### 4.1 Automatic Checkpointing During Training

```python
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 20,
        'save_checkpoint_path': 'checkpoints/best_model.pt',
        'save_every_n_epochs': 5  # Save every 5 epochs
    }
)

pipeline.fit(X_train, y_train)

# Checkpoints saved as:
# checkpoints/best_model.pt (best validation score)
# checkpoints/checkpoint_epoch_5.pt
# checkpoints/checkpoint_epoch_10.pt
# checkpoints/checkpoint_epoch_15.pt
# checkpoints/checkpoint_epoch_20.pt
```


### 4.2 Best Checkpoint Tracking

```python
import os
from pathlib import Path

checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

# Find best checkpoint
best_checkpoint = None
best_score = 0

for checkpoint_file in checkpoint_dir.glob('*.pt'):
    pipeline = TabularPipeline.load(str(checkpoint_file))
    score = pipeline.evaluate(X_val, y_val)['accuracy']
    
    if score > best_score:
        best_score = score
        best_checkpoint = checkpoint_file

print(f"Best checkpoint: {best_checkpoint} with score {best_score:.4f}")
```

---

## 5. PEFT-Specific Serialization

When using PEFT (LoRA), save and manage adapter weights:

### 5.1 Save with PEFT Adapters

```python
# Train with PEFT
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'peft_config': {'r': 8, 'lora_alpha': 16},
        'save_checkpoint_path': 'peft_model.joblib'
    }
)

pipeline.fit(X_train, y_train)

# Save includes both base model and LoRA adapters
pipeline.save('peft_pipeline.joblib')
```

### 5.2 Load PEFT Pipeline

```python
# Load includes automatic adapter reconstruction
pipeline = TabularPipeline.load('peft_pipeline.joblib')

# Adapters are already injected
predictions = pipeline.predict(X_test)
```

### 5.3 Extract LoRA Adapters Only

```python
import torch

# Save only adapter weights (minimal size)
if hasattr(pipeline.model, 'lora_A') and hasattr(pipeline.model, 'lora_B'):
    adapter_state = {
        'lora_A': pipeline.model.lora_A.state_dict(),
        'lora_B': pipeline.model.lora_B.state_dict()
    }
    torch.save(adapter_state, 'adapters.pt')
```

---

## 6. Configuration Serialization

Save pipeline configuration for reproducibility:

### 6.1 Save Configuration to YAML

```python
import yaml

config = {
    'model_name': pipeline.model_name,
    'task_type': pipeline.task_type,
    'tuning_strategy': pipeline.tuning_strategy,
    'tuning_params': pipeline.tuning_params,
    'processor_params': pipeline.processor_params,
    'model_params': pipeline.model_params
}

with open('pipeline_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

**Example Config File**:
```yaml
model_name: TabICL
task_type: classification
tuning_strategy: peft
tuning_params:
  device: cuda
  epochs: 5
  learning_rate: 2e-4
  peft_config:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
processor_params:
  imputation_strategy: median
  categorical_encoding: onehot
  scaling_strategy: standard
model_params:
  n_estimators: 16
```

### 6.2 Load Pipeline from Configuration

```python
import yaml
from tabtune import TabularPipeline

# Load config
with open('pipeline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Recreate pipeline
pipeline = TabularPipeline(**config)
```

---


## 7. Best Practices

### ✅ Do's

- ✅ Save full pipelines for production deployment
- ✅ Use checkpoints for long training runs
- ✅ Store configuration separately for reproducibility
- ✅ Version control YAML/JSON configs
- ✅ Keep multiple backups at different timestamps
- ✅ Document model metadata and creation date
- ✅ Test loading on different systems

### ❌ Don'ts

- ❌ Don't save only model weights without config
- ❌ Don't overwrite checkpoints without backup
- ❌ Don't forget to compress large pipelines
- ❌ Don't hardcode paths in production code
- ❌ Don't commit large `.joblib` files to git (use git-lfs)

---

## 8. Troubleshooting

### Issue: "ModuleNotFoundError when loading pipeline"
**Solution**: Ensure TabTune is installed in the target environment

```bash
pip install tabtune
```

### Issue: "Pickle protocol version mismatch"
**Solution**: Use compatible Python and PyTorch versions

```python
# Save with older protocol for compatibility
import joblib
joblib.dump(pipeline, 'compatible.joblib', protocol=2)
```

### Issue: "CUDA error when loading on CPU"
**Solution**: Specify device when loading

```python
pipeline = TabularPipeline.load('pipeline.joblib')
pipeline.model = pipeline.model.to('cpu')
```

### Issue: "Pipeline file corrupted or incomplete"
**Solution**: Restore from backup

```python
restored = TabularPipeline.load('backups/pipeline_backup_20250101_120000.joblib')
```

---

## 9. Summary Table

| Task | Method | File Size | Compatibility |
|------|--------|-----------|---|
| Full pipeline backup | `.save()` | ~300MB | joblib only |
| Model weights only | `torch.save(state_dict)` | ~100MB | PyTorch |
| Configuration only | YAML/JSON | <1MB | Any language |
| Training checkpoint | `.pt` | ~150MB | Development |
| Production package | `.joblib` + config | ~300MB+ | Python |

---

## 10. Next Steps

- [Advanced Topics](../advanced/memory-optimization.md) - Optimize for deployment
- [API Reference](../api/pipeline.md) - Complete API documentation
- [Examples](../examples/classification.md) - Real-world examples

---

Properly saving and loading pipelines ensures reproducibility, enables production deployment, and protects your trained models!