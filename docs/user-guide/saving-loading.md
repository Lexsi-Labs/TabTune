# Saving and Loading

## Overview

`pipe.save(path)` writes the fitted preprocessor, the weights (or the LoRA adapters) and the configuration; `TabularPipeline.load(path)` brings it back without refitting. Three formats:

| Format | Extension | Use Case | Size | Compatibility |
|--------|-----------|----------|------|---------------|
| **joblib** | `.joblib` | Production, complete pipeline | ~100-500MB | Python only |
| **PyTorch** | `.pt` | Model weights only | ~50-200MB | PyTorch ecosystems |
| **Checkpoint** | `.pt` | Training resume, intermediate states | ~50-200MB | Development |

## Pipeline serialization (joblib)

### Saving a pipeline

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
- Trained model weights
- DataProcessor state (imputer, encoder, scaler)
- Model configuration
- Tuning parameters
- Label encoder for target variable
- PEFT adapter weights (if applicable)

### Loading a pipeline

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

### Size optimization

If pipeline file is too large, compress it:

```python
import joblib

# Save with compression
joblib.dump(pipeline, 'my_pipeline.joblib', compress=3)  # compression=0-9

# Load compressed pipeline
pipeline = joblib.load('my_pipeline.joblib')
```

## Model-only serialization (PyTorch)

For minimal storage or deployment, save only model weights:

### Saving model Weights

```python
import torch

# Save model weights only
torch.save(pipeline.model.state_dict(), 'model_weights.pt')
```

### Loading model Weights

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
- Smaller file size (50-200MB)
- Faster loading
- Cross-platform compatible

**Disadvantages**:
- Requires manual DataProcessor setup
- Must know original configuration
- No automatic version compatibility

## Checkpoint management

For long training runs, save intermediate checkpoints:

### Automatic checkpointing during training

```python
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='finetune',
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

### Best checkpoint tracking

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

## PEFT-specific serialization

When using PEFT (LoRA), save and manage adapter weights:

### Save with PEFT adapters

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

### Load PEFT pipeline

```python
# Load includes automatic adapter reconstruction
pipeline = TabularPipeline.load('peft_pipeline.joblib')

# Adapters are already injected
predictions = pipeline.predict(X_test)
```

### Extract LoRA adapters only

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

## Configuration serialization

Save pipeline configuration for reproducibility:

### Save configuration to YAML

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

### Load pipeline from configuration

```python
import yaml
from tabtune import TabularPipeline

# Load config
with open('pipeline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Recreate pipeline
pipeline = TabularPipeline(**config)
```

## Troubleshooting

### "ModuleNotFoundError when loading pipeline"
**Solution**: Make sure TabTune is installed in the target environment

```bash
pip install tabtune
```

### "Pickle protocol version mismatch"
**Solution**: Use compatible Python and PyTorch versions

```python
# Save with older protocol for compatibility
import joblib
joblib.dump(pipeline, 'compatible.joblib', protocol=2)
```

### "CUDA error when loading on CPU"
**Solution**: Specify device when loading

```python
pipeline = TabularPipeline.load('pipeline.joblib')
pipeline.model = pipeline.model.to('cpu')
```

### "Pipeline file corrupted or incomplete"
**Solution**: Restore from backup

```python
restored = TabularPipeline.load('backups/pipeline_backup_20250101_120000.joblib')
```

## Summary table

| Task | Method | File Size | Compatibility |
|------|--------|-----------|---|
| Full pipeline backup | `.save()` | ~300MB | joblib only |
| Model weights only | `torch.save(state_dict)` | ~100MB | PyTorch |
| Configuration only | YAML/JSON | <1MB | Any language |
| Training checkpoint | `.pt` | ~150MB | Development |
| Production package | `.joblib` + config | ~300MB+ | Python |
