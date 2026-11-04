# ConTextTab

ConTextTab (SAP's `sap-rpt-1-oss`) is the one model here that reads your column names and your text cells. It embeds them with a language model, so a column called `annual_income` means something to it before it sees a single value. Weights are research-use only.

## Introduction

**ContextTab**

What it does differently:

- Use Feature Semantics: Understands column names and semantic meaning
- Text Integration: Handles free-text features naturally
- Modality-Aware Processing: Different encoders for different data modalities
- Semantic Embeddings: Uses pre-trained text embeddings for features
- Heterogeneous Data: Mixed numerical, categorical, and text features

The idea: tabular features plus language-model embeddings of the column names and text cells, so the model knows what a feature represents before it sees its values.

## Architecture

### High-level design

```mermaid
flowchart LR
    A[Raw Data] --> B[Feature Names]
    A --> C[Feature Values]
    B --> D[Text Encoder]
    C --> E[Value Encoder]
    D --> F[Semantic Embeddings]
    E --> G[Value Embeddings]
    F --> H[Fusion Layer]
    G --> H
    H --> I[ICL Module]
    I --> J[Context Processing]
    J --> K[Predictions]
```
<!--
### Core components

1. **Text Encoder** (`in_context_encoder`)
 - Encodes feature names and descriptions
 - Uses sentence transformers/BERT
 - Generates semantic embeddings
 - Pre-trained on diverse text

2. **Value Encoder** (`dense`)
 - Encodes feature values
 - Handles mixed modalities
 - Per-feature type processing
 - Numerical and categorical support

3. **Semantic Fusion** (implicit)
 - Combines text and value embeddings
 - Cross-modality attention
 - Unified representation

4. **ICL Predictor** (`output_head`)
 - In-context learning head
 - Context-aware predictions
 - Support/query set handling -->

### Semantic processing pipeline

```
Feature Names         Feature Values
     ↓                      ↓
  Text Encoder        Value Encoder
     ↓                      ↓
Semantic Vectors     Value Vectors
     ↓                      ↓
  ├─────Semantic Fusion─────┤
           ↓
    Joint Representation
           ↓
    ICL Predictor
           ↓
      Predictions
```

## Inference parameters

### Complete parameter reference

```python
model_params = {
    # Text encoding
    'text_encoder': 'sentence-transformers/all-MiniLM-L6-v2',  # BERT model
    'text_dim': 384,                       # Text embedding dimension
    
    # Value encoding
    'value_dim': 64,                       # Value embedding dimension
    'categorical_encoding': 'embedding',   # 'embedding' or 'onehot'
    
    # Fusion and processing
    'fusion_dim': 128,                     # Fused embedding dimension
    'dropout': 0.1,                        # Dropout rate
    
    # Training behavior
    'use_cache': True,                     # Cache embeddings
    'seed': 42                             # Reproducibility
}
```

### Parameter descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_encoder` | str | 'all-MiniLM-L6-v2' | Hugging Face model ID |
| `text_dim` | int | 384 | Output dimension of text encoder |
| `value_dim` | int | 64 | Dimension for value embeddings |
| `categorical_encoding` | str | 'embedding' | How to encode categoricals |
| `fusion_dim` | int | 128 | Fused representation dimension |
| `dropout` | float | 0.1 | Dropout probability |
| `use_cache` | bool | True | Cache computed embeddings |
| `seed` | int | 42 | Random seed |

### Text encoder options

```python
# Different pre-trained models (trade-off: speed vs quality)
text_encoders = {
    'all-MiniLM-L6-v2': {
        'dim': 384,
        'speed': 'Fast',
        'quality': 'Good'
    },
    'all-MiniLM-L12-v2': {
        'dim': 384,
        'speed': 'Medium',
        'quality': 'Better'
    },
    'all-mpnet-base-v2': {
        'dim': 768,
        'speed': 'Slow',
        'quality': 'Best'
    }
}
```

## Fine-tuning with ContextTab

ContextTab supports **base fine-tuning** via episodic training.

### Fine-tuning parameters

```python
tuning_params = {
    'device': 'cuda',
    'epochs': 10,                         # More epochs typically needed
    'learning_rate': 1e-4,                # Higher than other models
    'optimizer': 'adamw',                 # Optimizer type
    
    'batch_size': 8,                      # Standard batch size
    'show_progress': True                 # Progress bar
}
```

### Fine-tuning best practices

- Epochs: 5-15 (longer than most models)
- Learning Rate: 1e-4 to 5e-4 (higher than TabICL)
- Warmup: Include warmup for stability
- Scheduler: Use cosine decay for better convergence
- Early Stopping: Important due to text embedding complexity

### Fine-tuning stability

ContextTab training can be unstable due to:
- Complex text-value fusion
- High-dimensional embeddings
- Cross-modality interactions

**Recommendations**:
```python
tuning_params = {
    'device': 'cuda',
    'epochs': 10,
    'learning_rate': 1e-4,
    'show_progress': True
}
```

## Setup requirements

### HuggingFace hub access

ContextTab requires access to gated models on Hugging Face:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login with your token
huggingface-cli login

# Or set environment variable
export HF_TOKEN='your_token_here'
```

### Verify setup

```python
from huggingface_hub import login
import os

# Check token
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token)
    print("Logged into Hugging Face Hub")
else:
    print("HF_TOKEN not set - may fail for gated models")
```

### Model download

First-time usage downloads model (~2GB):
```python
# First use will download model
pipeline = TabularPipeline(model_name='ContextTab', tuning_strategy='inference')
# ... downloads and caches model
```

## Usage patterns

### Inference only

```python
from tabtune import TabularPipeline
import os

# Set HF token
os.environ['HF_TOKEN'] = 'your_token'

pipeline = TabularPipeline(
    model_name='ContextTab',
    tuning_strategy='inference',
    model_params={
        'text_encoder': 'sentence-transformers/all-MiniLM-L6-v2',
        'fusion_dim': 128
    }
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Base fine-tuning

```python
pipeline = TabularPipeline(
    model_name='ContextTab',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 10,
        'learning_rate': 1e-4,
        'warmup_steps': 200,
        'batch_size': 8,
        'show_progress': True
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### PEFT fine-tuning (experimental)

```python
# PEFT support is experimental for ContextTab
pipeline = TabularPipeline(
    model_name='ContextTab',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 10,
        'learning_rate': 2e-4,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        }
    }
)

# May have issues - finetune recommended
pipeline.fit(X_train, y_train)
```

## LoRA target modules (experimental)

When using PEFT, ContextTab targets:

```python
target_modules = [
    'in_context_encoder',      # Text encoder
    'dense',                   # Value encoder
    'output_head',             # Prediction head
    'embeddings'               # Embedding layers
]
```

### PEFT status

**Experimental**: PEFT support for ContextTab is experimental because:
- Complex embedding pipeline
- Cross-modality fusion issues
- Potential prediction inconsistencies

**Recommendation**: Use `finetune` strategy instead of `peft`

## Complete examples

### Text-heavy Dataset

```python
from tabtune import TabularPipeline
import os

# Example: Customer survey data with text responses
os.environ['HF_TOKEN'] = 'your_token'

# X contains columns like:
# - age (numerical)
# - category (categorical)
# - feedback_text (text)
# - rating (numerical)

pipeline = TabularPipeline(
    model_name='ContextTab',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda',
        'epochs': 10,
        'learning_rate': 1e-4,
        'warmup_steps': 200,
        'show_progress': True
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Semantic feature names

ContextTab uses meaningful feature names:

```python
# Good: Descriptive feature names (ContextTab works well)
X = pd.DataFrame({
    'customer_age': [25, 30, 45],
    'total_purchases_amount': [100, 250, 5000],
    'years_as_customer': [1, 5, 10],
    'product_category_preference': ['electronics', 'books', 'home']
})

# Less Good: Generic feature names (ContextTab has less semantic info)
X = pd.DataFrame({
    'f1': [25, 30, 45],
    'f2': [100, 250, 5000],
    'f3': [1, 5, 10],
    'f4': ['electronics', 'books', 'home']
})
```

### Production deployment - saving using joblib

```python
import joblib
import os

os.environ['HF_TOKEN'] = 'your_token'

# Train
pipeline = TabularPipeline(
    model_name='ContextTab',
    tuning_strategy='finetune',
    tuning_params={
        'epochs': 10,
        'learning_rate': 1e-4
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

# Save
pipeline.save('contexttab_production.joblib')

# In production (ensure HF_TOKEN is set)
loaded = TabularPipeline.load('contexttab_production.joblib')
predictions = loaded.predict(X_new)
```

## Performance characteristics

### Speed benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Inference (batch=1000) | 2-4s | Text encoding overhead |
| Fine-tuning (10 epochs, 100K) | 30-45m | Longer training |
| Prediction latency | 20-100ms | Per sample |
| Text embedding cache | 1-2s | One-time at startup |

### Memory usage

| Scenario | Memory | GPU VRAM |
|----------|--------|---------|
| Inference | 6-8 GB | 4GB minimum |
| Fine-tuning | 10-14 GB | 8GB recommended |
| Large text dim | Up to 16 GB | 10GB+ needed |
| With caching | Add 1-2 GB | For embeddings |

<!-- ### 9.3 Accuracy Profile

| Data Type | Accuracy | Notes |
|-----------|----------|-------|
| Pure numerical | 85% | Works but not optimal |
| Pure categorical | 87% | Good performance |
| Mixed + text | 90%+ | Excels here |
| Semantic names | +2-3% | Better with meaningful names |
 -->

<!-- ## 10. Special Handling

### Text columns

ContextTab automatically detects and processes text columns:

```python
X = pd.DataFrame({
    'age': [25, 30, 45],
    'product_name': ['Widget A', 'Widget B', 'Widget C'],  # Text
    'description': ['Best seller', 'New product', 'Premium'],  # Text
    'category': ['electronics', 'home', 'sports']
})

# ContextTab will:
# 1. Encode 'product_name' as text
# 2. Encode 'description' as text
# 3. Handle 'category' as categorical
```

### Missing values

ContextTab handles missing values naturally:

```python
# Text columns with NaN
X = pd.DataFrame({
    'feedback': ['Good product', None, 'Excellent'],  # NaN handled
    'category': ['A', 'B', 'C']
})

pipeline.fit(X_train, y_train)  # Works seamlessly
```
 -->

## Troubleshooting

### "HuggingFace login required"
**Solution**:
```bash
export HF_TOKEN='hf_xxxxxxxxxxxx'
# or
huggingface-cli login
```

### "Model download fails"
**Solution**: Check internet and token
```python
from huggingface_hub import model_info
try:
    info = model_info('sentence-transformers/all-MiniLM-L6-v2')
    print("Model accessible")
except Exception as e:
    print(f"Model access failed: {e}")
```

### "Slow inference due to text encoding"
**Solution**: Use faster text encoder
```python
model_params = {
    'text_encoder': 'sentence-transformers/all-MiniLM-L6-v2',  # Fastest
    # instead of 'all-mpnet-base-v2'  # Slowest
}
```

### "Training unstable or diverging"
**Solution**: Increase regularization
```python
tuning_params = {
    'learning_rate': 5e-5,  # Reduce
    'warmup_steps': 500,    # Increase
    'weight_decay': 0.1,    # Increase
    'gradient_clip_value': 0.5  # Tighter
}
```

### "Out of memory during training"
**Solution**: Reduce batch size
```python
tuning_params = {
    'batch_size': 4  # Instead of 8
}
```

## When to use ContextTab

**Use ContextTab when**:
- Dataset has text columns/features
- Feature names are semantic/meaningful
- Mixed data types (numerical + categorical + text)
- You have HuggingFace Hub access
- Accuracy is priority over speed

**Don't use ContextTab for**:
- Pure numerical data (use TabDPT)
- Generic feature names (limited benefit)
- Memory-constrained environments
- When you need fast training
- Without HuggingFace access

## Comparison with other models

| Aspect | ContextTab | TabICL | TabDPT | Mitra |
|--------|-----------|--------|--------|-------|
| Text Support | Excellent | No | No | No |
| Semantic Names | Uses | Ignores | Ignores | Ignores |
| Speed | Medium | Fast | Slow | Slow |
| Memory | Moderate | Moderate | High | Very High |
| Mixed Data | Excellent | Good | Good | Good |
| Accuracy | High | Good | Excellent | Excellent |
| PEFT | Exp | Full | Full | Full |

<!-- ## 15. Advanced Topics

### Custom text encoders

```python
from sentence_transformers import SentenceTransformer

# Use custom text encoder
custom_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Integration via model_params
model_params = {
    'text_encoder': 'paraphrase-MiniLM-L6-v2'
}
```

### Feature description enrichment

```python
# If feature names are generic, add descriptions
X_train = pd.DataFrame({
    'f1': [25, 30, 45],  # Customer age
    'f2': [100, 250, 5000],  # Total purchase amount
    'f3': ['A', 'B', 'C']  # Product category
})

# ContextTab will use column names, so rename:
X_train.columns = ['customer_age', 'purchase_amount', 'category']
``` -->

## Quick reference

| Task | Strategy | Config | Epochs |
|------|----------|--------|--------|
| Quick baseline | inference | default | 0 |
| Mixed data | finetune | learning_rate=1e-4 | 10 |
| Text-heavy | finetune | warmup=200 | 10 |
| Memory limited | finetune | batch_size=4 | 10 |
| Max accuracy | finetune | full tune | 15 |
