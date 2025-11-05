# Installation

This guide will walk you through installing TabTune and its dependencies for optimal performance across different environments.

---

## System Requirements


### **Python Version**
- **Python 3.10+** (required)
- Python 3.12+ (recommended for best performance)

### **Hardware**
- **Minimum**: 8GB RAM, 2GB free disk space
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **For Large Datasets**: 32GB+ RAM, multiple GPUs

---

## Installation Methods

### **Method 1: Install from Source (Recommended)**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lexsi-Labs/TabTune.git
   pip install -r requirements.txt
   cd TabTune
   pip install -e .
   ```

2. **Create virtual environment**
   ```bash
   # Using venv
   python -m venv tabtune-env
   source tabtune-env/bin/activate  # Linux/macOS
   # tabtune-env\Scripts\activate   # Windows
   
   # Or using conda
   conda create -n tabtune python=3.11
   conda activate tabtune
   ```

---


!!! tip "GPU Support"
    For optimal performance with large models, install CUDA-enabled PyTorch. Check your CUDA version with `nvidia-smi`.

---

## Core Dependencies

The following packages are automatically installed with TabTune:

### **Essential Packages**
```bash
# Core ML libraries
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Data handling
openml>=0.12.0
datasets>=2.0.0

# PEFT support
peft>=0.4.0
accelerate>=0.20.0
transformers>=4.30.0

# Utilities
joblib>=1.0.0
tqdm>=4.60.0
```

### **Model-Specific Dependencies**
```bash
# For ContextTab (requires HuggingFace Hub access)
huggingface-hub>=0.15.0
sentence-transformers>=2.2.0

# For advanced preprocessing
category-encoders>=2.5.0
```


---

## Verify Installation

### **Quick Verification**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test TabTune import
from tabtune import TabularPipeline
print("✅ TabTune successfully installed!")
```

### **GPU Verification**
```python
import torch

if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ℹ️ CUDA not available, using CPU")
```

### **Model Loading Test**
```python
from tabtune import TabularPipeline
import pandas as pd

# Quick smoke test
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
y = pd.Series([0, 1, 0])

pipeline = TabularPipeline(
    model_name="TabPFN",
    tuning_strategy="inference"
)

print("✅ Pipeline creation successful!")
```

---

## Troubleshooting

### **Common Installation Issues**

#### **Issue: ModuleNotFoundError for torch**
```bash
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/check_version_in_requirements
```

#### **Issue: CUDA out of memory during model loading**
```python
# Solution: Use smaller batch sizes or CPU fallback
pipeline = TabularPipeline(
    model_name="TabPFN",
    tuning_strategy="inference",
    tuning_params={"device": "cpu", "batch_size": 16}
)
```

#### **Issue: ContextTab model access denied**
```bash
# Solution: Set up HuggingFace token
export HF_TOKEN="your_huggingface_token"
# Or login interactively
huggingface-cli login
```

#### **Issue: Permission denied on Windows**
```bash
# Solution: Run as administrator or use --user flag
pip install --user -e .
```

### **Memory Issues**

#### **Large Dataset Handling**
```python
# Use chunked processing for large datasets
tuning_params = {
    "batch_size": 8,  # Reduce batch size
    "gradient_accumulation_steps": 4,  # Maintain effective batch size
    "device": "cuda"
}
```

#### **PEFT Memory Optimization**
```python
# Use PEFT for memory-efficient fine-tuning
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        "peft_config": {"r": 4}  # Lower rank for less memory
    }
)
```

---

## Environment Variables

Set these environment variables for optimal performance:

```bash
# HuggingFace token for gated models
export HF_TOKEN="your_token_here"

# Disable tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false

# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For debugging
export CUDA_LAUNCH_BLOCKING=1
```

---


## Next Steps

After successful installation:

1. **[Quick Start Guide](../getting-started/quick-start.md)** - Run your first tabtune example
2. **[Basic Concepts](../getting-started/basic-concepts.md)** - Understand the core architecture
3. **[Model Selection](../user-guide/model-selection.md)** - Choose the right model for your task

---

!!! success "Installation Complete"
    You're now ready to start using TabTune! If you encounter any issues, please check our [FAQ](../about/faq.md) or open an issue on [GitHub](https://github.com/Lexsi-Labs/TabTune/issues).