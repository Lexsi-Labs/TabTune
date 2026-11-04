# Memory Optimization: Techniques for Training Large Models

This document provides comprehensive strategies for optimizing memory usage when training TabTune models, enabling efficient use of limited GPU/CPU resources.

---

## 1. Introduction

Memory optimization is critical for:

- Training on GPUs with limited VRAM (< 8GB)
- Processing large datasets (100K+ rows)
- Using large model architectures
- Running multiple experiments simultaneously
- Deploying in resource-constrained environments

This guide covers techniques, tools, and trade-offs for memory efficiency.

---

## 2. Memory Profiling

### 2.1 Understanding Memory Usage

**Memory Breakdown** (typical forward/backward pass):

```
Model weights:        40-50% (frozen in PEFT)
Optimizer states:     20-30% (momentum, variance)
Gradients:           10-15%
Activations:         10-20% (intermediate values)
DataLoader buffers:   5-10%
─────────────────────────────
Total:               100%
```

### 2.2 Profiling Tools

#### PyTorch Memory Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Basic memory measurement
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Your training code
pipeline.fit(X_train, y_train)

# Print memory stats
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Detailed profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    pipeline.fit(X_train, y_train)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

#### Memory Monitoring Script

```python
import psutil
import nvidia_smi

class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.peak_gpu = 0
        self.peak_cpu = 0
    
    def record(self):
        """Record current memory usage."""
        try:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem = info.used / 1e9
            
            self.peak_gpu = max(self.peak_gpu, gpu_mem)
        except:
            gpu_mem = 0
        
        # CPU memory
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / 1e9
        self.peak_cpu = max(self.peak_cpu, cpu_mem)
        
        return gpu_mem, cpu_mem
    
    def report(self):
        """Print memory report."""
        print(f"Peak GPU: {self.peak_gpu:.2f} GB")
        print(f"Peak CPU: {self.peak_cpu:.2f} GB")

# Usage
monitor = MemoryMonitor()

# Periodically call during training
for epoch in range(num_epochs):
    gpu_mem, cpu_mem = monitor.record()
    print(f"Epoch {epoch}: GPU {gpu_mem:.2f}GB, CPU {cpu_mem:.2f}GB")

monitor.report()
```

---

## 3. Optimization Techniques

### 3.1 PEFT (LoRA) - Primary Technique

**Impact**: 60-90% memory reduction

```python
# Base fine-tuning: high memory
pipeline_base = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={'device': 'cuda', 'epochs': 5}
)
# Memory: ~12 GB for 100K samples

# PEFT fine-tuning: low memory
pipeline_peft = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'peft_config': {'r': 8}
    }
)
# Memory: ~4 GB for same task
```

**Choosing PEFT config for memory**:

```python
# Ultra-constrained (2GB GPU)
peft_config = {
    'r': 2,
    'lora_alpha': 4,
    'lora_dropout': 0.2
}

# Memory-constrained (4GB GPU)
peft_config = {
    'r': 4,
    'lora_alpha': 8,
    'lora_dropout': 0.1
}

# Moderate constraint (6GB GPU)
peft_config = {
    'r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05
}
```

### 3.2 Batch Size Reduction

**Impact**: 20-40% memory reduction per halving

```python
# Large batch (high memory)
tuning_params = {
    'batch_size': 64,
    'support_size': 512
}
# Memory: ~12 GB

# Reduced batch (medium memory)
tuning_params = {
    'batch_size': 32,
    'support_size': 256
}
# Memory: ~6 GB

# Small batch (low memory)
tuning_params = {
    'batch_size': 8,
    'support_size': 64
}
# Memory: ~2 GB
```

**Trade-offs**:
- Smaller batch: More gradient noise, longer convergence
- Larger batch: Faster convergence, higher memory

### 3.3 Gradient Accumulation

Simulate larger batch without increased memory:

```python
tuning_params = {
    'batch_size': 8,                      # Actual batch
    'gradient_accumulation_steps': 4,     # Accumulate 4x
    # Effective batch = 32
    'device': 'cuda'
}

# Memory cost: Similar to batch_size=8
# Effective batch size benefit: batch_size=32
```

### 3.4 Mixed Precision Training

**Impact**: 20-30% memory reduction

```python
# Standard (float32): high memory
tuning_params = {
    'device': 'cuda',
    'mixed_precision': None  # Full precision
}

# Half precision (float16): lower memory
tuning_params = {
    'device': 'cuda',
    'mixed_precision': 'fp16'  # Use 16-bit floats
}

# BFloat16 (better stability)
tuning_params = {
    'device': 'cuda',
    'mixed_precision': 'bf16'
}
```

**Implementation**:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Setup for mixed precision
scaler = GradScaler()

# In training loop
with autocast(dtype=torch.float16):
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3.5 Gradient Checkpointing

Trade computation for memory:

```python
tuning_params = {
    'device': 'cuda',
    'gradient_checkpoint': True  # Save memory at cost of computation
}

# Memory reduction: ~30-50%
# Computation increase: ~20-30% (recompute activations)
```

### 3.6 Data Loading Optimization

```python
# Efficient data loading
loader_config = {
    'batch_size': 32,
    'num_workers': 4,           # Parallel loading
    'pin_memory': True,         # Faster CPU→GPU transfer
    'persistent_workers': True,  # Keep workers alive
    'prefetch_factor': 2        # Prefetch next batches
}

# Or with TabTune
tuning_params = {
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda'
}
```

### 3.7 Model Architecture Reduction

Reduce model complexity:

```python
# Large model: high memory
model_params = {
    'd_model': 256,      # Embedding dimension
    'num_layers': 8,     # Transformer layers
    'num_heads': 16      # Attention heads
}

# Medium model: medium memory
model_params = {
    'd_model': 128,
    'num_layers': 4,
    'num_heads': 8
}

# Small model: low memory
model_params = {
    'd_model': 64,
    'num_layers': 2,
    'num_heads': 4
}
```

---

## 4. Optimization Strategies by Constraint

### 4.1 Severe Constraint (2GB GPU)

```python
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    model_params={
        'n_estimators': 8  # Reduce ensemble
    },
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-4,
        'batch_size': 4,           # Very small
        'support_size': 32,        # Small context
        'query_size': 16,
        'mixed_precision': 'fp16', # Use half precision
        'gradient_checkpoint': True,
        'num_workers': 0,          # No parallel loading
        'peft_config': {
            'r': 2,                # Very low rank
            'lora_alpha': 4,
            'lora_dropout': 0.2
        }
    }
)
```

### 4.2 Moderate Constraint (4GB GPU)

```python
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
        'mixed_precision': 'fp16',
        'num_workers': 2,
        'peft_config': {
            'r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.1
        }
    }
)
```

### 4.3 Comfortable (8GB+ GPU)

```python
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',  # Full fine-tuning possible
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5,
        'batch_size': 32,
        'support_size': 128,
        'query_size': 64,
        'num_workers': 4,
        'gradient_accumulation_steps': 2
    }
)
```

---

## 5. Advanced Techniques

### 5.1 Activation Checkpointing

```python
import torch.utils.checkpoint as checkpoint

class CheckpointedModel(torch.nn.Module):
    """Wrap model with checkpointing."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Checkpoint during forward pass
        return checkpoint.checkpoint(
            self.model,
            x,
            use_reentrant=False
        )
```

### 5.2 Quantization

Reduce model precision:

```python
import torch.quantization as quantization

def quantize_model(model, backend='fbgemm'):
    """Quantize model to int8."""
    model.qconfig = quantization.get_default_qconfig(backend)
    quantization.prepare(model, inplace=True)
    # Calibrate on data...
    quantization.convert(model, inplace=True)
    return model

# Usage
quantized_pipeline = quantize_model(pipeline.model)
```

### 5.3 Parameter Sharing

Share weights across layers:

```python
class ParameterSharingModel(torch.nn.Module):
    """Model with shared parameters."""
    
    def __init__(self, shared_layer, num_repeats):
        super().__init__()
        self.shared_layer = shared_layer
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for _ in range(self.num_repeats):
            x = self.shared_layer(x)
        return x
```

### 5.4 Knowledge Distillation

Train smaller student model from larger teacher:

```python
# Large teacher model (high accuracy)
teacher = TabularPipeline(model_name='TabDPT', tuning_strategy='base-ft')
teacher.fit(X_train, y_train)

# Small student model (low memory)
student = TabularPipeline(model_name='TabPFN', tuning_strategy='peft')

# Distillation: train student to match teacher
for batch in dataloader:
    teacher_logits = teacher.model(batch)
    student_logits = student.model(batch)
    
    # KL divergence loss
    loss = nn.KLDivLoss()(
        nn.log_softmax(student_logits),
        nn.softmax(teacher_logits)
    )
    # ... optimize ...
```

---

## 6. Memory-Time Trade-offs

### 6.1 Time-Memory Pareto Frontier

```python
configurations = [
    {
        'name': 'Max Speed',
        'batch_size': 64,
        'precision': 'fp32',
        'time': 30,      # minutes
        'memory': 16     # GB
    },
    {
        'name': 'Balanced',
        'batch_size': 32,
        'precision': 'fp32',
        'time': 45,
        'memory': 12
    },
    {
        'name': 'Efficient',
        'batch_size': 16,
        'precision': 'fp16',
        'time': 60,
        'memory': 6
    },
    {
        'name': 'Ultra-Efficient',
        'batch_size': 8,
        'precision': 'fp16',
        'peft': True,
        'time': 90,
        'memory': 3
    }
]
```

### 6.2 Choosing Configuration

```python
def choose_config(gpu_memory_gb, time_budget_hours):
    """Choose config based on constraints."""
    
    if gpu_memory_gb < 4:
        return 'peft'  # Must use PEFT
    elif gpu_memory_gb < 8:
        return 'fp16_batch16'
    elif gpu_memory_gb < 16:
        return 'fp16_batch32'
    else:
        return 'base_ft'

# Usage
config = choose_config(gpu_memory_gb=6, time_budget=4)
```

---

## 7. Monitoring During Training

### 7.1 Real-time Memory Tracking

```python
import logging
from datetime import datetime

class MemoryTracker:
    """Track memory throughout training."""
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.iteration = 0
        self.history = []
    
    def step(self):
        """Call after each training step."""
        self.iteration += 1
        
        if self.iteration % self.log_interval == 0:
            peak = torch.cuda.max_memory_allocated() / 1e9
            current = torch.cuda.memory_allocated() / 1e9
            
            self.history.append({
                'iteration': self.iteration,
                'peak': peak,
                'current': current,
                'time': datetime.now()
            })
            
            print(f"[{self.iteration}] Peak: {peak:.2f}GB, Current: {current:.2f}GB")
    
    def plot(self):
        """Plot memory over time."""
        import matplotlib.pyplot as plt
        
        iterations = [h['iteration'] for h in self.history]
        peaks = [h['peak'] for h in self.history]
        
        plt.plot(iterations, peaks)
        plt.xlabel('Iteration')
        plt.ylabel('Peak Memory (GB)')
        plt.title('GPU Memory Usage Over Time')
        plt.show()

# Usage
tracker = MemoryTracker()

for batch in dataloader:
    # ... training step ...
    tracker.step()

tracker.plot()
```

---

## 8. Debugging Memory Issues

### 8.1 OOM (Out of Memory) Error Handling

```python
def safe_train(model, dataloader, max_retries=3):
    """Train with automatic memory adaptation."""
    
    batch_size = 32
    
    for attempt in range(max_retries):
        try:
            # Try training with current batch size
            for batch in dataloader:
                # ... training code ...
                pass
            
            return  # Success
        
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # Reduce batch size and retry
                batch_size //= 2
                print(f"OOM detected. Retrying with batch_size={batch_size}")
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Recreate dataloader with smaller batch
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size
                )
            else:
                raise
    
    raise RuntimeError(f"Failed after {max_retries} attempts")
```

### 8.2 Memory Leak Detection

```python
import tracemalloc

tracemalloc.start()

# Your training code
pipeline.fit(X_train, y_train)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e9:.2f}GB; Peak: {peak / 1e9:.2f}GB")

# Find top memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
```

---

## 9. Complete Example: Memory-Optimized Training

```python
from tabtune import TabularPipeline
import torch

def memory_optimized_training(
    X_train, y_train, X_val, y_val,
    gpu_memory_gb=4
):
    """Train with automatic memory optimization."""
    
    # Determine configuration
    if gpu_memory_gb < 4:
        use_peft = True
        batch_size = 4
        support_size = 32
        mixed_precision = 'fp16'
        rank = 2
    elif gpu_memory_gb < 8:
        use_peft = True
        batch_size = 8
        support_size = 64
        mixed_precision = 'fp16'
        rank = 4
    else:
        use_peft = False
        batch_size = 32
        support_size = 128
        mixed_precision = None
        rank = 8
    
    # Create pipeline
    strategy = 'peft' if use_peft else 'base-ft'
    
    tuning_params = {
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4 if use_peft else 2e-5,
        'batch_size': batch_size,
        'support_size': support_size,
        'num_workers': 0,
        'pin_memory': False if gpu_memory_gb < 4 else True
    }
    
    if mixed_precision:
        tuning_params['mixed_precision'] = mixed_precision
    
    if use_peft:
        tuning_params['peft_config'] = {
            'r': rank,
            'lora_alpha': 2 * rank,
            'lora_dropout': 0.05
        }
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy=strategy,
        tuning_params=tuning_params
    )
    
    # Train with monitoring
    print(f"Training with strategy={strategy}, batch_size={batch_size}")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    metrics = pipeline.evaluate(X_val, y_val)
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    
    return pipeline

# Usage
pipeline = memory_optimized_training(
    X_train, y_train, X_val, y_val,
    gpu_memory_gb=4
)
```

---

## 10. Quick Reference

### Memory Reduction Techniques (by impact)

| Technique | Memory Saving | Time Overhead | Effort |
|-----------|---------------|---------------|--------|
| PEFT | 60-90% | None | Low |
| Batch size ÷2 | 50% | None | Low |
| Mixed precision | 20-30% | 20-30% | Medium |
| Gradient accumulation | 0% | 0% | Low |
| Gradient checkpoint | 30-50% | 20-30% | Medium |
| Quantization | 75% | 5-10% | High |

---

## 11. Best Practices

### ✅ Do's

- ✅ Profile memory before optimization
- ✅ Use PEFT for memory-constrained environments
- ✅ Start with small batch sizes
- ✅ Use mixed precision when possible
- ✅ Monitor memory during training
- ✅ Empty cache between experiments
- ✅ Use gradient accumulation for large effective batches

### ❌ Don'ts

- ❌ Don't forget to clear cache between runs
- ❌ Don't use full precision when half works
- ❌ Don't load entire dataset to memory
- ❌ Don't tune without monitoring memory
- ❌ Don't ignore OOM errors

---

## 12. Next Steps

- [Tuning Strategies](../user-guide/tuning-strategies.md) - PEFT details
- [PEFT & LoRA](peft-lora.md) - Memory-efficient fine-tuning
- [Advanced Topics](peft-lora.md) - Advanced optimizations
- [Multi-GPU](multi-gpu.md) - Distributed training

---

Optimize memory strategically to train powerful models on limited resources!