# Hyperparameter Tuning: Optimizing TabTune Model Performance

This document provides comprehensive guidance on hyperparameter tuning for TabTune models, including strategies, tools, and best practices for finding optimal configurations.

---

## 1. Introduction

Hyperparameter tuning is the process of systematically searching for the best model configuration to maximize performance on your specific task. This guide covers:

- **Search Strategies**: Grid search, random search, Bayesian optimization
- **Hyperparameter Spaces**: Ranges and distributions for each model
- **Tuning Tools**: Integration with Optuna, scikit-optimize, and hyperopt
- **Best Practices**: Efficient tuning workflows and validation strategies

---

## 2. Hyperparameter Landscape

### 2.1 Tunable Hyperparameters by Model

| Model | Critical | Important | Minor |
|-------|----------|-----------|-------|
| **TabPFN** | epochs, lr | temperature | batch_size |
| **TabICL** | n_episodes, lr | support_size, query_size, n_estimators | norm_methods |
| **OrionMSP** | n_episodes, lr | support_size, query_size | n_estimators |
| **OrionBix** | n_episodes, lr | support_size, query_size | n_estimators |
| **TabDPT** | support_size, lr | k_neighbors, num_layers | temperature |
| **Mitra** | support_size, lr | batch_size, num_layers | d_model |
| **ContextTab** | epochs, lr | warmup_steps | text_encoder |

### 2.2 Shared Hyperparameters

```python
shared_hparams = {
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    'epochs': [1, 3, 5, 10],
    'batch_size': [8, 16, 32, 64],
    'weight_decay': [0.0, 0.01, 0.1],
    'warmup_steps': [0, 100, 500, 1000]
}

# PEFT-specific
peft_hparams = {
    'r': [2, 4, 8, 16],
    'lora_alpha': [4, 8, 16, 32],
    'lora_dropout': [0.0, 0.05, 0.1, 0.2]
}
```

---

## 3. Search Strategies

### 3.1 Grid Search

Systematic evaluation of all parameter combinations.

**Advantages**:
- ✅ Exhaustive coverage
- ✅ Parallelize easily
- ✅ Reproducible

**Disadvantages**:
- ❌ Exponential complexity
- ❌ Wasteful on large spaces
- ❌ Poor scaling

```python
from sklearn.model_selection import ParameterGrid
from tabtune import TabularPipeline

# Define grid
param_grid = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'epochs': [3, 5],
    'batch_size': [16, 32]
}

# Grid search
best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='base-ft',
        tuning_params=params
    )
    
    pipeline.fit(X_train, y_train)
    score = pipeline.evaluate(X_val, y_val)['accuracy']
    
    if score > best_score:
        best_score = score
        best_params = params
        print(f"New best: {score:.4f} with {params}")

print(f"\nBest parameters: {best_params}")
print(f"Best score: {best_score:.4f}")
```

### 3.2 Random Search

Random sampling from parameter distributions.

**Advantages**:
- ✅ Covers parameter space more uniformly
- ✅ Scales well to large spaces
- ✅ Simple parallelization

**Disadvantages**:
- ❌ May miss optimal region
- ❌ Less reproducible

```python
import numpy as np
from scipy.stats import uniform, randint

# Define distributions
param_distributions = {
    'learning_rate': uniform(1e-5, 1e-3),
    'epochs': randint(1, 20),
    'batch_size': randint(8, 128),
    'weight_decay': uniform(0.0, 0.1)
}

# Random search
n_iter = 20
best_score = 0
best_params = None

for i in range(n_iter):
    # Sample random parameters
    params = {
        key: dist.rvs()
        for key, dist in param_distributions.items()
    }
    
    # Convert to integers
    params['epochs'] = int(params['epochs'])
    params['batch_size'] = int(params['batch_size'])
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='base-ft',
        tuning_params=params
    )
    
    pipeline.fit(X_train, y_train)
    score = pipeline.evaluate(X_val, y_val)['accuracy']
    
    if score > best_score:
        best_score = score
        best_params = params
        print(f"Iteration {i+1}/{n_iter}: {score:.4f}")

print(f"\nBest parameters: {best_params}")
```

### 3.3 Bayesian Optimization with Optuna

Intelligent search using Gaussian processes.

**Advantages**:
- ✅ Intelligent sampling
- ✅ Few evaluations needed
- ✅ Adaptively explores promising regions

**Disadvantages**:
- ❌ More complex
- ❌ Slower per iteration
- ❌ Requires more setup

```python
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    """Optuna objective function."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 1, 20)
    batch_size = trial.suggest_int('batch_size', 8, 128)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    
    tuning_params = {
        'device': 'cuda',
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay
    }
    
    # Train and evaluate
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='base-ft',
        tuning_params=tuning_params
    )
    
    pipeline.fit(X_train, y_train)
    score = pipeline.evaluate(X_val, y_val)['accuracy']
    
    return score

# Create study
study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner()
)

# Optimize
study.optimize(objective, n_trials=50, n_jobs=1)

# Get results
print(f"Best score: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

---

## 4. Model-Specific Tuning

### 4.1 TabPFN Tuning

Focus on inference parameters since base-ft is not primary use case:

```python
# Key hyperparameters
tabpfn_hparams = {
    'n_estimators': [8, 16, 32],           # Ensemble size
    'softmax_temperature': [0.5, 0.9, 1.5],  # Confidence
    'epochs': [1, 3, 5],                   # If fine-tuning
    'learning_rate': [1e-5, 2e-5, 5e-5]   # If fine-tuning
}

# Fine-tune only if needed
pipeline = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'learning_rate': 2e-5
    }
)
```

### 4.2 TabICL/OrionMSP/OrionBix Tuning

Optimize episodic training parameters:

```python
# Key hyperparameters
tabicl_hparams = {
    'support_size': [24, 48, 96],          # Context size
    'query_size': [16, 32, 64],            # Query size
    'n_episodes': [500, 1000, 2000],       # Training episodes
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'n_estimators': [16, 32, 64]           # Ensemble
}

# Recommended defaults
best_config = {
    'support_size': 48,
    'query_size': 32,
    'n_episodes': 1000,
    'learning_rate': 2e-5,
    'epochs': 5
}
```

### 4.3 TabDPT Tuning

Leverage pre-training and large context:

```python
# Key hyperparameters
tabdpt_hparams = {
    'support_size': [512, 1024, 2048],     # Large context
    'query_size': [128, 256, 512],
    'k_neighbors': [3, 5, 10],             # k-NN context
    'num_layers': [2, 4, 8],               # Architecture
    'learning_rate': [1e-5, 2e-5, 5e-5]
}

# Recommended for large datasets
best_config = {
    'support_size': 1024,
    'query_size': 256,
    'k_neighbors': 5,
    'learning_rate': 2e-5,
    'epochs': 3  # Few due to pre-training
}
```

### 4.4 Mitra Tuning

Optimize 2D attention parameters:

```python
# Key hyperparameters
mitra_hparams = {
    'support_size': [64, 128, 256],
    'query_size': [64, 128, 256],
    'd_model': [32, 64, 128],              # Embedding dim
    'num_layers': [1, 2, 4],
    'batch_size': [2, 4, 8],               # Must be small
    'learning_rate': [1e-5, 2e-5, 5e-5]
}

# Recommended
best_config = {
    'support_size': 128,
    'query_size': 128,
    'd_model': 64,
    'num_layers': 2,
    'batch_size': 4,  # Critical: keep small
    'learning_rate': 1e-5
}
```

### 4.5 PEFT Tuning

Optimize LoRA parameters:

```python
# LoRA hyperparameters
peft_hparams = {
    'r': [2, 4, 8, 16],
    'lora_alpha': [4, 8, 16, 32],  # Usually 2x rank
    'lora_dropout': [0.05, 0.1, 0.2],
    'learning_rate': [1e-4, 2e-4, 5e-4]  # Higher than base-ft
}

# Recommended
best_peft_config = {
    'r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'learning_rate': 2e-4  # 10x base-ft
}
```

---

## 5. Cross-Validation Strategy

### 5.1 k-Fold Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_hyperparams(X, y, model_name, params, k=5):
    """Evaluate hyperparameters via k-fold CV."""
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train and evaluate
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='base-ft',
            tuning_params=params
        )
        
        pipeline.fit(X_train_fold, y_train_fold)
        score = pipeline.evaluate(X_val_fold, y_val_fold)['accuracy']
        scores.append(score)
        
        print(f"Fold {fold_idx+1}/{k}: {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\nMean: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score

# Usage
mean, std = cross_validate_hyperparams(
    X_train, y_train,
    model_name='TabICL',
    params={'epochs': 5, 'learning_rate': 2e-5},
    k=5
)
```

### 5.2 Stratified k-Fold

For imbalanced classification:

```python
from sklearn.model_selection import StratifiedKFold

# Ensures class distribution preserved
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    # ... training code
```

---

## 6. Efficient Tuning Workflows

### 6.1 Cascading Search

Start coarse, then fine-grained:

```python
# Stage 1: Coarse grid search
stage1_params = {
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'epochs': [3, 5, 10]
}

# Find best in stage 1
best_params_stage1 = grid_search(stage1_params)

# Stage 2: Fine-grained around best
stage2_params = {
    'learning_rate': [
        best_params_stage1['learning_rate'] / 2,
        best_params_stage1['learning_rate'],
        best_params_stage1['learning_rate'] * 2
    ],
    'epochs': [
        best_params_stage1['epochs'] - 1,
        best_params_stage1['epochs'],
        best_params_stage1['epochs'] + 1
    ]
}

# Find best in stage 2
best_params_stage2 = grid_search(stage2_params)
```

### 6.2 Early Stopping During Tuning

```python
class EarlyStoppingTuner:
    """Tuner with early stopping."""
    
    def __init__(self, patience=3):
        self.patience = patience
        self.best_score = 0
        self.no_improve_count = 0
    
    def should_stop(self, score):
        """Check if tuning should stop."""
        if score > self.best_score:
            self.best_score = score
            self.no_improve_count = 0
            return False
        else:
            self.no_improve_count += 1
            return self.no_improve_count >= self.patience

# Usage
tuner = EarlyStoppingTuner(patience=5)

for params in param_grid:
    # ... train and evaluate ...
    
    if tuner.should_stop(score):
        print(f"Early stopping after {tuner.no_improve_count} iterations")
        break
```

---

## 7. Complete Tuning Examples

### 7.1 Simple Grid Search with Leaderboard

```python
from tabtune import TabularLeaderboard

# Define parameter grid
param_grid = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'epochs': [3, 5],
    'batch_size': [16, 32]
}

# Generate all combinations
from itertools import product
param_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in product(*param_grid.values())
]

# Use leaderboard for comparison
leaderboard = TabularLeaderboard(X_train, X_val, y_train, y_val)

for i, params in enumerate(param_combinations):
    config_name = f"Config_{i+1}_{params['learning_rate']}_{params['epochs']}"
    leaderboard.add_model(
        'TabICL',
        'base-ft',
        name=config_name,
        tuning_params=params
    )

results = leaderboard.run(rank_by='accuracy')
print(leaderboard.get_ranking())
```

### 7.2 Bayesian Optimization with Pruning

```python
import optuna

def objective_with_pruning(trial):
    """Objective with early pruning."""
    
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 1, 20)
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='base-ft',
        tuning_params={
            'learning_rate': learning_rate,
            'epochs': epochs
        }
    )
    
    # Train with intermediate reporting
    for epoch in range(epochs):
        # ... train for one epoch ...
        
        # Evaluate
        score = pipeline.evaluate(X_val, y_val)['accuracy']
        
        # Report intermediate value
        trial.report(score, epoch)
        
        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score

# Create study with pruning
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective_with_pruning, n_trials=30)
```

### 7.3 Parallel Tuning with Ray

```python
import ray
from ray import tune

# Initialize Ray
ray.init()

def train_model(config):
    """Trainable function for Ray."""
    
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='base-ft',
        tuning_params=config
    )
    
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_val, y_val)
    
    return metrics

# Parallel tuning
results = tune.run(
    train_model,
    config={
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'epochs': tune.randint(1, 20),
        'batch_size': tune.choice([16, 32, 64])
    },
    num_samples=30,
    verbose=1
)

ray.shutdown()
```

---

## 8. Tuning Summary & Defaults

### 8.1 Quick Reference Table

| Model | Learning Rate | Epochs | Support Size | Key Parameter |
|-------|---------------|--------|--------------|----------------|
| TabPFN | 2e-5 | 3-5 | N/A | n_estimators |
| TabICL | 2e-5 | 5 | 48 | n_episodes |
| OrionMSP | 2e-5 | 5 | 1024 | n_episodes |
| OrionBix | 2e-5 | 5 | 48 | n_episodes |
| TabDPT | 2e-5 | 3 | 1024 | k_neighbors |
| Mitra | 1e-5 | 3 | 128 | batch_size |
| ContextTab | 1e-4 | 10 | N/A | warmup_steps |
| PEFT | 2e-4 | 5 | 48 | rank (r) |

### 8.2 Tuning Priority

1. **Learning Rate**: 80% of impact
2. **Epochs**: 10% of impact
3. **Batch/Support Size**: 5% of impact
4. **Other**: 5% of impact

---

## 9. Best Practices

### ✅ Do's

- ✅ Start with default hyperparameters
- ✅ Use cross-validation for robustness
- ✅ Tune on validation set, evaluate on test set
- ✅ Focus on high-impact hyperparameters first
- ✅ Use multiple seeds for stability
- ✅ Log all experiments
- ✅ Parallelize when possible

### ❌ Don'ts

- ❌ Don't tune on test set
- ❌ Don't use learning rate as only tunable parameter
- ❌ Don't ignore data size when choosing ranges
- ❌ Don't forget to freeze random seed
- ❌ Don't tune without validation set
- ❌ Don't skip early stopping

---

## 10. Common Pitfalls & Solutions

### Issue: "Tuning is too slow"
**Solution**: 
- Use Bayesian optimization instead of grid search
- Parallelize across cores
- Use early stopping

### Issue: "Best tuned model still overfits"
**Solution**:
- Increase regularization (weight decay)
- Use PEFT instead of base-ft
- Reduce learning rate
- Add dropout

### Issue: "Tuning results don't transfer to test set"
**Solution**:
- Use larger validation set
- Use cross-validation
- Don't overfit to validation set
- Use proper hyperparameter ranges

---

## 11. Next Steps

- [Tuning Strategies](../user-guide/tuning-strategies.md) - Strategy details
- [Model Selection](../user-guide/model-selection.md) - Choosing models
- [TabularLeaderboard](../user-guide/leaderboard.md) - Systematic comparison
- [PEFT & LoRA](peft-lora.md) - PEFT-specific tuning

---

Systematic hyperparameter tuning unlocks the full potential of TabTune models!