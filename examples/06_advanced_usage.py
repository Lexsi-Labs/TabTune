"""
Example 6: Advanced Usage - PEFT Configuration and Hybrid Strategies
====================================================================

This example demonstrates advanced TabTune features:
- Detailed PEFT/LoRA configuration
- Hybrid strategies (combining meta-learning with PEFT)
- Custom preprocessing pipelines
- Advanced tuning parameters

Key Learning Points:
- Fine-tune PEFT configuration for optimal performance
- Combine different fine-tuning strategies
- Customize preprocessing for specific needs
- Advanced hyperparameter tuning

Dataset: OpenML 41027 (employee-satisfaction) or 1480 (sick)
Industry: HR/Business or Healthcare
Task: Classification
Note: Demonstrates advanced configurations
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import logging
import random
import openml

# Import TabTune components
from tabtune import TabularPipeline
from tabtune.logger import setup_logger

# ============================================================================
# SETUP: Reproducibility and Logging
# ============================================================================

def set_global_seeds(seed_value):
    """Set random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_global_seeds(42)

setup_logger(use_rich=True)
logger = logging.getLogger('tabtune')

# ============================================================================
# DATA LOADING: OpenML Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 6: Advanced Usage - PEFT and Hybrid Strategies")
logger.info("="*80)

# Try to load employee-satisfaction, fallback to sick, then breast_cancer
datasets_to_try = [41027, 1480]
dataset_loaded = False

for dataset_id in datasets_to_try:
    try:
        logger.info(f"\n📊 Loading OpenML dataset ID: {dataset_id}...")
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')
        
        logger.info(f"✅ Successfully loaded dataset: {dataset.name}")
        logger.info(f"   - Features: {X.shape[1]}")
        logger.info(f"   - Samples: {X.shape[0]}")
        dataset_loaded = True
        break
    except Exception as e:
        logger.warning(f"⚠️  Failed to load dataset {dataset_id}: {e}")
        continue

if not dataset_loaded:
    logger.info("Falling back to sklearn breast cancer dataset...")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    logger.info("✅ Loaded fallback dataset")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

logger.info(f"\n📊 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# ============================================================================
# ADVANCED FEATURE 1: Detailed PEFT Configuration
# ============================================================================

logger.info("\n" + "="*80)
logger.info("ADVANCED FEATURE 1: Detailed PEFT/LoRA Configuration")
logger.info("="*80)
logger.info("\n📝 PEFT (Parameter-Efficient Fine-Tuning) uses LoRA adapters")
logger.info("   Key parameters to tune:")
logger.info("   - r (rank): Lower = fewer parameters, faster training")
logger.info("   - lora_alpha: Scaling factor for LoRA updates")
logger.info("   - lora_dropout: Regularization in LoRA modules")
logger.info("   - target_modules: Which layers to apply LoRA (auto-detect if None)\n")

# PEFT Configuration 1: Low-rank (fewer parameters)
logger.info("🔧 Configuration 1: Low-rank PEFT (r=4)")
logger.info("   💾 Memory efficient, faster training, slightly lower capacity")

pipeline_peft_low = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,
        'learning_rate': 5e-5,
        'finetune_mode': 'meta-learning',
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'show_progress': False,
        'peft_config': {
            'r': 4,              # Low rank = fewer parameters
            'lora_alpha': 8,      # Scaling factor
            'lora_dropout': 0.05,
            'target_modules': None  # Auto-detect which layers to apply LoRA
        }
    }
)

logger.info("🔄 Training with low-rank PEFT...")
pipeline_peft_low.fit(X_train, y_train)
metrics_low = pipeline_peft_low.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_low.get('accuracy', 0):.4f}")

# PEFT Configuration 2: Higher-rank (more parameters)
logger.info("\n🔧 Configuration 2: Higher-rank PEFT (r=16)")
logger.info("   💾 More parameters, better capacity, slightly more memory")

pipeline_peft_high = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,
        'learning_rate': 5e-5,
        'finetune_mode': 'meta-learning',
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'show_progress': False,
        'peft_config': {
            'r': 16,             # Higher rank = more parameters
            'lora_alpha': 32,    # Higher scaling
            'lora_dropout': 0.1,  # Higher dropout for regularization
            'target_modules': None
        }
    }
)

logger.info("🔄 Training with higher-rank PEFT...")
pipeline_peft_high.fit(X_train, y_train)
metrics_high = pipeline_peft_high.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_high.get('accuracy', 0):.4f}")

# ============================================================================
# ADVANCED FEATURE 2: Hybrid Strategy - Meta-Learning + PEFT
# ============================================================================

logger.info("\n" + "="*80)
logger.info("ADVANCED FEATURE 2: Hybrid Strategy (Meta-Learning + PEFT)")
logger.info("="*80)
logger.info("\n📝 You can combine strategies:")
logger.info("   - Use PEFT for parameter efficiency")
logger.info("   - Use meta-learning for episodic training")
logger.info("   - Best of both worlds: Efficient + Effective\n")

logger.info("🔧 Hybrid Configuration: PEFT with Meta-Learning")
pipeline_hybrid = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='peft',  # Use PEFT for efficiency
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 3,
        'learning_rate': 5e-5,
        'finetune_mode': 'meta-learning',  # Combine with meta-learning
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'show_progress': False,
        'peft_config': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        }
    }
)

logger.info("🔄 Training with hybrid strategy (PEFT + Meta-Learning)...")
logger.info("   💡 Benefits:")
logger.info("      - Memory efficient (PEFT)")
logger.info("      - Effective training (Meta-Learning)")
pipeline_hybrid.fit(X_train, y_train)
metrics_hybrid = pipeline_hybrid.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_hybrid.get('accuracy', 0):.4f}")

# ============================================================================
# ADVANCED FEATURE 3: Custom Preprocessing Pipeline
# ============================================================================

logger.info("\n" + "="*80)
logger.info("ADVANCED FEATURE 3: Custom Preprocessing Pipeline")
logger.info("="*80)
logger.info("\n📝 Customize preprocessing for specific needs:")
logger.info("   - Advanced imputation strategies")
logger.info("   - Custom feature selection")
logger.info("   - Specialized resampling\n")

logger.info("🔧 Custom Preprocessing Configuration")
pipeline_custom = TabularPipeline(
    model_name='OrionMSP',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    # Custom preprocessing parameters
    processor_params={
        'imputation_strategy': 'iterative',      # Advanced imputation
        'categorical_encoding': 'target',        # Target encoding for categoricals
        'scaling_strategy': 'robust',            # Robust to outliers
        'resampling_strategy': 'smote',         # SMOTE for imbalanced data
        'feature_selection_strategy': 'select_k_best_anova'  # Feature selection
    }
)

logger.info("🔄 Training with custom preprocessing...")
logger.info("   ⚙️  Custom settings:")
logger.info("      - Iterative imputation")
logger.info("      - Target encoding for categoricals")
logger.info("      - Robust scaling")
logger.info("      - SMOTE resampling")
logger.info("      - ANOVA-based feature selection")
pipeline_custom.fit(X_train, y_train)
metrics_custom = pipeline_custom.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_custom.get('accuracy', 0):.4f}")

# ============================================================================
# ADVANCED FEATURE 4: Advanced Tuning Parameters
# ============================================================================

logger.info("\n" + "="*80)
logger.info("ADVANCED FEATURE 4: Advanced Tuning Parameters")
logger.info("="*80)
logger.info("\n📝 Fine-tune training process:")
logger.info("   - Batch sizes and gradient accumulation")
logger.info("   - Learning rate scheduling")
logger.info("   - Episode configuration for meta-learning\n")

logger.info("🔧 Advanced Tuning Configuration")
pipeline_advanced = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        # Advanced meta-learning parameters
        'support_size': 128,      # Support set size
        'query_size': 64,         # Query set size
        'steps_per_epoch': 10,    # Steps per epoch
        'batch_size': 4,          # Batch size for processing
        'show_progress': False
    },
    processor_params={
        'resampling_strategy': 'none'  # Disable resampling for this example
    }
)

logger.info("🔄 Training with advanced tuning parameters...")
pipeline_advanced.fit(X_train, y_train)
metrics_advanced = pipeline_advanced.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_advanced.get('accuracy', 0):.4f}")

# ============================================================================
# SUMMARY: Advanced Usage
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Advanced Usage Features")
logger.info("="*80)
logger.info("\n📊 Performance Comparison:")
logger.info(f"   Low-rank PEFT (r=4)    - Accuracy: {metrics_low.get('accuracy', 0):.4f}")
logger.info(f"   High-rank PEFT (r=16)  - Accuracy: {metrics_high.get('accuracy', 0):.4f}")
logger.info(f"   Hybrid (PEFT+Meta)     - Accuracy: {metrics_hybrid.get('accuracy', 0):.4f}")
logger.info(f"   Custom Preprocessing   - Accuracy: {metrics_custom.get('accuracy', 0):.4f}")
logger.info(f"   Advanced Tuning        - Accuracy: {metrics_advanced.get('accuracy', 0):.4f}")

logger.info("\n✨ Key Takeaways:")
logger.info("   1. PEFT Tuning: Adjust r, alpha, dropout for optimal balance")
logger.info("   2. Hybrid Strategies: Combine PEFT with meta-learning")
logger.info("   3. Custom Preprocessing: Tailor preprocessing to your data")
logger.info("   4. Advanced Parameters: Fine-tune training process")

logger.info("\n💡 Tips:")
logger.info("   - Start with default PEFT config, then tune")
logger.info("   - Lower r for memory constraints, higher r for better performance")
logger.info("   - Combine strategies for best results")
logger.info("   - Custom preprocessing can significantly improve results")

logger.info("\n" + "="*80)
logger.info("✅ Example 6 Complete: Advanced Usage Demonstration")
logger.info("="*80)
