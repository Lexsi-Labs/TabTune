"""
Example 3: Fine-Tuning Strategies Comparison
============================================

This example demonstrates TabTune's four fine-tuning strategies, showing how
each strategy works and when to use them.

Fine-Tuning Strategies:
1. Inference: Zero-shot predictions (no training)
2. Meta-Learning: Episodic fine-tuning (recommended for ICL models)
3. SFT (Supervised Fine-Tuning): Standard supervised training
4. PEFT: Parameter-Efficient Fine-Tuning with LoRA adapters

Key Learning Points:
- Different strategies for different use cases
- Meta-learning mimics in-context learning paradigm
- PEFT reduces memory usage significantly
- Easy to switch between strategies

Dataset: OpenML 42178 (Telco Customer Churn)
Industry: Telecom
Samples: ~7043
Task: Binary classification (customer churn prediction)
Note: Imbalanced dataset, good for demonstrating different strategies
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
# DATA LOADING: OpenML Telco Customer Churn Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 3: Fine-Tuning Strategies Comparison")
logger.info("="*80)
logger.info("\n📊 Loading Telco Customer Churn Dataset (OpenML ID: 42178)...")
logger.info("   Industry: Telecom")
logger.info("   Task: Predict customer churn")
logger.info("   Note: Imbalanced dataset - good for strategy comparison")

try:
    # Load the Telco Customer Churn dataset
    dataset = openml.datasets.get_dataset(42178, download_data=True, download_qualities=False)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Convert to pandas DataFrame/Series
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='target')
    
    logger.info(f"✅ Successfully loaded dataset: {dataset.name}")
    logger.info(f"   - Features: {X.shape[1]}")
    logger.info(f"   - Samples: {X.shape[0]}")
    logger.info(f"   - Target classes: {len(np.unique(y))}")
    logger.info(f"   - Target distribution: {dict(y.value_counts())}")
    
except Exception as e:
    logger.error(f"❌ Failed to load OpenML dataset: {e}")
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
# DEMONSTRATION: Four Fine-Tuning Strategies
# ============================================================================

logger.info("\n" + "="*80)
logger.info("KEY DEMONSTRATION: Four Fine-Tuning Strategies")
logger.info("="*80)

results = {}

# ============================================================================
# Strategy 1: Inference (Zero-Shot)
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STRATEGY 1: Inference (Zero-Shot Learning)")
logger.info("="*80)
logger.info("\n📝 What it does:")
logger.info("   - Uses pre-trained weights directly")
logger.info("   - No training/fine-tuning performed")
logger.info("   - Fastest approach")
logger.info("   - Best for: Quick experiments, baseline comparison")

logger.info("\n🔄 Initializing inference pipeline...")
pipeline_inference = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='inference',  # Zero-shot, no fine-tuning
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

logger.info("✅ Inference pipeline initialized")
logger.info("🔄 Fitting (just setting up inference context, no training)...")
pipeline_inference.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_inference = pipeline_inference.evaluate(X_test, y_test)
results['Inference'] = metrics_inference
logger.info(f"   Accuracy: {metrics_inference.get('accuracy', 0):.4f}")

# ============================================================================
# Strategy 2: Meta-Learning Fine-Tuning
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STRATEGY 2: Meta-Learning Fine-Tuning")
logger.info("="*80)
logger.info("\n📝 What it does:")
logger.info("   - Episodic training that mimics in-context learning")
logger.info("   - Creates support/query splits from training data")
logger.info("   - Trains model to learn from examples in-context")
logger.info("   - Best for: ICL models (TabICL, OrionMSP, TabDPT, Mitra)")
logger.info("   - Recommended: Default for ICL models")

logger.info("\n🔄 Initializing meta-learning pipeline...")
pipeline_meta = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='finetune',  # Fine-tuning enabled
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,  # Number of training epochs
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',  # Explicitly set meta-learning mode
        'support_size': 100,  # Size of support set for each episode
        'query_size': 50,     # Size of query set for each episode
        'n_episodes': 50,     # Number of training episodes
        'show_progress': True
    }
)

logger.info("✅ Meta-learning pipeline initialized")
logger.info("🔄 Fitting with meta-learning (episodic training)...")
pipeline_meta.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_meta = pipeline_meta.evaluate(X_test, y_test)
results['Meta-Learning'] = metrics_meta
logger.info(f"   Accuracy: {metrics_meta.get('accuracy', 0):.4f}")

# ============================================================================
# Strategy 3: Supervised Fine-Tuning (SFT)
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STRATEGY 3: Supervised Fine-Tuning (SFT)")
logger.info("="*80)
logger.info("\n📝 What it does:")
logger.info("   - Standard supervised training on batches")
logger.info("   - Uses all training data in each epoch")
logger.info("   - More traditional fine-tuning approach")
logger.info("   - Best for: When you want standard supervised learning")

logger.info("\n🔄 Initializing SFT pipeline...")
pipeline_sft = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,
        'learning_rate': 1e-5,
        'finetune_mode': 'sft',  # Supervised Fine-Tuning mode
        'batch_size': 32,        # Batch size for supervised training
        'show_progress': True
    }
)

logger.info("✅ SFT pipeline initialized")
logger.info("🔄 Fitting with SFT (supervised training on batches)...")
pipeline_sft.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_sft = pipeline_sft.evaluate(X_test, y_test)
results['SFT'] = metrics_sft
logger.info(f"   Accuracy: {metrics_sft.get('accuracy', 0):.4f}")

# ============================================================================
# Strategy 4: Parameter-Efficient Fine-Tuning (PEFT)
# ============================================================================

logger.info("\n" + "="*80)
logger.info("STRATEGY 4: Parameter-Efficient Fine-Tuning (PEFT/LoRA)")
logger.info("="*80)
logger.info("\n📝 What it does:")
logger.info("   - Uses LoRA (Low-Rank Adaptation) adapters")
logger.info("   - Only trains a small subset of parameters")
logger.info("   - Reduces memory usage by 60-80%")
logger.info("   - Best for: Large models, limited GPU memory")
logger.info("   - Trade-off: Slightly lower performance, much less memory")

logger.info("\n🔄 Initializing PEFT pipeline...")
pipeline_peft = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='peft',  # PEFT mode
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 3,
        'learning_rate': 5e-5,  # Higher LR common for PEFT
        'finetune_mode': 'meta-learning',  # Can combine PEFT with meta-learning
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'show_progress': True,
        # PEFT/LoRA configuration
        'peft_config': {
            'r': 8,              # LoRA rank (lower = fewer parameters)
            'lora_alpha': 16,    # Scaling factor
            'lora_dropout': 0.05 # Dropout in LoRA modules
        }
    }
)

logger.info("✅ PEFT pipeline initialized")
logger.info("   💾 Memory efficient: Only ~20% of parameters trained")
logger.info("🔄 Fitting with PEFT (memory-efficient training)...")
pipeline_peft.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_peft = pipeline_peft.evaluate(X_test, y_test)
results['PEFT'] = metrics_peft
logger.info(f"   Accuracy: {metrics_peft.get('accuracy', 0):.4f}")

# ============================================================================
# SUMMARY: Strategy Comparison
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Fine-Tuning Strategy Comparison")
logger.info("="*80)
logger.info("\n📊 Performance Comparison:")

for strategy, metrics in results.items():
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_score', 0)
    logger.info(f"   {strategy:15s} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

logger.info("\n✨ When to Use Each Strategy:")
logger.info("\n   1. Inference:")
logger.info("      - Quick baseline/experiments")
logger.info("      - No training time available")
logger.info("      - Pre-trained model is already good")
logger.info("\n   2. Meta-Learning:")
logger.info("      - ICL models (TabICL, OrionMSP, TabDPT, Mitra)")
logger.info("      - Want to improve in-context learning ability")
logger.info("      - Recommended default for ICL models")
logger.info("\n   3. SFT:")
logger.info("      - Traditional supervised learning")
logger.info("      - When you want standard batch training")
logger.info("      - Good for non-ICL models")
logger.info("\n   4. PEFT:")
logger.info("      - Limited GPU memory")
logger.info("      - Large models")
logger.info("      - Want to fine-tune with minimal resources")
logger.info("      - Can combine with meta-learning or SFT")

logger.info("\n" + "="*80)
logger.info("✅ Example 3 Complete: Fine-Tuning Strategies Demonstration")
logger.info("="*80)
