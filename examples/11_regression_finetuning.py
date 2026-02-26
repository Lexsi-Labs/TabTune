"""
Example 11: Fine-Tuning Strategies Comparison for Regression
============================================================

This example demonstrates TabTune's fine-tuning strategies explicitly applied
to Regression tasks, showing how different pipelines train on continuous targets.

Fine-Tuning Strategies:
1. Inference: Zero-shot predictions (no training)
2. Meta-Learning: Episodic fine-tuning (recommended for ICL models like TabDPT/ContextTab)
3. SFT (turn_by_turn): Sequential tuning (specifically requested by models like TabPFN)

Key Learning Points:
- Different strategies for different regression models
- Meta-learning mimics in-context learning predicting continuous targets
- Easy to switch between strategies safely

Dataset: California Housing (Subsampled)
Industry: Real Estate
Samples: ~1000
Task: Regression (predicting house prices)
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch
import logging
import random

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
# DATA LOADING: California Housing Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 12: Fine-Tuning Strategies Comparison for Regression")
logger.info("="*80)
logger.info("\\n📊 Loading California Housing Dataset...")
logger.info("   Task: Predict median house value (Continuous Target)")

try:
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    
    # Use a smaller subset for faster example execution
    X = X.iloc[:1000]
    y = y.iloc[:1000]
    
    logger.info(f"✅ Successfully loaded dataset: California Housing")
    logger.info(f"   - Features: {X.shape[1]}")
    logger.info(f"   - Samples: {X.shape[0]}")
    logger.info(f"   - Target Range: [{y.min():.2f}, {y.max():.2f}]")
    
except Exception as e:
    logger.error(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

results = {}

# ============================================================================
# DEMONSTRATION: Four Fine-Tuning Strategies
# ============================================================================

logger.info("\\n" + "="*80)
logger.info("KEY DEMONSTRATION: Fine-Tuning Strategies for Regression")
logger.info("="*80)

# ============================================================================
# Strategy 1: Inference (Zero-Shot) [ContextTab]
# ============================================================================

logger.info("\\n" + "="*80)
logger.info("STRATEGY 1: Inference (Zero-Shot Learning)")
logger.info("="*80)
logger.info("\\n📝 What it does:")
logger.info("   - Uses pre-trained weights directly")
logger.info("   - No training/fine-tuning performed on the continuous targets")

logger.info("\\n🔄 Initializing inference pipeline (ContextTab)...")
pipeline_inference = TabularPipeline(
    model_name='Mitra',
    task_type='regression',
    tuning_strategy='finetune',
    finetune_mode="turn_by_turn"
)

logger.info("🔄 Fitting (just setting up regression context, no training)...")
pipeline_inference.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_inference = pipeline_inference.evaluate(X_test, y_test)
results['Inference (ContextTab)'] = metrics_inference
logger.info(f"   R2 Score: {metrics_inference.get('r2', 0):.4f}")

# ============================================================================
# Strategy 2: Meta-Learning Fine-Tuning [TabDPT]
# ============================================================================

logger.info("\\n" + "="*80)
logger.info("STRATEGY 2: Meta-Learning Fine-Tuning")
logger.info("="*80)
logger.info("\\n📝 What it does:")
logger.info("   - Episodic training that mimics in-context learning")
logger.info("   - Recommended: Default for ICL models like TabDPT or ContextTab")

logger.info("\\n🔄 Initializing meta-learning pipeline (TabDPT)...")
pipeline_meta = TabularPipeline(
    model_name='TabDPT',
    task_type='regression',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,
        'learning_rate': 1e-4,
        'finetune_mode': 'turn_by_turn',  # Explicitly set meta-learning mode
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 20,
        'show_progress': True
    }
)

logger.info("🔄 Fitting with meta-learning (episodic training)...")
pipeline_meta.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_meta = pipeline_meta.evaluate(X_test, y_test)
results['Meta-Learning (TabDPT)'] = metrics_meta
logger.info(f"   R2 Score: {metrics_meta.get('r2', 0):.4f}")

# ============================================================================
# Strategy 3: Turn-by-Turn (SFT) Fine-Tuning [TabPFN]
# ============================================================================

logger.info("\\n" + "="*80)
logger.info("STRATEGY 3: Turn-by-Turn Fine-Tuning (SFT)")
logger.info("="*80)
logger.info("\\n📝 What it does:")
logger.info("   - Standard sequential supervised training")
logger.info("   - Distinct 'finetune_mode' required for models like TabPFN in Regression.")

logger.info("\\n🔄 Initializing Turn-by-Turn pipeline (TabPFN)...")
pipeline_sft = TabularPipeline(
    model_name='TabPFN',
    task_type='regression',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'finetune_mode': 'turn_by_turn',  # TabPFN regression explicitly uses 'turn_by_turn'
        'show_progress': True
    }
)

logger.info("🔄 Fitting with Turn-by-Turn supervised training...")
pipeline_sft.fit(X_train, y_train)
logger.info("📊 Evaluating...")
metrics_sft = pipeline_sft.evaluate(X_test, y_test)
results['SFT (TabPFN)'] = metrics_sft
logger.info(f"   R2 Score: {metrics_sft.get('r2', 0):.4f}")


# ============================================================================
# SUMMARY: Strategy Comparison
# ============================================================================

logger.info("\\n" + "="*80)
logger.info("SUMMARY: Fine-Tuning Strategy Comparison (Regression)")
logger.info("="*80)
logger.info("\\n📊 Performance Comparison:")

for strategy, metrics in results.items():
    r2 = metrics.get('r2', 0)
    rmse = metrics.get('rmse', 0)
    logger.info(f"   {strategy:30s} - R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

logger.info("\\n✨ When to Use Each Strategy:")
logger.info("\\n   1. Inference:")
logger.info("      - Pre-trained model context generalizes well continuously")
logger.info("\\n   2. Meta-Learning:")
logger.info("      - TabDPT/ContextTab predicting scaling deviations")
logger.info("\\n   3. SFT / Turn-by-Turn:")
logger.info("      - Required for models enforcing continuous loss adjustments sequentially (TabPFN).")

logger.info("\\n" + "="*80)
logger.info("✅ Example 12 Complete: Regression Fine-Tuning Strategies Demonstration")
logger.info("="*80)
