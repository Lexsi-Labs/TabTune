"""
Example 4: Model Comparison with TabularLeaderboard
===================================================

This example demonstrates TabTune's TabularLeaderboard utility, which makes it
easy to compare multiple models and configurations on the same dataset.

Key Learning Points:
- Easy model benchmarking with TabularLeaderboard
- Compare multiple models/configurations simultaneously
- Automatic ranking and metric reporting
- Side-by-side performance comparison

Dataset: OpenML 1461 (bank-marketing) - Bank Marketing Dataset
Industry: Banking/Finance
Samples: ~45211
Task: Binary classification (marketing campaign success)
Note: Large dataset, good for comprehensive comparisons
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
from tabtune import TabularLeaderboard
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
# DATA LOADING: OpenML Bank Marketing Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 4: Model Comparison with TabularLeaderboard")
logger.info("="*80)
logger.info("\n📊 Loading Bank Marketing Dataset (OpenML ID: 1461)...")
logger.info("   Industry: Banking/Finance")
logger.info("   Task: Predict marketing campaign success")
logger.info("   Note: Large dataset for comprehensive model comparison")

try:
    # Load the Bank Marketing dataset from OpenML
    # This is a larger dataset, good for comparing multiple models
    dataset = openml.datasets.get_dataset(1461, download_data=True, download_qualities=False)
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
# DEMONSTRATION: Using TabularLeaderboard
# ============================================================================

logger.info("\n" + "="*80)
logger.info("KEY DEMONSTRATION: Easy Model Comparison with TabularLeaderboard")
logger.info("="*80)
logger.info("\n✨ Benefits:")
logger.info("   - Compare multiple models simultaneously")
logger.info("   - Automatic ranking by metrics")
logger.info("   - Clean, organized results")
logger.info("   - No need to manually track results\n")

# ============================================================================
# Step 1: Initialize the Leaderboard
# ============================================================================

logger.info("1️⃣  Initializing TabularLeaderboard...")
logger.info("   📝 The leaderboard takes train/test splits as input")
logger.info("   📝 All models will be evaluated on the same test set")

leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
logger.info("✅ Leaderboard initialized")

# ============================================================================
# Step 2: Add Models to Compare
# ============================================================================

logger.info("\n2️⃣  Adding models to compare...")
logger.info("   📝 We'll add different models and configurations\n")

# Model 1: TabPFN with inference
logger.info("   ➕ Adding TabPFN (Inference mode)...")
leaderboard.add_model(
    model_name='TabPFN',
    tuning_strategy='inference',  # Zero-shot
    model_params={}  # Use default model parameters
)
logger.info("      ✅ TabPFN (inference) added")

# Model 2: TabICL with inference
logger.info("   ➕ Adding TabICL (Inference mode)...")
leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='inference',
    model_params={}
)
logger.info("      ✅ TabICL (inference) added")

# Model 3: TabICL with fine-tuning (meta-learning)
logger.info("   ➕ Adding TabICL (Fine-tuned with meta-learning)...")
leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='finetune',
    model_params={},
    tuning_params={
        'epochs': 2,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'show_progress': False  # Disable progress bars for cleaner output
    }
)
logger.info("      ✅ TabICL (finetuned) added")

# Model 4: OrionMSP with inference
logger.info("   ➕ Adding OrionMSP (Inference mode)...")
leaderboard.add_model(
    model_name='OrionMSP',
    tuning_strategy='inference',
    model_params={}
)
logger.info("      ✅ OrionMSP (inference) added")

# Model 5: OrionMSP with fine-tuning
logger.info("   ➕ Adding OrionMSP (Fine-tuned with meta-learning)...")
leaderboard.add_model(
    model_name='OrionMSP',
    tuning_strategy='finetune',
    model_params={},
    tuning_params={
        'epochs': 2,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'show_progress': False
    }
)
logger.info("      ✅ OrionMSP (finetuned) added")

logger.info("\n✅ All models added to leaderboard")

# ============================================================================
# Step 3: Run the Leaderboard
# ============================================================================

logger.info("\n3️⃣  Running leaderboard (training and evaluating all models)...")
logger.info("   📝 This will:")
logger.info("      - Train/evaluate each model configuration")
logger.info("      - Collect metrics for each")
logger.info("      - Rank models by specified metric")
logger.info("\n   ⏳ This may take a few minutes...\n")

# Run the leaderboard, ranking by ROC AUC score
# Other options: 'accuracy', 'f1_score', 'precision', 'recall', 'mcc'
results_df = leaderboard.run(rank_by='roc_auc_score')

# ============================================================================
# Step 4: Display Results
# ============================================================================

logger.info("\n4️⃣  Leaderboard Results:")
logger.info("="*80)
logger.info("\n📊 Models ranked by ROC AUC Score (best to worst):\n")

# Display the results in a readable format
for idx, row in results_df.iterrows():
    rank = idx + 1
    model_name = row.get('model_name', 'Unknown')
    strategy = row.get('tuning_strategy', 'Unknown')
    acc = row.get('accuracy', 0)
    roc_auc = row.get('roc_auc_score', 0)
    f1 = row.get('f1_score', 0)
    
    logger.info(f"   #{rank} {model_name:10s} ({strategy:10s})")
    logger.info(f"      Accuracy: {acc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")

logger.info("\n" + "="*80)

# ============================================================================
# SUMMARY: Leaderboard Benefits
# ============================================================================

logger.info("\n✨ Key Takeaways:")
logger.info("   1. Easy Comparison: All models evaluated on same test set")
logger.info("   2. Automatic Ranking: Results sorted by your chosen metric")
logger.info("   3. Comprehensive Metrics: Multiple metrics computed automatically")
logger.info("   4. Flexible: Add any model/configuration combination")
logger.info("   5. Clean Output: Organized results in DataFrame format")

logger.info("\n💡 Use Cases:")
logger.info("   - Model selection: Find best model for your dataset")
logger.info("   - Configuration tuning: Compare different hyperparameters")
logger.info("   - Strategy comparison: Compare inference vs fine-tuning")
logger.info("   - Baseline establishment: Compare against standard models")

logger.info("\n" + "="*80)
logger.info("✅ Example 4 Complete: Model Comparison Demonstration")
logger.info("="*80)
