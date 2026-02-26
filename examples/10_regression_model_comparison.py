"""
Example 10: Regression Model Comparison with TabularLeaderboard
===============================================================

This example demonstrates TabTune's TabularLeaderboard utility applied to Regression,
which makes it easy to compare multiple models and configurations on the same continuous dataset.

Key Learning Points:
- Easy model benchmarking for Regression with TabularLeaderboard
- Compare multiple regression models simultaneously (TabPFN, ContextTab, TabDPT, Mitra, Limix)
- Automatic ranking by regression metrics (R2, RMSE, MAE)
- Side-by-side performance comparison

Dataset: California Housing (Subsampled)
Industry: Real Estate
Samples: ~2000
Task: Regression (predicting house values)
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
# DATA LOADING: California Housing Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 11: Regression Model Comparison with TabularLeaderboard")
logger.info("="*80)
logger.info("\\n📊 Loading California Housing Dataset...")
logger.info("   Task: Predict median house value (Continuous Target)")

try:
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    
    # Use a smaller subset for faster example execution
    X = X.iloc[:2000]
    y = y.iloc[:2000]
    
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

# ============================================================================
# DEMONSTRATION: Using TabularLeaderboard for Regression
# ============================================================================

logger.info("\\n" + "="*80)
logger.info("KEY DEMONSTRATION: Regression Model Comparison")
logger.info("="*80)

# ============================================================================
# Step 1: Initialize the Leaderboard
# ============================================================================

logger.info("\\n1️⃣  Initializing TabularLeaderboard for Regression...")
leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test, task_type='regression')
logger.info("✅ Leaderboard initialized")

# ============================================================================
# Step 2: Add Regression Models to Compare
# ============================================================================

logger.info("\\n2️⃣  Adding regression models to compare...")

# Model 1: TabPFN
logger.info("   ➕ Adding TabPFN (Inference mode)...")
leaderboard.add_model(
    model_name='TabPFN',
    tuning_strategy='inference',
    model_params={}
)

# Model 2: ContextTab
logger.info("   ➕ Adding ContextTab (Inference mode)...")
leaderboard.add_model(
    model_name='ContextTab',
    tuning_strategy='inference',
    model_params={'regression_type': 'l2', 'num_regression_bins': 32}
)

# Model 3: TabDPT
logger.info("   ➕ Adding TabDPT (Inference mode)...")
leaderboard.add_model(
    model_name='TabDPT',
    tuning_strategy='inference',
    model_params={}
)

# Model 4: Mitra
logger.info("   ➕ Adding Mitra (Inference mode)...")
leaderboard.add_model(
    model_name='Mitra',
    tuning_strategy='inference',
    model_params={}
)

# Model 5: Limix
logger.info("   ➕ Adding Limix (Inference mode)...")
leaderboard.add_model(
    model_name='Limix',
    tuning_strategy='inference',
    model_params={}
)

logger.info("\\n✅ All regression models added to leaderboard")

# ============================================================================
# Step 3: Run the Leaderboard
# ============================================================================

logger.info("\\n3️⃣  Running leaderboard (training and evaluating all models)...")
logger.info("   ⏳ This may take a few minutes as it streams through Context sets...\\n")

# Run the leaderboard, ranking by R2 Score (higher is better)
# Other options: 'rmse', 'mae'
results_df = leaderboard.run(rank_by='r2')

# ============================================================================
# Step 4: Display Results
# ============================================================================

logger.info("\\n4️⃣  Leaderboard Results:")
logger.info("="*80)
logger.info("\\n📊 Models ranked by R2 Score (best to worst):\\n")

for idx, row in results_df.iterrows():
    rank = idx + 1
    model_name = row.get('model_name', 'Unknown')
    strategy = row.get('tuning_strategy', 'Unknown')
    r2 = row.get('r2', 0)
    rmse = row.get('rmse', 0)
    mae = row.get('mae', 0)
    
    logger.info(f"   #{rank} {model_name:10s} ({strategy:10s})")
    logger.info(f"      R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

logger.info("\\n" + "="*80)
logger.info("✅ Example 11 Complete: Regression Model Comparison Demonstration")
logger.info("="*80)
