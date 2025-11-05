"""
Example 1: Unified API Across Multiple Models
=============================================

This example demonstrates TabTune's core contribution: a unified API that works
identically across different tabular foundation models, regardless of their underlying
architecture or complexity.

Key Learning Points:
- The same .fit(), .predict(), .evaluate() interface works for all models
- No need to learn model-specific APIs
- Consistent behavior across TabPFN, TabICL, OrionMSP, and other models

Dataset: OpenML 31 (credit-g) - German Credit Dataset
Industry: Finance
Samples: ~1000
Task: Binary classification (good/bad credit risk)
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
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# Set seeds for reproducible results
set_global_seeds(42)

# Configure rich logging for better output
setup_logger(use_rich=True)
logger = logging.getLogger('tabtune')

# ============================================================================
# DATA LOADING: OpenML German Credit Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 1: Unified API Across Multiple Models")
logger.info("="*80)
logger.info("\n📊 Loading German Credit Dataset (OpenML ID: 31)...")
logger.info("   Industry: Finance")
logger.info("   Task: Predict credit risk (good/bad credit)")

try:
    # Load the German Credit dataset from OpenML
    # This dataset is commonly used in finance for credit scoring
    dataset = openml.datasets.get_dataset(31, download_data=True, download_qualities=False)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Ensure pandas DataFrame/Series format for consistency
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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

logger.info(f"\n📊 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# ============================================================================
# DEMONSTRATION: Unified API Across Different Models
# ============================================================================

logger.info("\n" + "="*80)
logger.info("KEY DEMONSTRATION: Same API, Different Models")
logger.info("="*80)
logger.info("\n🎯 We'll now use the EXACT SAME API (.fit, .predict, .evaluate)")
logger.info("   with three completely different models:\n")

# List of models to demonstrate
models_to_test = [
    ('TabPFN', 'inference'),
    ('TabICL', 'inference'),
    ('OrionMSP', 'inference')
]

results = {}

for model_name, strategy in models_to_test:
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {model_name} with {strategy} strategy")
    logger.info(f"{'='*80}")
    
    # ========================================================================
    # Step 1: Initialize Pipeline
    # ========================================================================
    # Notice: The initialization is identical regardless of model!
    logger.info(f"\n1️⃣  Initializing {model_name} pipeline...")
    pipeline = TabularPipeline(
        model_name=model_name,
        task_type='classification',
        tuning_strategy=strategy,
        tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # ========================================================================
    # Step 2: Fit the Model
    # ========================================================================
    # Notice: Same .fit() method for all models!
    logger.info(f"2️⃣  Fitting {model_name} on training data...")
    pipeline.fit(X_train, y_train)
    logger.info(f"   ✅ {model_name} training complete")
    
    # ========================================================================
    # Step 3: Make Predictions
    # ========================================================================
    # Notice: Same .predict() method for all models!
    logger.info(f"3️⃣  Making predictions with {model_name}...")
    predictions = pipeline.predict(X_test)
    logger.info(f"   ✅ Generated {len(predictions)} predictions")
    
    # ========================================================================
    # Step 4: Evaluate the Model
    # ========================================================================
    # Notice: Same .evaluate() method for all models!
    logger.info(f"4️⃣  Evaluating {model_name} performance...")
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Store results for comparison
    results[model_name] = metrics
    
    logger.info(f"\n✅ {model_name} evaluation complete!")
    logger.info(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    logger.info(f"   F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")

# ============================================================================
# SUMMARY: Unified API Benefits
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Benefits of Unified API")
logger.info("="*80)
logger.info("\n✨ Key Takeaways:")
logger.info("   1. Same initialization pattern for all models")
logger.info("   2. Same .fit() method - no model-specific training loops")
logger.info("   3. Same .predict() method - consistent prediction interface")
logger.info("   4. Same .evaluate() method - uniform metric reporting")
logger.info("\n📊 Performance Comparison:")
for model_name, metrics in results.items():
    logger.info(f"   {model_name:12s} - Accuracy: {metrics.get('accuracy', 0):.4f}, "
                f"F1: {metrics.get('f1_score', 0):.4f}")

logger.info("\n" + "="*80)
logger.info("✅ Example 1 Complete: Unified API Demonstration")
logger.info("="*80)
