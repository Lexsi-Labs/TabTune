"""
Example 2: Automated Model-Aware Preprocessing
==============================================

This example demonstrates TabTune's automated preprocessing system that automatically
adapts to different model requirements without manual configuration.

Key Learning Points:
- DataProcessor automatically handles model-specific preprocessing needs
- Different models require different preprocessing (e.g., TabPFN vs ContextTab)
- TabTune abstracts away these differences
- You can still customize preprocessing when needed

Dataset: OpenML 37 (diabetes) - Pima Indians Diabetes Dataset
Industry: Healthcare
Samples: ~768
Task: Binary classification (diabetes diagnosis)
Features: Mixed numerical and categorical, some missing values
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
# DATA LOADING: OpenML Diabetes Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 2: Automated Model-Aware Preprocessing")
logger.info("="*80)
logger.info("\n📊 Loading Pima Indians Diabetes Dataset (OpenML ID: 37)...")
logger.info("   Industry: Healthcare")
logger.info("   Task: Predict diabetes diagnosis")
logger.info("   Challenge: Mixed data types, missing values")

try:
    # Load the Diabetes dataset from OpenML
    # This dataset has mixed numerical features and may have missing values
    dataset = openml.datasets.get_dataset(37, download_data=True, download_qualities=False)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Convert to pandas DataFrame/Series
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='target')
    
    logger.info(f"✅ Successfully loaded dataset: {dataset.name}")
    logger.info(f"   - Features: {X.shape[1]} (mixed numerical/categorical)")
    logger.info(f"   - Samples: {X.shape[0]}")
    logger.info(f"   - Missing values: {X.isnull().sum().sum()}")
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
# DEMONSTRATION: Automatic Preprocessing for Different Models
# ============================================================================

logger.info("\n" + "="*80)
logger.info("KEY DEMONSTRATION: Automatic Model-Aware Preprocessing")
logger.info("="*80)
logger.info("\n🎯 Different models require different preprocessing:")
logger.info("   - TabPFN: Needs integer-encoded categoricals, specific scaling")
logger.info("   - ContextTab: Needs text embeddings, different encoding")
logger.info("   - ICL Models (TabICL/OrionMSP): Need specific padding, encoding")
logger.info("\n✨ TabTune handles all of this automatically!\n")

# ============================================================================
# Example 1: TabPFN - Automatic Preprocessing
# ============================================================================

logger.info("\n" + "="*80)
logger.info("MODEL 1: TabPFN - Automatic Preprocessing")
logger.info("="*80)

logger.info("\n📝 TabPFN requires:")
logger.info("   - Categorical features: Integer encoding")
logger.info("   - Numerical features: Standard scaling")
logger.info("   - Feature padding to specific dimensions")

# Initialize TabPFN - preprocessing is automatic!
pipeline_tabpfn = TabularPipeline(
    model_name='TabPFN',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

logger.info("\n✅ Initialized TabPFN pipeline")
logger.info("   ⚙️  DataProcessor will automatically:")
logger.info("      - Detect categorical vs numerical features")
logger.info("      - Apply integer encoding for categoricals")
logger.info("      - Apply appropriate scaling for numericals")
logger.info("      - Handle missing values")
logger.info("      - Pad features to TabPFN's required dimensions")

# Fit - preprocessing happens automatically during fit()
logger.info("\n🔄 Fitting TabPFN (preprocessing happens automatically)...")
pipeline_tabpfn.fit(X_train, y_train)
logger.info("✅ Fit complete - all preprocessing was automatic!")

# Evaluate
logger.info("\n📊 Evaluating TabPFN...")
metrics_tabpfn = pipeline_tabpfn.evaluate(X_test, y_test)
logger.info(f"   Accuracy: {metrics_tabpfn.get('accuracy', 0):.4f}")

# ============================================================================
# Example 2: TabICL - Different Automatic Preprocessing
# ============================================================================

logger.info("\n" + "="*80)
logger.info("MODEL 2: TabICL - Different Automatic Preprocessing")
logger.info("="*80)

logger.info("\n📝 TabICL requires:")
logger.info("   - Different categorical encoding (compatible with ICL)")
logger.info("   - Feature padding for in-context learning")
logger.info("   - Support/query split formatting")

# Initialize TabICL - different automatic preprocessing!
pipeline_tabicl = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

logger.info("\n✅ Initialized TabICL pipeline")
logger.info("   ⚙️  DataProcessor will automatically:")
logger.info("      - Apply ICL-compatible preprocessing")
logger.info("      - Handle categorical encoding differently than TabPFN")
logger.info("      - Format data for in-context learning")
logger.info("      - Handle missing values")

# Fit - different preprocessing happens automatically
logger.info("\n🔄 Fitting TabICL (different automatic preprocessing)...")
pipeline_tabicl.fit(X_train, y_train)
logger.info("✅ Fit complete - ICL-specific preprocessing was automatic!")

# Evaluate
logger.info("\n📊 Evaluating TabICL...")
metrics_tabicl = pipeline_tabicl.evaluate(X_test, y_test)
logger.info(f"   Accuracy: {metrics_tabicl.get('accuracy', 0):.4f}")

# ============================================================================
# Example 3: Custom Preprocessing Parameters
# ============================================================================

logger.info("\n" + "="*80)
logger.info("MODEL 3: Customizing Preprocessing (When Needed)")
logger.info("="*80)

logger.info("\n📝 Sometimes you want to customize preprocessing:")
logger.info("   - Different imputation strategy")
logger.info("   - Different scaling method")
logger.info("   - Feature selection")

# Initialize with custom preprocessing parameters
pipeline_custom = TabularPipeline(
    model_name='OrionMSP',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    # Custom preprocessing parameters
    processor_params={
        'imputation_strategy': 'median',  # Use median instead of default
        'scaling_strategy': 'robust',     # Use robust scaling
        'resampling_strategy': 'none'     # Disable resampling
    }
)

logger.info("\n✅ Initialized OrionMSP with custom preprocessing")
logger.info("   ⚙️  Custom parameters override defaults:")
logger.info("      - Imputation: median (instead of default)")
logger.info("      - Scaling: robust (instead of default)")
logger.info("      - Resampling: disabled")

# Fit with custom preprocessing
logger.info("\n🔄 Fitting with custom preprocessing...")
pipeline_custom.fit(X_train, y_train)
logger.info("✅ Fit complete with custom preprocessing!")

# Evaluate
logger.info("\n📊 Evaluating with custom preprocessing...")
metrics_custom = pipeline_custom.evaluate(X_test, y_test)
logger.info(f"   Accuracy: {metrics_custom.get('accuracy', 0):.4f}")

# ============================================================================
# SUMMARY: Automated Preprocessing Benefits
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Benefits of Automated Preprocessing")
logger.info("="*80)
logger.info("\n✨ Key Takeaways:")
logger.info("   1. No manual preprocessing code needed - TabTune handles it")
logger.info("   2. Model-aware: Different models get appropriate preprocessing")
logger.info("   3. Customizable: Override defaults when needed via processor_params")
logger.info("   4. Consistent: Same preprocessing pipeline for train/test")
logger.info("\n📊 Performance Comparison:")
logger.info(f"   TabPFN (auto)    - Accuracy: {metrics_tabpfn.get('accuracy', 0):.4f}")
logger.info(f"   TabICL (auto)    - Accuracy: {metrics_tabicl.get('accuracy', 0):.4f}")
logger.info(f"   OrionMSP (custom)- Accuracy: {metrics_custom.get('accuracy', 0):.4f}")

logger.info("\n" + "="*80)
logger.info("✅ Example 2 Complete: Automated Preprocessing Demonstration")
logger.info("="*80)
