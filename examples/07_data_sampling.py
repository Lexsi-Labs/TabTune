"""
Example 7: Data Sampling and Resampling Strategies for Inference
=================================================================

This example demonstrates TabTune's resampling capabilities during inference:
- Different resampling strategies (SMOTE, oversampling, undersampling, KNN)
- Resampling affects which examples are used for inference-based models

Key Learning Points:
- Resampling during inference balances in-context learning examples
- Handle imbalanced datasets with different resampling strategies
- Feature selection for dimensionality reduction
- Compare performance with/without resampling during inference
- Choose appropriate strategy for your data

Dataset: Synthetic highly imbalanced dataset
- Samples: 2000
- Task: Binary classification
- Imbalance: 5% vs 95% (extreme imbalance to clearly show impact)
Note: Resampling happens during inference setup for in-context learning models
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
# DATA LOADING: Highly Imbalanced Synthetic Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 7: Data Sampling and Resampling Strategies for Inference")
logger.info("="*80)
logger.info("\n📊 Creating highly imbalanced synthetic dataset...")
logger.info("   Purpose: Demonstrate resampling impact during inference")
logger.info("   Note: Resampling affects in-context learning examples")
logger.info("   Imbalance: 5% vs 95% (extreme to clearly show differences)")

# Create a highly imbalanced synthetic dataset to demonstrate resampling impact
# Using extreme imbalance (5% vs 95%) to clearly show resampling effects
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, weights=[0.05, 0.95],  # Extreme imbalance: 5% vs 95%
    n_clusters_per_class=1, random_state=42
)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y, name='target')

logger.info(f"✅ Created synthetic highly imbalanced dataset")
logger.info(f"   - Features: {X.shape[1]}")
logger.info(f"   - Samples: {X.shape[0]}")
logger.info(f"   - Target classes: {len(np.unique(y))}")

# Show class distribution (highly imbalanced)
class_dist = y.value_counts().sort_index()
logger.info(f"   - Class distribution: {dict(class_dist)}")
imbalance_ratio = class_dist.min() / class_dist.max()
logger.info(f"   - Imbalance ratio: {imbalance_ratio:.3f} (lower = more imbalanced)")
logger.info(f"   - Class 0: {class_dist[0]} samples ({100*class_dist[0]/len(y):.1f}%)")
logger.info(f"   - Class 1: {class_dist[1]} samples ({100*class_dist[1]/len(y):.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

logger.info(f"\n📊 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
logger.info(f"   Training class distribution: {dict(y_train.value_counts())}")

# ============================================================================
# DEMONSTRATION 1: No Resampling (Baseline)
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 1: Baseline (No Resampling)")
logger.info("="*80)
logger.info("\n📝 Starting with baseline - no resampling")
logger.info("   This shows performance on imbalanced data\n")

pipeline_no_resample = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'none'  # No resampling
    }
)

logger.info("🔄 Training baseline model...")
pipeline_no_resample.fit(X_train, y_train)
metrics_no_resample = pipeline_no_resample.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_no_resample.get('accuracy', 0):.4f}")
logger.info(f"   ✅ F1-Score: {metrics_no_resample.get('f1_score', 0):.4f}")

# ============================================================================
# DEMONSTRATION 2: SMOTE Resampling
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 2: SMOTE (Synthetic Minority Oversampling)")
logger.info("="*80)
logger.info("\n📝 SMOTE creates synthetic samples of minority class:")
logger.info("   - Generates new samples based on existing minority samples")
logger.info("   - Balances classes without losing information")
logger.info("   - Good for moderately imbalanced data\n")

pipeline_smote = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'smote'  # SMOTE resampling
    }
)

logger.info("🔄 Training with SMOTE resampling...")
pipeline_smote.fit(X_train, y_train)
metrics_smote = pipeline_smote.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_smote.get('accuracy', 0):.4f}")
logger.info(f"   ✅ F1-Score: {metrics_smote.get('f1_score', 0):.4f}")

# ============================================================================
# DEMONSTRATION 3: Random Oversampling
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 3: Random Oversampling")
logger.info("="*80)
logger.info("\n📝 Random oversampling duplicates minority class samples:")
logger.info("   - Simple and fast")
logger.info("   - Can lead to overfitting")
logger.info("   - Good for slightly imbalanced data\n")

pipeline_oversample = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'random_over'  # Random oversampling
    }
)

logger.info("🔄 Training with random oversampling...")
pipeline_oversample.fit(X_train, y_train)
metrics_oversample = pipeline_oversample.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_oversample.get('accuracy', 0):.4f}")
logger.info(f"   ✅ F1-Score: {metrics_oversample.get('f1_score', 0):.4f}")

# ============================================================================
# DEMONSTRATION 4: KNN Resampling
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 4: KNN Resampling")
logger.info("="*80)
logger.info("\n📝 KNN resampling uses K-nearest neighbors:")
logger.info("   - Uses local neighborhood information")
logger.info("   - Good for imbalanced datasets")
logger.info("   - Similar to SMOTE but uses KNN approach\n")

pipeline_knn = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'knn'  # KNN resampling
    }
)

logger.info("🔄 Training with KNN resampling...")
pipeline_knn.fit(X_train, y_train)
metrics_knn = pipeline_knn.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_knn.get('accuracy', 0):.4f}")
logger.info(f"   ✅ F1-Score: {metrics_knn.get('f1_score', 0):.4f}")

# ============================================================================
# DEMONSTRATION 5: Feature Selection
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 5: Feature Selection Strategies")
logger.info("="*80)
logger.info("\n📝 Feature selection reduces dimensionality:")
logger.info("   - Removes irrelevant/redundant features")
logger.info("   - Improves training speed")
logger.info("   - Can improve performance by removing noise\n")

# Variance threshold - removes low variance features
logger.info("\n🔧 Feature Selection 1: Variance Threshold")
logger.info("   Removes features with low variance (constant or near-constant)")
pipeline_variance = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'none',
        'feature_selection_strategy': 'variance'  # Variance-based selection
    }
)

logger.info("🔄 Training with variance-based feature selection...")
pipeline_variance.fit(X_train, y_train)
metrics_variance = pipeline_variance.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_variance.get('accuracy', 0):.4f}")

# ANOVA-based feature selection
logger.info("\n🔧 Feature Selection 2: ANOVA F-test")
logger.info("   Selects features with highest F-scores (most discriminative)")
pipeline_anova = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'none',
        'feature_selection_strategy': 'select_k_best_anova'  # ANOVA-based
    }
)

logger.info("🔄 Training with ANOVA-based feature selection...")
pipeline_anova.fit(X_train, y_train)
metrics_anova = pipeline_anova.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_anova.get('accuracy', 0):.4f}")

# ============================================================================
# DEMONSTRATION 6: Combined Strategies
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 6: Combined Resampling + Feature Selection")
logger.info("="*80)
logger.info("\n📝 Combining strategies:")
logger.info("   - SMOTE resampling + Feature selection\n")

pipeline_combined = TabularPipeline(
    model_name='TabDPT',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={
        'resampling_strategy': 'smote',           # Resample for balance
        'feature_selection_strategy': 'variance'  # Select informative features
    }
)

logger.info("🔄 Training with combined strategies...")
pipeline_combined.fit(X_train, y_train)
metrics_combined = pipeline_combined.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {metrics_combined.get('accuracy', 0):.4f}")
logger.info(f"   ✅ F1-Score: {metrics_combined.get('f1_score', 0):.4f}")

# ============================================================================
# SUMMARY: Sampling Strategy Comparison
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Sampling Strategy Comparison")
logger.info("="*80)
logger.info("\n📊 Resampling Strategy Comparison:")
logger.info(f"   No Resampling     - Acc: {metrics_no_resample.get('accuracy', 0):.4f}, "
            f"F1: {metrics_no_resample.get('f1_score', 0):.4f}")
logger.info(f"   SMOTE             - Acc: {metrics_smote.get('accuracy', 0):.4f}, "
            f"F1: {metrics_smote.get('f1_score', 0):.4f}")
logger.info(f"   Random Oversample - Acc: {metrics_oversample.get('accuracy', 0):.4f}, "
            f"F1: {metrics_oversample.get('f1_score', 0):.4f}")
logger.info(f"   KNN Resampling    - Acc: {metrics_knn.get('accuracy', 0):.4f}, "
            f"F1: {metrics_knn.get('f1_score', 0):.4f}")

logger.info("\n📊 Combined Strategy:")
logger.info(f"   SMOTE + Selection - Acc: {metrics_combined.get('accuracy', 0):.4f}, "
            f"F1: {metrics_combined.get('f1_score', 0):.4f}")

logger.info("\n✨ Key Takeaways:")
logger.info("   1. Resampling during inference balances in-context learning examples")
logger.info("   2. SMOTE often works better than simple oversampling")
logger.info("   3. Feature selection can improve speed and performance")
logger.info("   4. Combined strategies can give best results")
logger.info("   5. F1-score is important for imbalanced datasets")
logger.info("   6. Extreme imbalance (5% vs 95%) clearly shows resampling impact")

logger.info("\n💡 When to Use Each Strategy:")
logger.info("   - No Resampling: Balanced dataset or when imbalance is intentional")
logger.info("   - SMOTE: Moderately imbalanced, want synthetic samples")
logger.info("   - Random Oversample: Slightly imbalanced, need simple solution")
logger.info("   - KNN: Imbalanced data with local structure")
logger.info("   - Feature Selection: High-dimensional data, want to reduce noise")
logger.info("   - Note: Resampling affects which examples are used for in-context learning")

logger.info("\n" + "="*80)
logger.info("✅ Example 7 Complete: Data Sampling Demonstration")
logger.info("="*80)
