"""
Example 5: Checkpoint Management - Save/Load Pipelines
======================================================

This example demonstrates TabTune's checkpoint management capabilities:
- Saving and loading complete pipelines
- Saving model checkpoints during training
- Resuming training from checkpoints
- Pipeline serialization

Key Learning Points:
- Save entire pipelines (model + preprocessing) as one file
- Automatic checkpoint saving during training
- Load saved pipelines for inference
- Resume training from saved checkpoints

Dataset: OpenML 45547 (Cardiovascular Disease)
Industry: Healthcare
Samples: ~70000
Task: Binary classification (cardiovascular disease prediction)
Note: Large dataset demonstrates checkpoint utility for long training
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
from pathlib import Path

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
# DATA LOADING: OpenML Cardiovascular Disease Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 5: Checkpoint Management - Save/Load Pipelines")
logger.info("="*80)
logger.info("\n📊 Loading Cardiovascular Disease Dataset (OpenML ID: 45547)...")
logger.info("   Industry: Healthcare")
logger.info("   Task: Predict cardiovascular disease")
logger.info("   Note: Large dataset - demonstrates checkpoint benefits")

try:
    # Load the Cardiovascular Disease dataset
    dataset = openml.datasets.get_dataset(45547, download_data=True, download_qualities=False)
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
# DEMONSTRATION 1: Save Complete Pipeline
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 1: Saving Complete Pipeline")
logger.info("="*80)
logger.info("\n📝 Saving a complete pipeline includes:")
logger.info("   - Trained model weights")
logger.info("   - Preprocessing pipeline (fitted)")
logger.info("   - Model configuration")
logger.info("   - Everything needed for inference\n")

# Train a pipeline
logger.info("🔄 Training pipeline...")
pipeline = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 2,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'show_progress': False
    }
)

pipeline.fit(X_train, y_train)
logger.info("✅ Pipeline trained")

# Save the complete pipeline
save_path = "saved_pipeline_example.joblib"
logger.info(f"\n💾 Saving complete pipeline to: {save_path}")
logger.info("   📝 This saves:")
logger.info("      - Model weights")
logger.info("      - Preprocessing transformers")
logger.info("      - Configuration")
logger.info("      - Everything needed to make predictions")

pipeline.save(save_path)
logger.info(f"✅ Pipeline saved to {save_path}")

# ============================================================================
# DEMONSTRATION 2: Load Pipeline and Use for Inference
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 2: Loading Pipeline for Inference")
logger.info("="*80)
logger.info("\n📝 Loading a saved pipeline allows you to:")
logger.info("   - Use the model without retraining")
logger.info("   - Deploy to production")
logger.info("   - Share with others\n")

# Load the saved pipeline
logger.info(f"📂 Loading pipeline from: {save_path}")
loaded_pipeline = TabularPipeline.load(save_path)
logger.info("✅ Pipeline loaded successfully")

# Use loaded pipeline for predictions
logger.info("\n🔄 Making predictions with loaded pipeline...")
predictions = loaded_pipeline.predict(X_test)
logger.info(f"✅ Generated {len(predictions)} predictions")

# Evaluate loaded pipeline
logger.info("\n📊 Evaluating loaded pipeline...")
metrics = loaded_pipeline.evaluate(X_test, y_test)
logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")

# ============================================================================
# DEMONSTRATION 3: Automatic Checkpoint Saving During Training
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 3: Automatic Checkpoint Saving")
logger.info("="*80)
logger.info("\n📝 During training, you can automatically save checkpoints:")
logger.info("   - Save at regular intervals")
logger.info("   - Resume training from last checkpoint")
logger.info("   - Prevent loss of progress if training is interrupted\n")

# Create checkpoint directory
checkpoint_dir = Path("checkpoints_example")
checkpoint_dir.mkdir(exist_ok=True)
logger.info(f"📁 Created checkpoint directory: {checkpoint_dir}")

# Train with checkpoint saving enabled
logger.info("\n🔄 Training with automatic checkpoint saving...")
pipeline_checkpoint = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='finetune',
    tuning_params={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 3,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        'support_size': 100,
        'query_size': 50,
        'n_episodes': 50,
        'show_progress': False,
        # Checkpoint configuration
        'checkpoint_dir': str(checkpoint_dir),  # Directory to save checkpoints
        'save_checkpoint_path': str(checkpoint_dir / "model_checkpoint.pt")  # Specific path
    }
)

pipeline_checkpoint.fit(X_train, y_train)
logger.info("✅ Training complete - checkpoints saved automatically")

# List saved checkpoints
checkpoint_files = list(checkpoint_dir.glob("*.pt"))
if checkpoint_files:
    logger.info(f"\n📋 Saved checkpoints:")
    for ckpt in checkpoint_files:
        logger.info(f"   - {ckpt.name}")

# ============================================================================
# CLEANUP: Remove Example Files
# ============================================================================

logger.info("\n" + "="*80)
logger.info("CLEANUP: Removing Example Files")
logger.info("="*80)

# Remove saved files (optional - comment out to keep them)
if os.path.exists(save_path):
    os.remove(save_path)
    logger.info(f"🗑️  Removed: {save_path}")

# Optionally remove checkpoint directory
# import shutil
# if checkpoint_dir.exists():
#     shutil.rmtree(checkpoint_dir)
#     logger.info(f"🗑️  Removed: {checkpoint_dir}")

# ============================================================================
# SUMMARY: Checkpoint Management Benefits
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Checkpoint Management Benefits")
logger.info("="*80)
logger.info("\n✨ Key Takeaways:")
logger.info("   1. Save Complete Pipelines: Model + preprocessing in one file")
logger.info("   2. Easy Deployment: Load and use saved pipelines anywhere")
logger.info("   3. Automatic Checkpoints: Save during long training runs")
logger.info("   4. Version Control: Save multiple checkpoint versions")

logger.info("\n💡 Use Cases:")
logger.info("   - Production Deployment: Save trained model for serving")
logger.info("   - Experiment Tracking: Save checkpoints at each epoch")
logger.info("   - Model Sharing: Share complete pipelines with team")
logger.info("   - Reproducibility: Load exact model state")

logger.info("\n" + "="*80)
logger.info("✅ Example 5 Complete: Checkpoint Management Demonstration")
logger.info("="*80)
