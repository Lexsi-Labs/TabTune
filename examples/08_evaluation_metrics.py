"""
Example 8: Advanced Evaluation - Fairness and Calibration
=========================================================

This example demonstrates TabTune's comprehensive evaluation capabilities:
- Calibration metrics (Brier score, calibration curves)
- Fairness evaluation with sensitive features
- Demographic parity and equalized odds
- Comparing standard vs. fairness-aware metrics

Key Learning Points:
- Assess model calibration (probability reliability)
- Evaluate fairness across demographic groups
- Use fairness metrics alongside accuracy
- Ensure responsible AI deployment

Dataset: OpenML 31 (credit-g) - German Credit Dataset
Industry: Finance
Samples: ~1000
Task: Binary classification (credit risk)
Note: Contains demographic features suitable for fairness analysis
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
# DATA LOADING: OpenML German Credit Dataset
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 8: Advanced Evaluation - Fairness and Calibration")
logger.info("="*80)
logger.info("\n📊 Loading German Credit Dataset (OpenML ID: 31)...")
logger.info("   Industry: Finance")
logger.info("   Task: Predict credit risk (good/bad credit)")
logger.info("   Note: Contains demographic features for fairness analysis")

try:
    # Load the German Credit dataset
    dataset = openml.datasets.get_dataset(31, download_data=True, download_qualities=False)
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
    
    # Check for potential sensitive features (common in credit datasets)
    logger.info("\n🔍 Checking for potential sensitive features...")
    potential_sensitive = [col for col in X.columns if any(
        keyword in col.lower() for keyword in ['age', 'sex', 'gender', 'race', 'ethnicity']
    )]
    if potential_sensitive:
        logger.info(f"   Found potential sensitive features: {potential_sensitive}")
    else:
        logger.info("   No obvious sensitive features found - will create synthetic one for demo")
    
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
# PREPARATION: Create Sensitive Feature for Fairness Analysis
# ============================================================================

# If no obvious sensitive feature exists, create a synthetic one based on a feature
# This is for demonstration purposes - in real scenarios, use actual demographic data
logger.info("\n📝 Preparing sensitive feature for fairness analysis...")
if 'age' in X.columns or 'A1' in X.columns:
    # Use age or first feature as basis for sensitive attribute
    feature_name = 'age' if 'age' in X.columns else X.columns[0]
    sensitive_feature = (X_test[feature_name] > X_test[feature_name].median()).astype(int)
    sensitive_feature.name = 'sensitive_group'
    logger.info(f"   Created sensitive feature based on: {feature_name}")
else:
    # Create synthetic sensitive feature
    sensitive_feature = pd.Series(
        np.random.choice([0, 1], size=len(X_test), p=[0.6, 0.4]),
        index=X_test.index,
        name='sensitive_group'
    )
    logger.info("   Created synthetic sensitive feature for demonstration")

logger.info(f"   Sensitive feature distribution: {dict(sensitive_feature.value_counts())}")

# ============================================================================
# DEMONSTRATION 1: Standard Evaluation
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 1: Standard Evaluation Metrics")
logger.info("="*80)
logger.info("\n📝 Standard metrics focus on overall performance:")
logger.info("   - Accuracy, F1-score, ROC-AUC")
logger.info("   - Don't consider fairness or calibration\n")

# Train a model
logger.info("🔄 Training model for evaluation...")
pipeline = TabularPipeline(
    model_name='TabICL',
    task_type='classification',
    tuning_strategy='inference',
    tuning_params={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    processor_params={'resampling_strategy': 'none'}
)

pipeline.fit(X_train, y_train)

# Standard evaluation
logger.info("\n📊 Standard Evaluation Metrics:")
standard_metrics = pipeline.evaluate(X_test, y_test)
logger.info(f"   ✅ Accuracy: {standard_metrics.get('accuracy', 0):.4f}")
logger.info(f"   ✅ F1-Score: {standard_metrics.get('f1_score', 0):.4f}")
logger.info(f"   ✅ ROC-AUC: {standard_metrics.get('roc_auc_score', 0):.4f}")

# ============================================================================
# DEMONSTRATION 2: Calibration Evaluation
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 2: Calibration Metrics")
logger.info("="*80)
logger.info("\n📝 Calibration measures probability reliability:")
logger.info("   - Well-calibrated: predicted probability matches actual probability")
logger.info("   - Important for: risk assessment, decision-making")
logger.info("   - Metrics: Brier score (lower is better), calibration curves\n")

logger.info("📊 Evaluating Model Calibration...")
logger.info("   This measures how well predicted probabilities match actual probabilities")

try:
    calibration_metrics = pipeline.evaluate_calibration(
        X_test, y_test, 
        n_bins=15,  # Number of bins for calibration curve
        output_format='rich'
    )
    
    logger.info("\n✅ Calibration Metrics:")
    logger.info(f"   Brier Score: {calibration_metrics.get('brier_score_loss', 'N/A')}")
    logger.info("   (Lower Brier score = better calibration)")
    
    if 'expected_calibration_error' in calibration_metrics:
        logger.info(f"   Expected Calibration Error: {calibration_metrics.get('expected_calibration_error', 'N/A')}")
    
except Exception as e:
    logger.warning(f"⚠️  Calibration evaluation encountered an issue: {e}")
    logger.info("   (This may be due to dataset characteristics)")

# ============================================================================
# DEMONSTRATION 3: Fairness Evaluation
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 3: Fairness Metrics")
logger.info("="*80)
logger.info("\n📝 Fairness metrics evaluate model behavior across groups:")
logger.info("   - Demographic Parity: Equal positive rate across groups")
logger.info("   - Equalized Odds: Equal TPR and FPR across groups")
logger.info("   - Equal Opportunity: Equal TPR across groups")
logger.info("   - Important for: Responsible AI, avoiding discrimination\n")

logger.info(f"📊 Evaluating Fairness with respect to: {sensitive_feature.name}")
logger.info("   This analyzes if the model behaves differently across groups")

try:
    fairness_metrics = pipeline.evaluate_fairness(
        X_test, y_test, 
        sensitive_features=sensitive_feature,
        output_format='rich'
    )
    
    logger.info("\n✅ Fairness Metrics:")
    
    if 'demographic_parity_difference' in fairness_metrics:
        dp_diff = fairness_metrics['demographic_parity_difference']
        logger.info(f"   Demographic Parity Difference: {dp_diff:.4f}")
        logger.info("      (Should be close to 0 for fairness)")
    
    if 'equalized_odds_difference' in fairness_metrics:
        eo_diff = fairness_metrics['equalized_odds_difference']
        logger.info(f"   Equalized Odds Difference: {eo_diff:.4f}")
        logger.info("      (Should be close to 0 for fairness)")
    
    if 'equal_opportunity_difference' in fairness_metrics:
        eopp_diff = fairness_metrics['equal_opportunity_difference']
        logger.info(f"   Equal Opportunity Difference: {eopp_diff:.4f}")
        logger.info("      (Should be close to 0 for fairness)")
    
    # Show per-group metrics if available
    if 'group_metrics' in fairness_metrics:
        logger.info("\n   Per-Group Performance:")
        for group, metrics in fairness_metrics['group_metrics'].items():
            acc = metrics.get('accuracy', 0)
            logger.info(f"      Group {group}: Accuracy = {acc:.4f}")
    
except Exception as e:
    logger.warning(f"⚠️  Fairness evaluation encountered an issue: {e}")
    logger.info("   (This may be due to dataset characteristics or group sizes)")

# ============================================================================
# DEMONSTRATION 4: Combined Evaluation
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 4: Comprehensive Evaluation Summary")
logger.info("="*80)
logger.info("\n📝 Best Practice: Use all three evaluation types:")
logger.info("   1. Standard metrics: Overall performance")
logger.info("   2. Calibration: Probability reliability")
logger.info("   3. Fairness: Behavior across groups\n")

logger.info("✅ Evaluation Complete")
logger.info("\n📊 Summary:")
logger.info("   Standard Metrics:")
logger.info(f"      Accuracy: {standard_metrics.get('accuracy', 0):.4f}")
logger.info(f"      F1-Score: {standard_metrics.get('f1_score', 0):.4f}")
logger.info(f"      ROC-AUC: {standard_metrics.get('roc_auc_score', 0):.4f}")

logger.info("\n   Calibration:")
try:
    brier = calibration_metrics.get('brier_score_loss', 'N/A')
    if isinstance(brier, float) and not np.isnan(brier):
        logger.info(f"      Brier Score: {brier:.4f}")
    else:
        logger.info(f"      Brier Score: {brier}")
except:
    logger.info("      (See calibration metrics above)")

logger.info("\n   Fairness:")
logger.info("      (See fairness metrics above)")

# ============================================================================
# SUMMARY: Evaluation Best Practices
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Evaluation Best Practices")
logger.info("="*80)
logger.info("\n✨ Key Takeaways:")
logger.info("   1. Standard Metrics: Important for overall performance")
logger.info("   2. Calibration: Critical when probabilities matter")
logger.info("   3. Fairness: Essential for responsible AI deployment")
logger.info("   4. Combined Evaluation: Use all three for comprehensive assessment")

logger.info("\n💡 When to Use Each:")
logger.info("\n   Standard Metrics (.evaluate()):")
logger.info("      - Initial model assessment")
logger.info("      - Comparing different models")
logger.info("      - Overall performance tracking")
logger.info("\n   Calibration Metrics (.evaluate_calibration()):")
logger.info("      - Risk assessment applications")
logger.info("      - When predicted probabilities are used")
logger.info("      - Medical diagnosis, credit scoring")
logger.info("\n   Fairness Metrics (.evaluate_fairness()):")
logger.info("      - Production deployment")
logger.info("      - Models affecting people")
logger.info("      - Compliance with regulations")
logger.info("      - Ethical AI considerations")

logger.info("\n⚠️  Important Notes:")
logger.info("   - Fairness evaluation requires sensitive features")
logger.info("   - Calibration needs sufficient samples per bin")
logger.info("   - Balance performance with fairness considerations")
logger.info("   - Document evaluation methodology")

logger.info("\n" + "="*80)
logger.info("✅ Example 8 Complete: Fairness and Calibration Demonstration")
logger.info("="*80)
