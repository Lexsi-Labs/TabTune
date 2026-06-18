"""
Example 12: TabPFN v3 — Inference & Fine-Tuning (Classification + Regression)
============================================================================

This example demonstrates TabTune's integration of TabPFN v3 (model name
`TabPFNv3`), the newest PriorLabs Prior-Fitted Network. It walks through the full
capability surface for both classification and regression:

Classification strategies:
1. Inference     : Zero-shot predictions (no training)
2. Meta-Learning : Episodic fine-tuning (default for TabPFN v3)
3. SFT           : Supervised single-episode fine-tuning
4. Native        : PriorLabs FinetunedTabPFNClassifier (V3 checkpoint pinned)
5. PEFT / LoRA   : Parameter-efficient fine-tuning via the meta-learning loop

Regression strategies:
1. Inference     : Zero-shot regression
2. Native        : PriorLabs FinetunedTabPFNRegressor (bar-distribution loss)
3. Turn-by-turn  : Lightweight episodic regression fine-tuning

Key Learning Points:
- TabPFN v3 is a new model entry, used exactly like any other TabTune model
- Native fine-tuning updates the v3 weights (TabTune pins ModelVersion.V3)
- The same unified .fit()/.predict()/.evaluate() API works across all modes
- v3 supports full PEFT/LoRA, unlike the experimental status on the original TabPFN

Datasets:
- Classification: OpenML 42178 (Telco Customer Churn) — Telecom, ~7043 samples
- Regression    : California Housing (subsampled) — Real Estate, ~1000 samples
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=" * 80)
logger.info("EXAMPLE 12: TabPFN v3 — Inference & Fine-Tuning")
logger.info("=" * 80)
logger.info(f"   Device: {DEVICE}")

# ============================================================================
# DATA LOADING: Classification (OpenML Telco Customer Churn)
# ============================================================================

def load_classification_data():
    """Load the Telco Customer Churn dataset, with an sklearn fallback."""
    logger.info("\n📊 Loading Telco Customer Churn (OpenML ID: 42178)...")
    try:
        import openml
        dataset = openml.datasets.get_dataset(
            42178, download_data=True, download_qualities=False
        )
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')
        logger.info(f"✅ Loaded {dataset.name}: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        logger.error(f"❌ OpenML load failed ({e}); falling back to breast cancer dataset")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target

    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# ============================================================================
# DATA LOADING: Regression (California Housing, subsampled)
# ============================================================================

def load_regression_data(n_samples=1000):
    """Load a subsampled California Housing dataset for fast regression demos."""
    logger.info("\n📊 Loading California Housing (subsampled) for regression...")
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    if len(X) > n_samples:
        idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    logger.info(f"✅ Regression data: {X.shape[0]} samples, {X.shape[1]} features")
    return train_test_split(X, y, test_size=0.25, random_state=42)


# ============================================================================
# CLASSIFICATION
# ============================================================================

def run_classification():
    X_train, X_test, y_train, y_test = load_classification_data()
    results = {}

    # --- Strategy 1: Inference (zero-shot) ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 1: Inference (Zero-Shot)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="classification",
        tuning_strategy="inference",
        model_params={"device": DEVICE},
    )
    pipe.fit(X_train, y_train)
    results["Inference"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 2: Meta-Learning fine-tuning (default) ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 2: Meta-Learning Fine-Tuning")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="classification",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "meta-learning",
            "device": DEVICE,
            "epochs": 2,
            "batch_size": 128,
            "learning_rate": 1e-5,
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["Meta-Learning"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 3: SFT ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 3: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="classification",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "sft",
            "device": DEVICE,
            "epochs": 2,
            "learning_rate": 1e-5,
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["SFT"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 4: Native (PriorLabs finetuner, V3-pinned) ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 4: Native Fine-Tuning (V3-pinned)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="classification",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "native",
            "device": DEVICE,
            "epochs": 5,
            "learning_rate": 1e-5,
            "early_stopping": True,
            "early_stopping_patience": 3,
        },
    )
    pipe.fit(X_train, y_train)
    results["Native"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 5: PEFT / LoRA ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 5: PEFT / LoRA")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="classification",
        tuning_strategy="peft",
        tuning_params={
            "finetune_mode": "meta-learning",   # PEFT routes through meta/sft loop
            "device": DEVICE,
            "epochs": 2,
            "batch_size": 128,
            "learning_rate": 5e-5,
            "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["PEFT"] = pipe.evaluate(X_test, y_test)

    # --- Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: TabPFN v3 Classification Strategies")
    logger.info("=" * 80)
    for strategy, metrics in results.items():
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_score", metrics.get("f1", 0))
        logger.info(f"   {strategy:15s} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

    return results


# ============================================================================
# REGRESSION
# ============================================================================

def run_regression():
    X_train, X_test, y_train, y_test = load_regression_data()
    results = {}

    # --- Strategy 1: Inference ---
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION — STRATEGY 1: Inference (Zero-Shot)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="regression",
        tuning_strategy="inference",
        model_params={"device": DEVICE},
    )
    pipe.fit(X_train, y_train)
    results["Inference"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 2: Native regression fine-tuning ---
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION — STRATEGY 2: Native Fine-Tuning (V3-pinned)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="regression",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "native",
            "device": DEVICE,
            "epochs": 5,
            "learning_rate": 1e-5,
            "early_stopping": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["Native"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 3: Turn-by-turn regression fine-tuning ---
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION — STRATEGY 3: Turn-by-Turn Fine-Tuning")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabPFNv3",
        task_type="regression",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "turn_by_turn",
            "device": DEVICE,
            "epochs": 2,
            "batch_size": 128,
            "learning_rate": 1e-5,
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["Turn-by-Turn"] = pipe.evaluate(X_test, y_test)

    # --- Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: TabPFN v3 Regression Strategies")
    logger.info("=" * 80)
    for strategy, metrics in results.items():
        # regression metrics typically include rmse / mae / r2
        rmse = metrics.get("rmse", metrics.get("RMSE", float("nan")))
        r2 = metrics.get("r2", metrics.get("r2_score", float("nan")))
        logger.info(f"   {strategy:15s} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cls_results = run_classification()
    reg_results = run_regression()

    logger.info("\n" + "=" * 80)
    logger.info("✨ When to Use Each TabPFN v3 Strategy")
    logger.info("=" * 80)
    logger.info("   Inference     : quick baseline, zero training, with uncertainty")
    logger.info("   Meta-Learning : preserve in-context generalization (default)")
    logger.info("   SFT           : fast single-task specialization")
    logger.info("   Native        : strongest single-task adaptation + early stopping")
    logger.info("   PEFT / LoRA   : memory-efficient fine-tuning of large models")
    logger.info("   Regression    : native (bar-distribution) or turn-by-turn")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Example 12 Complete: TabPFN v3 Demonstration")
    logger.info("=" * 80)
