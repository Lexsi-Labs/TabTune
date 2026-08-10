"""
Example 13: TabFM (Google) — Inference & Fine-Tuning (Classification + Regression)
==================================================================================

This example demonstrates TabTune's integration of Google Research's TabFM
(model name `TabFM`), a zero-shot, hybrid-attention tabular foundation model.
It walks through the full capability surface for both classification and
regression using the same unified API as every other TabTune model.

Classification strategies:
1. Inference     : Zero-shot in-context predictions (no training)
2. Meta-Learning : Episodic fine-tuning (default, matches TabFM's ICL paradigm)
3. SFT           : Supervised single-episode fine-tuning
4. PEFT / LoRA   : Parameter-efficient fine-tuning (LoRA on attention + ICL blocks)

Regression strategies:
1. Inference     : Zero-shot regression
2. Turn-by-turn  : Lightweight episodic regression fine-tuning

Key Learning Points:
- TabFM is a new model entry, used exactly like any other TabTune model
- Inference passes raw mixed-type frames straight to TabFM (best zero-shot accuracy)
- Fine-tuning / PEFT act on TabFM's underlying PyTorch backbone
- The same unified .fit()/.predict()/.evaluate() API works across all modes

Requires: pip install "tabfm[pytorch]"  (weights auto-download from the HF Hub)

Datasets:
- Classification: OpenML 42178 (Telco Customer Churn), sklearn fallback
- Regression    : California Housing (subsampled)
"""

import logging
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from tabtune import TabularPipeline
from tabtune.logger import setup_logger


def set_global_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


set_global_seeds(42)
setup_logger(use_rich=True)
logger = logging.getLogger("tabtune")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=" * 80)
logger.info("EXAMPLE 13: TabFM (Google) — Inference & Fine-Tuning")
logger.info("=" * 80)
logger.info(f"   Device: {DEVICE}")


def load_classification_data():
    logger.info("\n📊 Loading Telco Customer Churn (OpenML ID: 42178)...")
    try:
        import openml

        dataset = openml.datasets.get_dataset(42178, download_data=True, download_qualities=False)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")
        logger.info(f"✅ Loaded {dataset.name}: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        logger.error(f"❌ OpenML load failed ({e}); falling back to breast cancer dataset")
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target

    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def load_regression_data(n_samples=1000):
    logger.info("\n📊 Loading California Housing (subsampled) for regression...")
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    if len(X) > n_samples:
        idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    logger.info(f"✅ Regression data: {X.shape[0]} samples, {X.shape[1]} features")
    return train_test_split(X, y, test_size=0.25, random_state=42)


def run_classification():
    X_train, X_test, y_train, y_test = load_classification_data()
    results = {}

    # --- Strategy 1: Inference (zero-shot) ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 1: Inference (Zero-Shot)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabFM",
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
        model_name="TabFM",
        task_type="classification",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "meta-learning",
            "device": DEVICE,
            "epochs": 3,
            "learning_rate": 2e-6,
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
        model_name="TabFM",
        task_type="classification",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "sft",
            "device": DEVICE,
            "epochs": 3,
            "learning_rate": 2e-6,
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["SFT"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 4: PEFT / LoRA ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION — STRATEGY 4: PEFT / LoRA")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabFM",
        task_type="classification",
        tuning_strategy="peft",
        tuning_params={
            "finetune_mode": "meta-learning",  # PEFT routes through the meta/sft loop
            "device": DEVICE,
            "epochs": 3,
            "learning_rate": 2e-6,
            "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["PEFT"] = pipe.evaluate(X_test, y_test)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: TabFM Classification Strategies")
    logger.info("=" * 80)
    for strategy, metrics in results.items():
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_score", metrics.get("f1", 0))
        logger.info(f"   {strategy:15s} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

    return results


def run_regression():
    X_train, X_test, y_train, y_test = load_regression_data()
    results = {}

    # --- Strategy 1: Inference ---
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION — STRATEGY 1: Inference (Zero-Shot)")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabFM",
        task_type="regression",
        tuning_strategy="inference",
        model_params={"device": DEVICE},
    )
    pipe.fit(X_train, y_train)
    results["Inference"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 2: Turn-by-turn regression fine-tuning ---
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION — STRATEGY 2: Turn-by-Turn Fine-Tuning")
    logger.info("=" * 80)
    pipe = TabularPipeline(
        model_name="TabFM",
        task_type="regression",
        tuning_strategy="finetune",
        tuning_params={
            "finetune_mode": "turn_by_turn",
            "device": DEVICE,
            "epochs": 3,
            "learning_rate": 2e-6,
            "show_progress": True,
        },
    )
    pipe.fit(X_train, y_train)
    results["Turn-by-Turn"] = pipe.evaluate(X_test, y_test)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: TabFM Regression Strategies")
    logger.info("=" * 80)
    for strategy, metrics in results.items():
        rmse = metrics.get("rmse", metrics.get("RMSE", float("nan")))
        r2 = metrics.get("r2", metrics.get("r2_score", float("nan")))
        logger.info(f"   {strategy:15s} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return results


if __name__ == "__main__":
    cls_results = run_classification()
    reg_results = run_regression()

    logger.info("\n" + "=" * 80)
    logger.info("✨ When to Use Each TabFM Strategy")
    logger.info("=" * 80)
    logger.info("   Inference     : quick zero-shot baseline, no training")
    logger.info("   Meta-Learning : preserve in-context generalization (default)")
    logger.info("   SFT           : fast single-task specialization")
    logger.info("   PEFT / LoRA   : memory-efficient adaptation of the backbone")
    logger.info("   Regression    : zero-shot or episodic turn-by-turn fine-tuning")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Example 13 Complete: TabFM Demonstration")
    logger.info("=" * 80)
