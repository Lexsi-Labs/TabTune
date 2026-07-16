<div align="center">
  <a href="https://github.com/Lexsi-Labs/TabTune">
    <img src="https://raw.githubusercontent.com/Lexsi-Labs/TabTune/refs/heads/docs/assets/tabtunelogo.png" alt="TabTune Logo"  width="1000">
  </a>
  <br>
</div>

  
# TabTune - A Unified Library for Inference and Fine-Tuning Tabular Foundation Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](https://github.com/Lexsi-Labs/TabTune)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02802-b31b1b.svg)](https://arxiv.org/abs/2511.02802)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/dSB62Q7A)

A powerful and flexible Python library designed to simplify the **training and fine-tuning** of modern foundation models on tabular data.

Provides a **high-level, scikit-learn-compatible API** that abstracts away the complexities of data preprocessing and model-specific training loops, allowing you to focus on results.

---

## 🚀 Core Features

The library is built on **four main components** that work together seamlessly:

- **`DataProcessor`** -- A smart, model-aware data preparation engine.  
  Automatically handles imputation, scaling, and categorical encoding based on the requirements of the selected model (e.g., integer encoding for TabPFN, text embeddings for ContextTab).

- **`TuningManager`** -- The computational core of the library.  
  Manages the model adaptation process, applying the correct training strategy—whether it's _zero-shot inference_, _episodic fine-tuning_ for ICL models, or _full fine-tuning_ with optional PEFT (Parameter-Efficient Fine-Tuning).

- **`TabularPipeline`** -- The main user-facing object.  
  Provides simple yet efficient functionalities - `.fit()`, `.predict()`, `.evaluate()`, `.save()`, and `.load()` API that chains all components into a seamless, end-to-end experience.

- **`TabularLeaderboard`** -- A leaderboard utility for model comparison.  
  Makes it easy to compare multiple models and strategies on the same dataset splits with automatic ranking and metric reporting.

---

## 🤔 Why TabTune?

Using diverse tabular foundation models often requires writing model-specific boilerplate for data preparation, training, and inference. TabTune solves this by providing:

- **Unified API**: A single, consistent interface (`.fit()`, `.predict()`, `.evaluate()`) for multiple models such as TabPFN, TabPFNv2.6, TabPFNv3, TabICL, TabICLv2, Mitra, ContextTab, TabDPT, OrionMSP, and OrionBix.

- **Automated Preprocessing**: The DataProcessor is model-aware, automatically applying the correct transformations without manual configuration.

- **Flexible Fine-Tuning Strategies**: 
  - **Inference mode** for zero-shot predictions
  - **Meta-learning mode** for episodic fine-tuning (recommended for ICL models)
  - **Supervised Fine-Tuning (SFT)** for task-optimized learning
  - **PEFT mode** for parameter-efficient adaptation using LoRA adapters

- **Easy Model Comparison**: The TabularLeaderboard allows you to benchmark multiple models and strategies to quickly find the best performer.

- **Checkpoint Management**: Automatic saving and loading of fine-tuned model weights with support for resuming training.

---


## 🚀 What's New in this release

-   ✅ **TabFM Integration (Google Research)** -- Full end-to-end support for Google's new zero-shot tabular foundation model `TabFM`, integrated exactly like the other TFMs: unified `.fit()/.predict()/.evaluate()`, model-aware preprocessing, inference, episodic meta-learning fine-tuning, SFT, and **PEFT/LoRA** for both classification and regression. TabFM also plugs into ensembling, distillation, calibration and fairness workflows out of the box.

-   🔬 **"Beyond Accuracy" TabFM Study** -- A reproducible A100-ready study suite (`experiments/tabfm_study/`) auditing TabFM on **calibration** (the *calibration tax of adaptation*), **fairness** (the *synthetic-prior fairness* hypothesis), **ensembling diversity**, and **1000× distillation** — all driven through TabTune's unified harness.

-   ✅ **TabPFN v3 Integration** -- Full support for the latest PriorLabs Model : `TabPFNv3`, with end-to-end inference and fine-tuning (native, meta-learning, SFT, PEFT/LoRA) for both classification and regression. Added as a new model entry alongside the existing TabPFNv2.6 integration.

-   ✅ **Causal Inference Module Integration** -- Full support for treatment effect estimation using tabular foundation models through a unified `CausalAnalysis` API, enabling identification, estimation, and refutation workflows.

-   ✅ **Six Causal Estimators** -- Includes Double Machine Learning (DML), S-Learner, T-Learner, X-Learner, R-Learner, and Causal Forests for robust average and heterogeneous treatment effect estimation.

-   ✅ **Built-in Causal Validation** -- Supports formal causal identification, placebo tests, random common cause checks, subset stability analysis, and sensitivity analysis through an integrated refutation framework.

-   ✅ **Fairness & Compliance Audits** -- Includes proxy attribute detection and counterfactual fairness evaluation with automated reporting for fairness-critical deployments.

-   ✅ **Counterfactual & Heterogeneous Effect Analysis** -- Supports per-row Conditional Average Treatment Effects (CATE), counterfactual prediction, and treatment effect exploration at the individual level.

-   ✅ **CausalLeaderboard Benchmarking** -- Compare multiple `(Estimator × TFM)` combinations using treatment effect stability, confidence intervals, and refutation pass rates.

---

## 📊 Supported Models

| Model | Family / Paradigm | Key Innovation | Supported Strategies |
|-------|------------------|----------------|----------------------|
| **TabPFN-v2** | PFN / ICL | Approximates Bayesian inference on synthetic data | Inference, Meta-Learning FT, SFT, PEFT*, Regression, Regression FT |
| **TabICL** | Scalable ICL | Two-stage column-then-row attention | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionMSP v1.0** | Scalable ICL | Multi-Scale Sparse Attention | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionMSP v1.5** | Scalable ICL | Stabilized prototype refinement | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionBix** | Scalable ICL | Tabular Bi-Axial In-Context Learning | Inference, Meta-Learning FT, SFT, PEFT |
| **Mitra** | Scalable ICL | 2D attention (row & column) | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression-FT |
| **ContextTab** | Semantics-Aware ICL | Modality-specific semantic embeddings | Inference, Full Fine-Tuning, PEFT*, Regression, Regression-FT |
| **TabDPT** | Denoising Transformer | Denoising pre-training | Inference, Meta-Learning FT, SFT, Regression, Regression-FT |
| **LimiX** | Probabilistic / ICL | Likelihood-based mixture modeling; uncertainty-aware | Inference, Regression, Regression-FT |
| **TabPFN-v2.6** | PFN / ICL | Latest PriorLabs release with native finetuning API | Inference, Meta-Learning FT, SFT, Native FT, Regression, Regression FT |
| **TabPFN-v3** | PFN / ICL | Newest PriorLabs Prior-Fitted Network; updated architecture and checkpoints | Inference, Meta-Learning FT, SFT, Native FT, PEFT, Regression, Regression FT |
| **TabICLv2** | Scalable ICL | Improved column-then-row attention | Inference, FT, Regression, Regression FT |
| **TabFM** | Hybrid-Attention ICL (Google) | Alternating row/column attention → row compression → 24-block causal ICL Transformer; pretrained purely on synthetic SCM data | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression FT |
 
*Note: PEFT for ContextTab and TabPFN is experimental; `inference` strategy is fully supported.*

---

## ⚙️ Installation

```bash
git clone https://github.com/Lexsi-Labs/TabTune.git
cd TabTune
pip install -r requirements.txt
pip install -e .
```

> **TabFM (Google):** the `TabFM` model requires the optional `tabfm` package with the
> PyTorch backend. Install it alongside TabTune with `pip install "tabfm[pytorch]"`.
> Pretrained weights (`google/tabfm-1.0.0-pytorch`) are auto-downloaded from the Hugging Face Hub on first use.

---

## 🟦 TabFM (Google) — Quick Start

TabFM is used through the same unified API as every other model:

```python
from tabtune import TabularPipeline

# Zero-shot in-context classification (no training)
pipe = TabularPipeline(model_name="TabFM", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)
print(pipe.evaluate(X_test, y_test))

# Parameter-efficient fine-tuning (LoRA on TabFM's attention + ICL blocks)
pipe = TabularPipeline(
    model_name="TabFM",
    task_type="classification",
    tuning_strategy="peft",
    tuning_params={
        "epochs": 5,
        "learning_rate": 2e-6,
        "finetune_mode": "meta-learning",   # or "sft"
        "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    },
)
pipe.fit(X_train, y_train)

# Regression (episodic turn-by-turn fine-tuning also supported)
reg = TabularPipeline(model_name="TabFM", task_type="regression", tuning_strategy="inference")
reg.fit(X_train, y_train)
print(reg.evaluate(X_test, y_test))
```

---

## ⚡ Quick Start: End-to-End Workflow

Here is a complete example of loading a dataset, fine-tuning a TabPFN model, saving the pipeline, and making predictions.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
from tabtune.TabularPipeline.pipeline import TabularPipeline

# 1. Load a dataset from OpenML
dataset = openml.datasets.get_dataset(42178)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Configure and Initialize the Pipeline
pipeline = TabularPipeline(
    model_name="TabPFN",
    task_type="classification",
    tuning_strategy="inference",  # or 'finetune'
    tuning_params={"device": "cpu"}
)

# 3. Fit the pipeline on the raw training data
pipeline.fit(X_train, y_train)

# 4. Save the fine-tuned pipeline
pipeline.save("fitted_pipeline.joblib")

# 5. Load the pipeline and make predictions on new data
loaded_pipeline = TabularPipeline.load("fitted_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)

# 6. Evaluate the pipeline
metrics = pipeline.evaluate(X_test, y_test)
print(metrics)
```

---

## 🎯 Tuning Strategies

TabTune provides multiple fine-tuning strategies to suit different use cases:

### Inference Mode
Zero-shot predictions without any training. The model uses its pre-trained weights directly on your data.

```python
pipeline = TabularPipeline(
    model_name="TabPFN",
    tuning_strategy="inference"
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Base Fine-Tuning (`base-ft`)
Full parameter fine-tuning. Updates all model weights using task data.

- **Meta-Learning (default for ICL models)**: Episodic training that mimics the in-context learning paradigm
- **SFT (Supervised Fine-Tuning)**: Standard supervised training on batches

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="finetune",  # Defaults to 'base-ft'
    tuning_params={
        "epochs": 5,
        "learning_rate": 1e-5,
        "finetune_mode": "meta-learning"  # or "sft"
    }
)
pipeline.fit(X_train, y_train)
```

### Native Fine-Tuning (TabPFNv2.6 and TabPFNv3)
TabPFNv2.6 and TabPFNv3 expose PriorLabs' `FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor` directly, offering their native advanced fine-tuning pipeline. For TabPFNv3, TabTune pins the v3 checkpoint so native fine-tuning updates the v3 weights.

```python
# Classification
pipeline = TabularPipeline(
    model_name="TabPFNv26",
    task_type="classification",
    tuning_strategy="finetune",
    finetune_mode="native",         # uses FinetunedTabPFNClassifier
    tuning_params={
        "epochs": 30,
        "learning_rate": 1e-5,
        "early_stopping": True,
        "early_stopping_patience": 8,
    }
)
pipeline.fit(X_train, y_train)

# Regression
pipeline = TabularPipeline(
    model_name="TabPFNv26",
    task_type="regression",
    tuning_strategy="finetune",
    finetune_mode="native",         # uses FinetunedTabPFNRegressor
    tuning_params={
        "epochs": 30,
        "learning_rate": 1e-5,
        "early_stopping": True,
    }
)
pipeline.fit(X_train, y_train)

# TabPFNv3 — same native API (V3 checkpoint pinned automatically)
pipeline = TabularPipeline(
    model_name="TabPFNv3",
    task_type="classification",
    tuning_strategy="finetune",
    tuning_params={
        "finetune_mode": "native",  # uses FinetunedTabPFNClassifier (V3-pinned)
        "epochs": 30,
        "learning_rate": 1e-5,
        "early_stopping": True,
        "early_stopping_patience": 8,
    }
)
pipeline.fit(X_train, y_train)
```

### PEFT Mode (Parameter-Efficient Fine-Tuning)
Applies LoRA (Low-Rank Adaptation) adapters to only a subset of parameters, reducing memory and computation.

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        "epochs": 10,
        "learning_rate": 5e-5,
        "peft_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05
        }
    }
)
pipeline.fit(X_train, y_train)
```

**PEFT Support by Model**:
- ✅ **Full Support**: TabICL, OrionMSP, OrionBix, TabDPT, Mitra, TabFM
- ⚠️ **Experimental**: ContextTab and TabPFN (may cause prediction issues; use 'base-ft' instead)

---

## 📊 Evaluation Metrics

When calling `.evaluate()`, TabTune computes the following metrics:

- **Accuracy** -- Fraction of correct predictions
- **Weighted F1 Score** -- Harmonic mean of precision and recall, weighted by class support
- **ROC AUC Score** -- Area under the Receiver Operating Characteristic curve (binary and multi-class supported)
- **Matthews Correlation Coefficient (MCC)** -- Correlation between predicted and actual values
- **Precision & Recall** -- Per-class performance metrics
- **Brier Score** -- Mean squared error of probabilistic predictions

```python
metrics = pipeline.evaluate(X_test, y_test)
print(metrics)
# Output: {'accuracy': 0.92, 'f1_score': 0.89, 'roc_auc_score': 0.95, ...}
```

---

# 📈 Using Regression in TabTune

TabTune now fully supports regression tasks with standardized evaluation
metrics.

## Example: Housing Price Prediction

``` python
from tabtune import TabularPipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = TabularPipeline(
    model_name="OrionMSP",
    task_type="regression",
    tuning_strategy="inference",
    tuning_params={
        "epochs": 5,
        "learning_rate": 2e-5
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(metrics)
```

### Supported Regression Metrics

-   RMSE
-   MAE
-   R² Score


---

## 🔁 Resampling & Context Sampling (Fine-Tuning)

TabTune provides **two complementary mechanisms** for handling data
imbalance and episodic construction:

1.  **Dataset-Level Resampling** (via `DataProcessor`)
2.  **Context / Support-Query Sampling** (for meta-learning models)

Both integrate seamlessly into `TabularPipeline`.

---

## ✅ Supported Resampling Strategies

  Strategy         Description                       Task Support
  ---------------- --------------------------------- ----------------
  `smote`          Synthetic minority oversampling   Classification
  `random_over`    Random oversampling               Classification
  `random_under`   Random undersampling              Classification
  `tomek`          Tomek links cleaning              Classification
  `kmeans`         KMeans-SMOTE hybrid               Classification
  `knn`            KNN-based synthetic sampling      Classification

> Resampling is primarily designed for **imbalanced classification
> tasks**.

---

# Resampling in Action

Resampling is configured through `processor_params` and is applied before training. An example usage is as follows :-

``` python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="inference",
    processor_params={
        "resampling_strategy": "smote"
    },
    tuning_params={
        "epochs": 5,
        "learning_rate": 2e-5
    }
)

pipeline.fit(X_train, y_train)
```
----
### 🧩 Ensembling Strategies

TabTune-Ensemble extends the core library with multi-model ensembling via the `TabularEnsemble` class, combining predictions from multiple TFMs for improved accuracy and robustness.

Six strategies are supported, from simple averaging to competition-grade cascade stacking:

| Strategy | Best For |
|----------|----------|
| `weighted_averaging` | Fast baseline; low-latency production |
| `greedy_selection` | **Recommended default** — general-purpose |
| `stacking` | Diverse model errors; large datasets |
| `temperature_scaled` | Calibrated probabilities; risk-sensitive tasks |
| `cascade_stacking` | Maximum accuracy; competition settings |
| `random_init` | Epistemic uncertainty estimation |

```python
from tabtune.ensemble import TabularEnsemble

ensemble = TabularEnsemble(
    models=[
        {"model_name": "TabPFN",   "tuning_strategy": "inference"},
        {"model_name": "TabICLv2", "tuning_strategy": "inference"},
        {"model_name": "OrionMSP", "tuning_strategy": "inference"},
    ],
    ensemble_strategy="greedy_selection",  
    task_type="classification",
)

ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
metrics = ensemble.evaluate(X_test, y_test)

print(metrics["ensemble"])          
print(ensemble.get_leaderboard())   
```



---

### 🧩 Distillation
TabTune extends the core library with model-agnostic knowledge distillation via the `TabDistiller` class, compressing any TFM teacher into a lightweight student model for fast, deployable inference.

Four student backends are supported, each suited to different deployment constraints:

| Student | `student` value | Best For |
|---------|-----------------|----------|
| LightGBM | `"lgbm"` | **Recommended default** — fast, robust, near-teacher accuracy |
| XGBoost | `"xgb"` | Strong GBDT alternative; marginally slower than LightGBM |
| CatBoost | `"catboost"` | Datasets with high-cardinality categorical features |
| MLP (PyTorch) | `"mlp"` | Neural student requirement; high variance on small datasets |

```python
from tabtune.distillation import TabDistiller

# Single teacher → student
distiller = TabDistiller(
    teachers="TabPFNv26",         # exact model name string required
    student="lgbm",               # or "xgb", "catboost", "mlp"
    task_type="classification",
    temperature=3.0,              # Hinton-style soft-label temperature
    alpha=0.7,                    # KL loss weight; (1 - alpha) = CE weight
    n_folds=5,                    # k-fold cross-prediction (leakage fix for ICL models)
    adaptive_temperature=True,    # per-sample temperature scaling
    confidence_weighting=True,    # weight loss by teacher confidence
)
distiller.fit(X_train, y_train)
predictions = distiller.predict(X_test)
metrics = distiller.compare(X_test, y_test)   # teacher vs student + retention %
print(metrics)

distiller.save("student.pkl")     # serializes student only; teacher stripped
```

Multi-teacher distillation is supported by passing a list — soft labels are averaged across teachers before student training:

```python
# Multi-teacher → student (soft labels averaged)
distiller = TabDistiller(
    teachers=["TabPFNv26", "TabICLv2", "OrionMSPv1.5"],
    student="xgb",
    task_type="classification",
    temperature=4.0,
    alpha=0.6,
)
distiller.fit(X_train, y_train)
predictions = distiller.predict(X_test)
```

Pre-fitted `TabularPipeline` objects can also be passed directly, skipping the teacher fit step:

```python
from tabtune import TabularPipeline
from tabtune.distillation import TabDistiller

pipe = TabularPipeline(model_name="TabICLv2", tuning_strategy="inference")
pipe.fit(X_train, y_train)

distiller = TabDistiller(teachers=[pipe], student="lgbm", task_type="classification")
distiller.fit(X_train, y_train)
```
---

### 🔬 Causal Inference

TabTune extends the core library with causal reasoning through the `CausalAnalysis` class, enabling estimation of treatment effects rather than only making predictions.

The module follows the standard causal workflow:

- **Identify** — Construct causal graphs and identify valid adjustment sets.
- **Estimate** — Compute treatment effects using state-of-the-art causal estimators powered by TFMs.
- **Refute** — Validate estimates using placebo, sensitivity, and robustness checks.

Six estimators are supported:

| Estimator | Best For |
|------------|----------|
| `dml` | **Recommended default** — robust ATE estimation |
| `s_learner` | Simple baseline |
| `t_learner` | Balanced treatment groups |
| `x_learner` | Imbalanced treatment assignment |
| `r_learner` | Advanced heterogeneous effects |
| `causal_forest` | Non-parametric CATE estimation |

```python
from tabtune.causal import CausalAnalysis

causal = CausalAnalysis(
    model_name="TabPFNv26",
    task_type="regression",
    treatment="treated",
    outcome="outcome",
    confounders=["age", "income", "score"],
    estimator="dml"
)

causal.fit(X_train, y_train)

results = causal.evaluate(
    X_test,
    y_test,
    include=("effect", "refutation")
)

print(results["effect"])
```

**Additional capabilities** include:
- Heterogeneous treatment effect (CATE) estimation
- Counterfactual prediction
- Proxy attribute auditing
- Counterfactual fairness evaluation
- Multi-model benchmarking with `CausalLeaderboard`

---
## 🏆 Model Comparison with TabularLeaderboard

The `TabularLeaderboard` makes it easy to compare multiple models and strategies on the same dataset.

```python
from tabtune.TabularLeaderboard.leaderboard import TabularLeaderboard

# 1. Initialize the leaderboard with your data splits
leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)

# 2. Add model configurations to compare
leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='inference',
    model_params={'n_estimators': 16}
)

leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='finetune',
    model_params={'n_estimators': 16},
    tuning_params={'epochs': 5, 'learning_rate': 1e-5, 'finetune_mode': 'meta-learning'}
)

leaderboard.add_model(
    model_name='TabPFN',
    tuning_strategy='inference'
)

# 3. Run the benchmark and display ranked results
leaderboard.run()
```

---

## 🛠️ API Reference

### TabularPipeline Constructor

```python
TabularPipeline(
    model_name: str,
    task_type: str = 'classification',
    tuning_strategy: str = 'inference',
    tuning_params: dict | None = None,
    processor_params: dict | None = None,
    model_params: dict | None = None,
    model_checkpoint_path: str | None = None,
    finetune_mode: str = 'meta-learning'
)
```

#### Key Parameters:

- **`model_name`** (str): The name of the model to use. Supported values: `'TabPFN'`, `'TabPFNv26'`, `'TabPFNv3'`, `'TabICL'`, `'TabICLv2'`, `'ContextTab'`, `'Mitra'`, `'TabDPT'`, `'OrionMSP'`, `'OrionMSPv1.5'`, `'OrionBix'`, `'Limix'`, `'TabFM'`.

- **`task_type`** (str): The type of task — `'classification'` or `'regression'`.

- **`tuning_strategy`** (str): The strategy for model adaptation: `'inference'`, `'finetune'`, or `'peft'`.

- **`finetune_mode`** (str, optional): Controls the fine-tuning algorithm. If `None`, a smart default is chosen per task type (`'turn_by_turn'` for regression, `'meta-learning'` for classification). Supported values per model:
  - `'meta-learning'` — episodic meta-learning (TabICL, TabICLv2, OrionMSP, OrionBix, TabDPT, Mitra, TabPFNv26, TabPFNv3, TabFM)
  - `'sft'` — supervised fine-tuning (TabPFN, TabPFNv26, TabPFNv3, Mitra, TabDPT, TabFM)
  - `'native'` — PriorLabs native finetuner with bar distribution loss, AMP, early stopping (**TabPFNv2.6 and TabPFNv3**, classification and regression)
  - `'turn_by_turn'` / `'tbt'` — episodic turn-by-turn (TabPFN regression, TabPFNv3 regression, Mitra regression, TabDPT regression, ContextTab regression)

- **`tuning_params`** (dict, optional): Parameters for the `TuningManager`:
  - `epochs` (int): Number of training epochs
  - `learning_rate` (float): Learning rate for optimization
  - `batch_size` (int): Batch size for fine-tuning
  - `device` (str): `'cuda'` or `'cpu'`
  - `save_checkpoint_path` (str): Path to save fine-tuned weights
  - `checkpoint_dir` (str): Directory for automatic checkpoint saving
  - `show_progress` (bool): Whether to show progress bars
  - `peft_config` (dict): Configuration for LoRA adapters
  - `early_stopping` (bool): Enable early stopping — **TabPFNv2.6 / TabPFNv3 native mode only**
  - `early_stopping_patience` (int): Patience for early stopping — **TabPFNv2.6 / TabPFNv3 native mode only**
  - `n_estimators_finetune` (int): Ensemble size during fine-tuning — **TabPFNv2.6 / TabPFNv3 native mode only**

- **`processor_params`** (dict, optional): Parameters for the `DataProcessor`:
  - `imputation_strategy` (str): `'mean'`, `'median'`, `'iterative'`, `'knn'`
  - `categorical_encoding` (str): `'onehot'`, `'ordinal'`, `'target'`, `'hashing'`, `'binary'`
  - `scaling_strategy` (str): `'standard'`, `'minmax'`, `'robust'`, `'power_transform'`
  - `resampling_strategy` (str): `'smote'`, `'random_over'`, `'random_under'`, `'tomek'`, `'kmeans'`, `'knn'`
  - `feature_selection_strategy` (str): `'variance'`, `'select_k_best_anova'`, `'select_k_best_chi2'`

- **`model_params`** (dict, optional): Model-specific parameters.

- **`model_checkpoint_path`** (str, optional): Path to a `.pt` file containing pre-trained model weights.

---

## 💾 Checkpoint Management

### Automatic Checkpoint Saving

Fine-tuned models are automatically saved during training:

```python
tuning_params = {
    'save_checkpoint_path': './checkpoints/my_model.pt',
    'checkpoint_dir': './checkpoints'  # Used if save_checkpoint_path is None
}
```

### Manual Checkpoint Loading

```python
# Load pre-trained weights when initializing
pipeline = TabularPipeline(
    model_name="TabPFN",
    model_checkpoint_path="./checkpoints/pretrained.pt"
)
```

### Pipeline Serialization

```python
# Save entire pipeline
pipeline.save("my_pipeline.joblib")

# Load and use
loaded_pipeline = TabularPipeline.load("my_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)
```

---

## 🔧 PEFT/LoRA Configuration

LoRA (Low-Rank Adaptation) adapters can significantly reduce memory usage during fine-tuning.

```python
peft_config = {
    'r': 8,                   # LoRA rank (lower = fewer parameters)
    'lora_alpha': 16,         # Scaling factor for LoRA updates
    'lora_dropout': 0.05,     # Dropout in LoRA modules
    'target_modules': None    # Auto-detect by model (optional override)
}

pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        'epochs': 10,
        'learning_rate': 5e-5,
        'peft_config': peft_config
    }
)
```

**Memory Savings**: PEFT typically reduces memory usage by 60-80% compared to full fine-tuning.

---

## 🏆 Example Notebooks

|Below are 17 Example Notebooks showcasing all the features of the Library in-depth!

| Serial No. | Name | Task Performed | Link To Notebook |
|---|------|------|------|
| 1 | Unified API | Showcasing A Unified API Across Multiple Models |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1KcaSdYRjZnMlb0MLmQ5IlnbPDiuEr1Ld?usp=sharing) |
| 2 |  Automated Model-Aware Preprocessing | The Automated preprocessing system explained |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/12BQ12VJrxtTDslgjnjm26yi3a0PYXqZT?usp=sharing) |
| 3 | Fine-Tuning Strategies | TabTune's four fine-tuning strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1QixfiNCjF1IQV9NooMipPpnH4ETcEQwg?usp=sharing) |
| 4 | Model Comparison | Model Comparison with TabularLeaderboard |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1PZW3iPQOvwh0kroGytMzYTGc6ZVUzuvg?usp=sharing) |
| 5 | Checkpoint Management | Checkpoint Management - Save/Load Pipelines |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1DBTGEPpYLJjU9Aj7lzHoX3JtwaNOC0jn?usp=sharing) |
| 6 | Advanced Usage | PEFT Configuration and Hybrid Strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1V3XGLeKrXSJwavaULMncZiM7uVE8sz0h?usp=sharing) |
| 7 |  Resampling |  Resampling Strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1EHGrrSm7EalVRvzkH1RUHsNSLzmn10lM?usp=sharing) |
| 8 | Regression - 1| Introduction to Regression - Inference |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1lBt0QZWqlwhEg2ul_nVPAeC-w3Are0At) |
| 9 | Regression - 2| Introduction to Regression - Finetune |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1FFuaRBDtJZFAQF-JDIxRAjtgOZ1rmHd1?usp=sharing) |
| 10 | Evaluation Metrics | Evaluation Metrics involved |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/18TxyTyBGAGrIVf6zLjURDChG0vM4V02M?usp=sharing) |
| 11 | Benchmarking | Standard Benchmarking Techniques |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1lcoVMPz_3X5_5taNdB9doTGoN05krNRw?usp=sharing) |
| 12 | TabPFNv2.6 | TabPFNv2.6 — Classification and Regression |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1-5fh2kU9sDidXmm095489f3sxNLssW_M) |
| 13 | TabICLv2 | TabICLv2 — Classification and Regression |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/13lv9Z5QNzaAp_2ArkTXGRKDjDFbKAq3Q) |
| 14 | Ensembling Strategies| TabTune's 6 Ensembling Strategies  |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/19TUTBuJ1VNIbp5hLdU4D64c2_RfwFQC8) |
| 15 | Distillation | With Single and Multi Teachers |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1Fo2zH7jDgYjkYhgI33SyuVgnrhMsdvUH)| 
| 16 | Causal Inference | Estimate Treatment Effect using TFMs |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1CWYo3ynOxw0ysV4iDz_8VNCBjMK3WIyd?usp=sharing)| 

## 🚀 Advanced Usage

### Custom Preprocessing

Override default preprocessing for specific needs:

```python
processor_params = {
    'imputation_strategy': 'iterative',
    'categorical_encoding': 'target',
    'scaling_strategy': 'robust',
    'resampling_strategy': 'smote'
}

pipeline = TabularPipeline(
    model_name="TabICL",
    processor_params=processor_params
)
```

### Hybrid Fine-Tuning

Combine meta-learning with PEFT for optimal results:

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        'epochs': 20,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        'peft_config': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1
        }
    }
)
```

---


## 📖 Documentation

For detailed documentation, API reference, model configurations, and usage examples, please visit: **[Documentation](https://tabtune.lexsi.ai/)**

---

## Acknowledgments

TabTune is built upon the excellent work of the following projects and research teams:


- **[OrionMSP1.0/1.5](https://github.com/Lexsi-Labs/OrionMSP)** - Multi-Scale Sparse Attention for Tabular In-Context Learning
- **[OrionBix](https://github.com/Lexsi-Labs/OrionBix)** - Tabular BiAxial In-Context Learnin
- **[TabPFN](https://github.com/PriorLabs/TabPFN)** - Prior-data Fitted Networks for tabular data
- **[TabICL](https://github.com/soda-inria/tabicl)** - Tabular In-Context Learning with scalable attention
- **[Mitra (Tab2D)](https://github.com/autogluon/autogluon)** - 2D Attention mechanism (Tab2D) for tabular data, included within AutoGluon
- **[ContextTab](https://github.com/SAP-samples/contexttab)** - Semantics-Aware In-Context Learning for Tabular Data
- **[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference)** - Denoising Pre-training Transformer for Tabular Data
- **[AutoGluon](https://github.com/autogluon/autogluon)** - AutoML framework that inspired our unified API design
- **[LimiX](https://github.com/limix-ldm-ai/LimiX)** – Likelihood-based mixture modeling and probabilistic inference framework for structured tabular learning  
- **[TabFM](https://github.com/google-research/tabfm)** – Google Research's zero-shot, hybrid-attention tabular foundation model pretrained on synthetic structural causal models  

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size` in `tuning_params`
- Use `tuning_strategy='peft'` for PEFT mode
- Decrease `n_ensembles` or `context_size` for inference

### PEFT Compatibility Issues
- Some models have experimental PEFT support; use 'base-ft' strategy instead
- Check logs for model-specific warnings

### Device Mismatch
- Ensure `device` parameter matches your hardware (cuda/cpu)
- Use `torch.cuda.is_available()` to check GPU availability

---

## 🗃️ License

This project is released under the MIT License.  
Please cite appropriately if used in academic or production projects.

**Citation:**

```bibtex
@misc{tanna2025tabtuneunifiedlibraryinference,
      title={TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models}, 
      author={Aditya Tanna and Pratinav Seth and Mohamed Bouadi and Utsav Avaiya and Vinay Kumar Sankarapu},
      year={2025},
      eprint={2511.02802},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.02802}, 
}
```

---

## 📫 Join Community / Contribute

- Issues and discussions are welcomed on the [GitHub issue tracker](https://github.com/Lexsi-Labs/TabTune/issues) and [Discord](https://discord.gg/ckVbEJGW) .
- Please see the **Contributing** section for contribution standards, code reviews, and documentation tips.

---
## Contact

<div align="center">
  <a href="https://lexsi.ai/">
    <img src="https://raw.githubusercontent.com/Lexsi-Labs/TabTune/refs/heads/docs/assets/lexsilogowhite.png" width="300">
  </a>
  <br>
  <a href="https://lexsi.ai/">https://www.lexsi.ai</a>
  <br><br>
  Paris 🇫🇷 · Mumbai 🇮🇳 · London 🇬🇧 
  <br><br>
</div>