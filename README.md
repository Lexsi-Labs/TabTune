<div align="center">
  <a href="https://github.com/Lexsi-Labs/TabTune">
    <img src="https://raw.githubusercontent.com/Lexsi-Labs/TabTune/refs/heads/docs/assets/tabtunelogo.png" alt="TabTune Logo"  width="1000">
  </a>
  <br>
</div>

  
# TabTune - A Unified Library for Inference and Fine-Tuning Tabular and Time Series Foundation Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](https://github.com/Lexsi-Labs/TabTune)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02802-b31b1b.svg)](https://arxiv.org/abs/2511.02802)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/dSB62Q7A)

A Python library for the **inference and fine-tuning** of foundation models on tabular data and, since 0.2.1, on time series.

It provides a **high-level, scikit-learn-compatible API** that takes care of data preprocessing and model-specific training loops. `TabularPipeline` covers 21 tabular foundation models; `TimeSeriesPipeline` covers nine time series foundation models, every tabular model as a forecaster, and five statistical baselines, with forecasting, anomaly detection, imputation and embeddings behind one call.

---

## 🚀 Core Features

The library is built on **five main components**:

- **`DataProcessor`** -- A smart, model-aware data preparation engine.  
  Automatically handles imputation, scaling, and categorical encoding based on the requirements of the selected model (e.g., integer encoding for TabPFN, text embeddings for ContextTab).

- **`TuningManager`** -- The computational core of the library.  
  Manages the model adaptation process, applying the correct training strategy—whether it's _zero-shot inference_, _episodic fine-tuning_ for ICL models, or _full fine-tuning_ with optional PEFT (Parameter-Efficient Fine-Tuning).

- **`TabularPipeline`** -- The main user-facing object.  
  Provides simple yet efficient functionalities - `.fit()`, `.predict()`, `.evaluate()`, `.save()`, and `.load()` API that chains all components into a seamless, end-to-end experience.

- **`TabularLeaderboard`** -- A leaderboard utility for model comparison.  
  Makes it easy to compare multiple models and strategies on the same dataset splits with automatic ranking and metric reporting.

- **`TimeSeriesPipeline`** -- The same pattern for time series.  
  `fit`, `predict`, `evaluate`, `backtest`, `calibrate`, `save` and `load` over nine vendored time series foundation models, every tabular model as a forecaster and five baselines, with `TimeSeriesLeaderboard`, `TimeSeriesEnsemble` and `TimeSeriesBenchmark` beside it.

---

## 🤔 Why TabTune?

Using diverse tabular foundation models often requires writing model-specific boilerplate for data preparation, training, and inference. TabTune solves this by providing:

- **Unified API**: A single, consistent interface (`.fit()`, `.predict()`, `.evaluate()`) across **21 models** — TabPFN, TabPFNv2.6, TabPFNv3, TabPFNv3.5, TabPFNv3.5-fast, TabICL, TabICLv2, OrionMSP, OrionMSPv1.5, OrionBix, Mitra, Mitra v2, ContextTab, TabDPT, LimiX, TabFM, xRFM, iLTM, EXAONE Tabular, Causilo and TabLDM.

- **Automated Preprocessing**: The DataProcessor is model-aware, automatically applying the correct transformations without manual configuration.

- **Flexible Fine-Tuning Strategies**: 
  - **Inference mode** for zero-shot predictions
  - **Meta-learning mode** for episodic fine-tuning (recommended for ICL models)
  - **Supervised Fine-Tuning (SFT)** for task-optimized learning
  - **PEFT mode** for parameter-efficient adaptation using LoRA adapters

- **Easy Model Comparison**: The TabularLeaderboard allows you to benchmark multiple models and strategies to quickly find the best performer.

- **Checkpoint Management**: Automatic saving and loading of fine-tuned model weights with support for resuming training.

- **Deployment Awareness**: A torch-free model registry records each checkpoint's capability envelope and weight license, so a class-count mismatch or a research-only license fails in milliseconds instead of after a multi-gigabyte download.

- **Time Series Too**: One `TimeSeriesPipeline` for Chronos, Chronos-Bolt, Chronos-2, TimesFM 3, Toto 1 and 2, TiRex, TiRex-2 and Time-MoE, with the model code vendored so no model package is installed. Zero-shot, full fine-tuning and LoRA for all nine; conformal intervals, covariates, multivariate targets, anomaly detection, imputation and embeddings from the same fitted model.

- **One Logger**: `setup_logger` configures Rich, plain or JSON console output and rotating log files once, and every pipeline, leaderboard and benchmark reports through it.

---


## 🚀 What's New in this release

-   ✅ **Time series on `TimeSeriesPipeline`** - Time series models use the same pipeline pattern as tabular ones: `TimeSeriesPipeline(model_name, task_type, tuning_strategy, tuning_params, model_params, forecast_params)` with `fit`, `predict`, `evaluate`, `save` and `load`. Tasks are forecasting, anomaly detection, imputation and embeddings; strategies are zero-shot `inference`, `finetune` and `peft` (LoRA). The pipeline adds conformal `calibrate()`, rolling-origin `backtest()`, context-length tuning and automatic model selection, plus `TimeSeriesLeaderboard`, `TimeSeriesEnsemble` and `TimeSeriesBenchmark`. The `tabtune.tsfm` facade is removed. 

- ✅ **Five new tabular models** - **TabPFN v3.5** and **TabPFN v3.5-fast** (Prior Labs; one multitask checkpoint carries both the classification and the regression head), **Causilo** (Nums AI; column attention, row mixing and a pooled prediction stage), **TabLDM** (Xiaomi; dual-stream feature grouping with a sparse mixture of experts) and **Mitra v2** (AWS; new weights on Mitra's 2D row-and-column attention). All five are in the registry with their envelopes and weight licenses. Their preprocessing needs `skrub`, which is now a core dependency.

-   ✅ **Tabular models as forecasters** - Every tabular model with a regression head forecasts as `TabularTS-<name>`, with TabPFN-TS's time features or a pooled lag design. `TabPFN-TS` is re-implemented on TabTune's vendored TabPFN, and `TabularTS-GBM` is a weight-free gradient-boosting baseline. 

-   ✅ **Time series histories as tabular features** - `tabtune.bridge.SeriesFeaturizer` gives each table row leak-safe features of its entity's history (summary statistics and model embeddings, as of the row's own timestamp), for any `TabularPipeline` model.

---

## 📊 Supported Models

**21 models across seven architectural families.** The `Commercial` column reflects the
**weight** license, which is what decides whether you can ship.

| Model | Family / Paradigm | Key Innovation | Supported Strategies |
|-------|------------------|----------------|----------------------|
| **TabPFN-v2** | PFN / ICL | Approximates Bayesian inference on synthetic data | Inference, Meta-Learning FT, SFT, PEFT*, Regression, Regression FT |
| **TabPFN-v2.6** | PFN / ICL | PriorLabs release with native finetuning API | Inference, Meta-Learning FT, SFT, Native FT, Regression, Regression FT | 
| **TabPFN-v3** | PFN / ICL | Column embedding → row aggregation → ICL over compressed rows; 160 classes, 20k features | Inference, Meta-Learning FT, SFT, Native FT, PEFT, Regression, Regression FT |
| **TabPFN-v3.5** | PFN / ICL | New architecture; one multitask checkpoint with both classification and regression heads; native text and datetime columns | Inference, Meta-Learning FT, SFT, Native FT, PEFT, Regression, Regression FT |
| **TabPFN-v3.5-fast** | PFN / ICL | Separately trained lower-cost v3.5 checkpoint on the same runtime | Inference, Meta-Learning FT, SFT, Native FT, PEFT, Regression, Regression FT |
| **TabICL** | Scalable ICL | Two-stage column-then-row attention | Inference, Meta-Learning FT, SFT, PEFT | 
| **TabICLv2** | Scalable ICL | QASSMax normalisation + native quantile regression head | Inference, FT, Regression, Regression FT | 
| **OrionMSP v1.0** | Scalable ICL | Multi-Scale Sparse Attention | Inference, Meta-Learning FT, SFT, PEFT | 
| **OrionMSP v1.5** | Scalable ICL | Stabilized prototype refinement | Inference, Meta-Learning FT, SFT, PEFT | 
| **OrionBix** | Scalable ICL | Tabular Bi-Axial In-Context Learning | Inference, Meta-Learning FT, SFT, PEFT | 
| **Mitra** | Scalable ICL | 2D attention (row & column), mixed synthetic priors | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression-FT | 
| **Mitra v2** | Scalable ICL | Second-generation Mitra weights on the same vendored Tab2D architecture | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression-FT | 
| **ContextTab** | Semantics-Aware ICL | Modality-specific semantic embeddings; first-class text and datetime | Inference, Full Fine-Tuning, PEFT*, Regression, Regression-FT | 
| **TabDPT** | Denoising Transformer | Denoising pre-training + retrieval-based context | Inference, Meta-Learning FT, SFT, Regression, Regression-FT | 
| **LimiX** | Probabilistic / ICL | Likelihood-based mixture modeling; uncertainty-aware | Inference, Regression, Regression-FT | 
| **TabFM** | Hybrid-Attention ICL (Google) | Alternating row/column attention → CLS row compression → causal ICL Transformer | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression FT | 
| **xRFM** | Kernel / Feature Learning | Recursive Feature Machine: AGOP feature learning, tree-partitioned EigenPro. **No pretrained weights — trains from scratch** | Inference, Refit, Refine, PEFT†, Regression | 
| **iLTM** | Hypernetwork | Hypernetwork generates MLP ensembles from dataset embeddings; GBDT tree embeddings + retrieval | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression FT | 
| **EXAONE Tabular** | Cross-Axis ICL (LG AI Research) | Cross-axis Summary Transformer (CAST); ~21M params, 8-member ensemble, ECOC for >10 classes | Inference, Meta-Learning FT, SFT, PEFT‡, Regression‡‡ | 
| **Causilo** | ICL (Nums AI) | Column attention → row mixing → pooled prediction over a pretrained context; ECOC for >10 classes | Inference, Meta-Learning FT, PEFT, Regression, Regression FT, Regression PEFT | 
| **TabLDM** | ICL (Xiaomi) | Dual-stream feature grouping, attention residual and sparse mixture of experts; pretrained only on SCM synthetic data | Inference, Meta-Learning FT, PEFT, Regression, Regression FT, Regression PEFT | 



\* PEFT is experimental for ContextTab and TabPFN; `inference` is fully supported.
† xRFM's `peft` is low-rank adaptation of the learned **M** matrix, not LoRA over linear layers.
‡ EXAONE's projections are raw `nn.Parameter` tensors applied through `F.linear`, so the LoRA injector wraps zero adapters and the run proceeds as a full fine-tune.


---

## 🧭 Model Registry: Envelopes and Licensing

The registry answers two questions without downloading a multi-gigabyte checkpoint:
*will this model accept my data*, and *can I actually deploy this*.

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="TabFM",
    task_type="classification",
    envelope_mode="error",        # 'error' | 'warn' (default) | 'ignore'
    license_mode="commercial",    # 'research' (default) | 'commercial' | 'ignore'
)

```


---

## ⚙️ Installation

```bash
git clone https://github.com/Lexsi-Labs/TabTune.git
cd TabTune
pip install -r requirements.txt
pip install -e .
```

**Optional extras**

```bash
pip install "tabtune[distillation]"   # LightGBM students for tabtune.distillation
pip install "tabtune[serving]"        # ONNX export of distilled students
pip install "tabtune[interactive]"    # rich notebook display helpers
pip install "tabtune[colab]"          # pin core packages to Colab's versions
pip install "tabtune[docs]"           # mkdocs toolchain
pip install "tabtune[dev]"            # pytest, ruff, black, pre-commit
pip install "tabtune[toto2]"          # the Toto 2 time series model (Python 3.12+)
```


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

## 🟪 EXAONE Tabular (LG AI Research) — Quick Start

EXAONE Tabular is an in-context learner built on the **Cross-axis Summary Transformer
(CAST)**: 3 summary tokens per row pool that row's columns, 32 summary tokens per feature
group pool that column across rows, and the two axes alternate for 12 blocks under
SSMax-normalised attention. At ~21M parameters it is the smallest bundled foundation
model, which is why the released default is an 8-member ensemble.

```python
from tabtune import TabularPipeline

# Zero-shot in-context classification
pipeline = TabularPipeline(
    model_name="EXAONE",                # aliases: EXAONETabular, exaone-tabular, ...
    task_type="classification",
    tuning_strategy="inference",
    model_params={"n_ensemble": 8},
)
pipeline.fit(X_train, y_train)
print(pipeline.evaluate(X_test, y_test))

# Episodic meta-learning fine-tuning (the default finetune mode)
pipeline = TabularPipeline(
    model_name="EXAONE",
    task_type="classification",
    tuning_strategy="finetune",
    finetune_mode="meta-learning",
    tuning_params={"epochs": 2, "learning_rate": 1e-5},
)
pipeline.fit(X_train, y_train)
```

### Three limits, and not one of them is an error

| Limit | Value | What happens when you exceed it |
|---|---:|---|
| Support rows | `100,000` | Random subsample down to the limit |
| Features | `100` | Attention-based selection of the top 100 |
| Classes | `10` | ECOC decomposition, one full ensemble forward per codebook row |

`max_classes` is deliberately left `None` in the registry. It is a **hard** constraint that
raises even under `envelope_mode='warn'`, so declaring the 10-class head capacity would
reject datasets this model handles by design via ECOC.

```python
from tabtune.registry import check_envelope, get_model_spec

for v in check_envelope("EXAONE", n_rows=250_000, n_features=120, n_classes=14):
    print(f"[{v.severity}] {v.message}")     # rows and features warn; 14 classes is fine

print(get_model_spec("EXAONE").envelope.max_classes)   # None
```



---

## ⏱️ Time Series

`TimeSeriesPipeline` mirrors `TabularPipeline`: name a model, a task and a tuning strategy, call `fit`, then `predict`. The data is a long frame with one row per series and timestamp, and a `TimeSeriesSchema` names the role of each column.

```python
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

df = make_panel(n_series=5, length=400, freq="h")                 # or your own long frame
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)

pipe = TimeSeriesPipeline("ChronosBolt", forecast_params={"prediction_length": 24})
forecast = pipe.fit(history, schema).predict()
forecast.to_pandas()          # item_id, timestamp, target, point, 0.1, 0.5, 0.9
pipe.evaluate(actual)         # MAE, RMSE, sMAPE, MASE, pinball, WQL, coverage
pipe.backtest(windows=3)      # rolling-origin scores
pipe.calibrate(windows=4)     # conformal intervals, for any model
```

Swap `"ChronosBolt"` for `"TiRex"`, `"TabPFN-TS"` or `"SeasonalNaive"` and nothing else changes.

### Models

**Nine time series foundation models, vendored.** No model package is installed: the upstream code lives in `tabtune/models/` with its licence files and is imported only when a pipeline is fitted. All nine run zero-shot, fine-tune fully or with LoRA, and serve embeddings.

| Model | Default checkpoint | Multivariate | Covariates | NaN in history | Max context | Weight licence |
|---|---|---|---|---|---|---|
| **Chronos** | `amazon/chronos-t5-small` | no | no | yes | 512 | Apache-2.0 |
| **Chronos-Bolt** | `amazon/chronos-bolt-small` | no | no | yes | 2,048 | Apache-2.0 |
| **Chronos-2** | `amazon/chronos-2` | yes | past + known, incl. categorical | yes | 8,192 | Apache-2.0 |
| **TimesFM 3** | `google/timesfm-3.0-pytorch` | yes | past + known, numeric | yes | 15,360 | non-commercial |
| **Toto 1** | `Datadog/Toto-Open-Base-1.0` | yes | past + known, numeric | yes | 4,096 | Apache-2.0 |
| **Toto 2** | `Datadog/Toto-2.0-22m` | yes | no | yes | 4,096 | Apache-2.0 (`tabtune[toto2]`) |
| **TiRex** | `NX-AI/TiRex` | no | no | yes | 2,048 | NXAI Community (conditional) |
| **TiRex-2** | `NX-AI/TiRex-2` | yes | past + known, numeric | yes | from checkpoint | unverified |
| **Time-MoE** | `Maple728/TimeMoE-50M` | no | no | no | 4,096 | unverified |

**Tabular foundation models as forecasters.** `TabPFN-TS` re-implements the TabPFN-TS recipe on TabTune's vendored TabPFN, and every tabular model with a regression head is registered as `TabularTS-<name>` (17 today), with TabPFN-TS's time features or a pooled lag design (`model_params={"features": "time" | "lags"}`). `TabularTS-GBM` is a weight-free gradient-boosting forecaster, and any scikit-learn regressor drops in through `SklearnBackend`. Experimental

**Statistical baselines.** `SeasonalNaive`, `Naive`, `Mean`, `Drift` and `WindowAverage`, so every leaderboard has a floor.

`tabtune timeseries list-models` prints the whole catalog with licences; `list_time_series_models(task=..., strategy=..., commercial_ok=True)` filters it in Python.

### What the pipeline guarantees

- **Checks before downloads.** The registry confirms the model supports the task and strategy, `license_mode="commercial"` refuses non-commercial weights, and `envelope_mode` warns (or errors) when the horizon exceeds what the checkpoint was documented for.
- **Quantiles that mean what they say.** You ask for `quantile_levels`; each adapter maps them onto what its model produces (interpolate inside the grid, clamp outside it with a warning, or refuse a level the model never produced). Quantiles are sorted before they reach you, and `forecast.metadata["point_forecast"]` says whether the point is a mean or a median.
- **Covariates a model cannot read are an error**, with the models that can read them named. Known covariates travel in a separate `future_df`.
- **No configuration is scored on data it was fitted on.** `backtest`, `TimeSeriesLeaderboard` and `TimeSeriesEnsemble` train without the scored windows.

### Four tasks from one fitted model

```python
TimeSeriesPipeline("ChronosBolt", task_type="anomaly_detection", task_params={"method": "interval"}).fit(df, schema).predict()
TimeSeriesPipeline("ChronosBolt", task_type="imputation").fit(df_with_gaps, schema).predict()
TimeSeriesPipeline("TiRex", task_type="embedding").fit(df, schema).predict().embeddings
```

### Compare, combine, select

```python
from tabtune.TimeSeries import TimeSeriesLeaderboard, TimeSeriesEnsemble

board = TimeSeriesLeaderboard(history, schema, forecast_params={"prediction_length": 24}, windows=2)
board.add_models(["SeasonalNaive", "Chronos2", "TiRex", "TabularTS-GBM"])
board.add_model("TiRex", tuning_strategy="peft", tuning_params={"epochs": 2}, label="TiRex LoRA")
board.run()

ensemble = TimeSeriesEnsemble(["ChronosBolt", "TiRex", "SeasonalNaive"], forecast_params={"prediction_length": 24})
ensemble.fit(history, schema).predict()

best = TimeSeriesPipeline.select(history, schema, {"prediction_length": 24},
                                 candidates=["Chronos2", "TiRex", "SeasonalNaive"], license_mode="commercial")
```

### Covariates and multivariate targets

```python
schema = TimeSeriesSchema(target=["load", "price"], item_id="site",
                          past_covariates=["temperature"], known_covariates=["promo"])
pipe = TimeSeriesPipeline("Chronos2", forecast_params={"prediction_length": 24})
pipe.fit(history, schema, future_df=future_promo)      # known covariates over the horizon
pipe.evaluate(actual)["per_target"]
```

### Series histories as tabular features

```python
from tabtune.bridge import SeriesFeaturizer

X = SeriesFeaturizer(transactions, id_col="customer", time_col="date", cutoff="as_of",
                     model="TiRex", n_components=4).fit_transform(customers)
```

Each row gets summary statistics and model embeddings of its entity's history as of its own timestamp, for any `TabularPipeline` model.

### Command line

```bash
tabtune timeseries list-models --commercial
tabtune timeseries forecast  --data sales.csv --model Chronos2 --horizon 14 --output forecast.csv
tabtune timeseries evaluate  --data sales.csv --model TiRex --horizon 14
tabtune timeseries anomalies --data sensors.csv --model ChronosBolt
tabtune timeseries benchmark --models Chronos2,TiRex,SeasonalNaive --data a.csv --data b.csv --horizon 24
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

### Fine-Tuning (`finetune`)
Full parameter fine-tuning. Updates all model weights using task data.

- **Meta-Learning (default for ICL models)**: Episodic training that mimics the in-context learning paradigm
- **SFT (Supervised Fine-Tuning)**: Standard supervised training on batches

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="finetune",  # finetune_mode defaults to 'meta-learning'
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
- ⚠️ **Experimental**: ContextTab and TabPFN (may cause prediction issues; use 'finetune' instead)

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
    finetune_mode: str | None = None,
    *,
    cache: str | bool | None = None,        # new in 0.2.0
    envelope_mode: str = 'warn',            # new in 0.2.0
    license_mode: str = 'research',         # new in 0.2.0
    validate: bool = True,                  # new in 0.2.0
)
```

#### Key Parameters:

- **`model_name`** (str): Model name or alias. Resolution ignores case, hyphens, underscores and whitespace, so `'TabPFN-v2.6'` and `'tabpfnv26'` are equivalent. 21 models are registered — call `tabtune.registry.list_model_names()` rather than hardcoding a list.

- **`task_type`** (str): The type of task — `'classification'` or `'regression'`.

- **`tuning_strategy`** (str): The strategy for model adaptation: `'inference'`, `'finetune'`, or `'peft'`.

- **`finetune_mode`** (str, optional): Controls the fine-tuning algorithm. If `None`, a smart default is chosen per task type (`'turn_by_turn'` for regression, `'meta-learning'` for classification). Supported values per model:
  - `'meta-learning'` — episodic meta-learning (TabICL, TabICLv2, OrionMSP, OrionMSPv1.5, OrionBix, TabDPT, Mitra, TabPFNv26, TabPFNv3, TabFM, iLTM, EXAONE)
  - `'sft'` — supervised fine-tuning (TabPFN, TabPFNv26, TabPFNv3, Mitra, TabDPT, TabFM, ContextTab, iLTM, EXAONE)
  - `'native'` — PriorLabs native finetuner with bar distribution loss, AMP, early stopping (**TabPFNv2.6 and TabPFNv3**, classification and regression)
  - `'turn_by_turn'` / `'tbt'` — episodic turn-by-turn; the default for regression (TabPFN, TabPFNv26, TabPFNv3, TabICLv2, Mitra, TabDPT, ContextTab, LimiX, TabFM, iLTM, EXAONE)
  - `'refit'` / `'refine'` — **xRFM only**, which has no gradient-descent fine-tuning. `refit` fits the RFM from scratch; `refine` warm-starts from the learned **M** matrix

  Not every model implements every mode. Check `get_model_spec(name).finetune_modes`.

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

#### Keyword-only parameters (new in 0.2.0)

- **`cache`** (str | bool | None): Prediction cache — `'memory'`, `'disk'`, `None`, or a `PredictionCache`. Enabling it collapses `evaluate()`'s three redundant forward passes into one.

- **`envelope_mode`** (str): How to treat data outside the model's documented limits — `'error'`, `'warn'` (default) or `'ignore'`. Architectural limits such as the class-count ceiling always raise unless this is `'ignore'`.

- **`license_mode`** (str): `'research'` (default), `'commercial'` to fail fast on weights that forbid commercial use, or `'ignore'`.

- **`validate`** (bool): Check model / task / strategy against the registry before loading weights. Set `False` to use a model TabTune does not know about.

```python
pipeline = TabularPipeline(
    model_name="TabICLv2",
    tuning_strategy="finetune",
    cache="disk",
    envelope_mode="error",
    license_mode="commercial",
)
print(pipeline.cache.stats)   # hits / misses / stores / hit_rate
```

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

> **Two models where `peft` does not mean what you expect.**
>
> - **xRFM** has no gradient-descent fine-tuning at all. Its `peft` performs low-rank
>   adaptation of the learned **M** matrix, not LoRA over linear layers, so
>   `target_modules` does not apply.
> - **EXAONE Tabular** applies its projections as raw `nn.Parameter` tensors through
>   `F.linear` rather than `nn.Linear` submodules. The injector finds nothing to wrap,
>   logs a warning, and the run proceeds as a **full fine-tune** — so do not attribute
>   a memory number to LoRA here.

---

## 🎯 Uncertainty You Can Actually Deploy

Tabular foundation models win benchmarks on accuracy and lose them on
uncertainty: their probabilities come out of an in-context softmax that was
never calibrated to your dataset. TabTune could already *measure* that
(`evaluate_calibration` reports ECE/MCE/Brier); `tabtune.uncertainty` adds the
two standard fixes. Both consume only `predict_proba`/`predict`, so they work
for every bundled model, for distilled students, for ensembles, and for plain
scikit-learn estimators.

```python
from tabtune.uncertainty import ConformalClassifier, Recalibrator

# Prediction sets with a distribution-free 90% coverage guarantee
cp = ConformalClassifier(pipe, method="lac", alpha=0.1).calibrate(X_cal, y_cal)
sets = cp.predict_set(X_test)         # which labels you cannot rule out

# Post-hoc recalibration - same API, better probabilities
pipe = Recalibrator(pipe, method="temperature").fit(X_cal, y_cal)

# One-call diagnostic: ECE/MCE/Brier + coverage + set sizes + SSCS
pipe.uncertainty_report(X_test, y_test, X_cal=X_cal, y_cal=y_cal)
```

The guarantee is stated precisely: **marginal** coverage under exchangeability,
not per-subgroup, and it degrades under distribution shift. The report's
size-stratified coverage score (SSCS) is there to show how far conditional
coverage falls short of the marginal number. The calibration split must be
disjoint from training — passing the pipeline's own training frame raises
rather than silently voiding the guarantee. Regression gets
`ConformalRegressor` (absolute-residual for any model, CQR where native
quantiles exist).

See the `tabtune.uncertainty` module docstrings and
[`examples/18_uncertainty.py`](examples/18_uncertainty.py).

---

## 📉 Shift-Aware Evaluation

An IID cross-validation score is the wrong question for a model that will be deployed
against future or out-of-cohort data. `ShiftEvaluator` reports the **gap** between the
two, which is the number that predicts production behaviour.

```python
from tabtune import TabularPipeline
from tabtune.evaluation import TemporalSplit, GroupedSplit, ShiftEvaluator

def factory():
    return TabularPipeline("TabICLv2", task_type="classification")

evaluator = ShiftEvaluator(splits={
    "temporal": TemporalSplit(4, time_col="date"),
    "grouped":  GroupedSplit(5),
})

report = evaluator.run(factory, X, y, groups=site_ids, drop_split_columns=True)
print(report)
```

```
ShiftReport(TabICLv2, task=classification, metric=roc_auc_score)
  iid                  roc_auc_score=0.9124 (baseline)
  temporal             roc_auc_score=0.8689  gap -0.0435
```

A model scoring 0.87 with a 0.004 gap is a better production bet than one scoring 0.89
with a 0.06 gap. Splits available: `TemporalSplit` (forward chaining, never trains on the
future), `GroupedSplit` (leave-groups-out), `StratifiedGroupedSplit` (grouped + class
balance).

> `drop_split_columns=True` removes split-defining columns from the features. Without it,
> the model can read the very column that defines the split and learn the cut point
> instead of the signal.
>
> Check `report.failures()` before quoting any aggregate — a fold can fail for ordinary
> reasons, and the report keeps going.

---

## 🗂️ Typed Configuration

Every knob is described by a pydantic schema, which makes an experiment a file you can
commit. Plain dicts still work; the difference is that a typo now warns instead of
vanishing.

```python
from tabtune.config import load_config

cfg = load_config("experiments/tabiclv2_finetune.yaml", strict=True)
cfg.validate_against_registry()          # fails before any weights load
cfg.resolved_finetune_mode()             # 'meta-learning'
```

```yaml
model_name: TabICLv2
task_type: classification
tuning_strategy: finetune
envelope_mode: error
license_mode: commercial

tuning:
  epochs: 10
  learning_rate: 1.0e-5
  seed: 0
  early_stopping: true
  validation_split: 0.15

processor:
  scaling_strategy: standard
  imputation_strategy: none
```

Use `strict=True` in CI so a stale config fails the job rather than quietly running
something else.

---

## 🪵 Logging

Every pipeline, leaderboard, ensemble and benchmark reports through the `tabtune` logger. `setup_logger` configures it once, for both the tabular and the time series modules:

```python
from tabtune import setup_logger

setup_logger()                                   # Rich in a terminal or notebook, plain text when redirected
setup_logger(level="WARNING")                    # warnings only
setup_logger(log_file="runs/experiment.jsonl", file_format="json", file_level="DEBUG")   # structured events on disk
```

Console output goes to stderr by default; `stream=sys.stdout` restores the old stream. Repeated calls replace only the handlers TabTune created. See [`docs/development/logging.md`](docs/development/logging.md) and [`examples/logging_showcase.py`](examples/logging_showcase.py).

---

## ⚡ Prediction Caching

```python
pipeline = TabularPipeline("TabICLv2", cache="memory")   # or "disk"
pipeline.fit(X_train, y_train)
pipeline.evaluate(X_test, y_test)

print(pipeline.cache.stats)
# hits=2 misses=1 stores=1 hit_rate=67%
```

Entries are keyed on a fingerprint covering the fitted model **and** the input data, so
refitting or changing the data invalidates automatically. Disk caching pays off most in
leaderboard runs and shift-evaluation sweeps, which query the same test rows repeatedly.

---

## 🏆 Example Notebooks

Example notebooks showcasing the library's features in depth. Runnable scripts for every feature also live in `examples/`.

### Tabular

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
| 17 | EXAONE Model | End-to-end usecase of the EXAONE Model |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1RR-mhJkGW0xpAU0si76ru-1NuggPKDNH#scrollTo=89s93vNcnSk0)| 

### Time series

The notebooks live in [`notebooks/timeseries/`](notebooks/timeseries/). Each one runs on a CPU; foundation model weights download from the Hugging Face Hub on first use.

| Serial No. | Name | Task Performed | Notebook |
|---|------|------|------|
| 1 | Quickstart | Fit, forecast, plot, evaluate, backtest, save; swap the model by name | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/01_quickstart_forecasting.ipynb) |
| 2 | Model catalog and licences | Every registered model, its envelope and weight licence; `license_mode`, `envelope_mode`, the CLI | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/02_model_catalog_and_licenses.ipynb) |
| 3 | Quantiles and calibration | Quantile levels per model, interval coverage, conformal `calibrate()` for any forecaster | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/03_probabilistic_forecasts_and_calibration.ipynb) |
| 4 | Covariates and multivariate | Past and known covariates with `future_df`, refusals, joint forecasts of several targets | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/04_covariates_and_multivariate.ipynb) |
| 5 | Fine-tuning and LoRA | `finetune` and `peft` with early stopping, the training report, fair comparison on a leaderboard | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/05_fine_tuning_and_lora.ipynb) |
| 6 | Anomalies, imputation, embeddings | Three more tasks from one fitted model; `SeriesFeaturizer` history features for a tabular model | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/06_anomalies_imputation_embeddings.ipynb) |
| 7 | Leaderboards, ensembles, selection | `TimeSeriesLeaderboard`, `TimeSeriesEnsemble`, `TimeSeriesPipeline.select` with licence filtering | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/07_leaderboard_ensemble_and_selection.ipynb) |
| 8 | Tabular models as forecasters | TabPFN-TS, `TabularTS-*`, the time and lags designs, any scikit-learn regressor | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/08_tabular_models_as_forecasters.ipynb) |
| 9 | Benchmarks and the CLI | `TimeSeriesBenchmark` on built-in and your own datasets, reports, `tabtune timeseries` commands | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/notebooks/timeseries/09_benchmark_and_cli.ipynb) |


---

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

Full documentation: **[tabtune.lexsi.ai](https://tabtune.lexsi.ai/)**

| Topic | Page |
|---|---|
| Models compared (all 21) | `docs/models/overview.md` |
| One page per model | `docs/models/*.md` (TabPFN v2 to v3.5, TabICL/v2, Orion, TabDPT, Mitra/v2, ConTextTab, LimiX, TabFM, EXAONE, xRFM, iLTM, Causilo, TabLDM) |
| Time series models (`TimeSeriesPipeline`) | `docs/timeseries/overview.md`, `docs/timeseries/models.md`, `docs/timeseries/fine-tuning.md` |
| Time series notebooks | `notebooks/timeseries/` |
| Logging | `docs/development/logging.md` |
| Tabular models as forecasters | `docs/timeseries/tabular-forecasters.md` |
| Time series histories as tabular features | `docs/timeseries/series-features.md` |
| Real-weight benchmarks | `benchmarks/README.md` |

The registry, configuration, caching, shift-evaluation, uncertainty, ensembling, distillation
and causal modules are documented in their module docstrings and in `examples/`.

Build the docs locally:

```bash
pip install "tabtune[docs]"
mkdocs serve
```

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
- **[Chronos](https://github.com/amazon-science/chronos-forecasting)** - Chronos, Chronos-Bolt and Chronos-2 time series foundation models (Amazon)
- **[TimesFM](https://github.com/google-research/timesfm)** - Decoder-only time series foundation model (Google Research)
- **[TiRex](https://github.com/NX-AI/tirex)** and **[TiRex-2](https://github.com/NX-AI/tirex-2)** - xLSTM time series forecasters (NXAI)
- **[Time-MoE](https://github.com/Time-MoE/Time-MoE)** - Mixture-of-experts time series foundation models
- **[Toto](https://github.com/DataDog/toto)** - Toto 1 and Toto 2 time series foundation models for observability data (Datadog)
- **[TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series)** - Time series forecasting with TabPFN

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size` in `tuning_params`
- Use `tuning_strategy='peft'` for PEFT mode
- Decrease `n_ensembles` or `context_size` for inference

### PEFT Compatibility Issues
- Some models have experimental PEFT support; use the 'finetune' strategy instead
- Check logs for model-specific warnings

### Device Mismatch
- Ensure `device` parameter matches your hardware (cuda/cpu)
- Use `torch.cuda.is_available()` to check GPU availability

---

## 🗃️ License

This project is released under the MIT License.  
Please cite appropriately if used in academic or production projects.

Vendored model code keeps its own license, in the `LICENSE` file of its folder under
`tabtune/models/`: Chronos, TimesFM, Toto, TiRex-2 and Time-MoE are Apache-2.0, and TiRex is under the
NXAI Community License. Built with technology from NXAI. Weight licences differ from code licences and are
recorded per checkpoint in the registry; `license_mode="commercial"` checks them before any download.

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
