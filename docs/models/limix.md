# LimiX

LimiX (v1) treats a table as a joint distribution over its variables and their missingness, which is why one model serves classification, regression, missing-value imputation and embedding extraction without a task-specific head. In TabTune it is `Limix`; its successor is [LimiX-2](limixv2.md).

## Introduction
### What is LimiX?

LimiX was trained with masked joint-distribution modelling under an episodic, context-conditional objective. At inference it adapts through in-context learning; no weights change.

Key characteristics:

- Unified Interface: One model API for classification, regression, and other structured-data tasks.
- Context-Conditional Prediction: Learns to predict targets/variables conditional on a selected context window (episodes).
- Handles Missingness Naturally: Missingness is part of the modeled joint distribution, not just a preprocessing afterthought.
- Retrieval-Enhanced Inference: Uses attention-based retrieval to choose "best" in-context samples per query/test row.
- Strong Generalization: Built to handle varying schema sizes, categorical/numerical mixes, and distribution shifts.

### Model variants

LimiX commonly ships in two instantiations:

- LimiX-16M: Higher capacity; best performance across most benchmarks.
- LimiX-2M: Smaller footprint; strong performance under tight compute/memory.

In TabTune, you can treat these as **model variants** (e.g., via a `variant` or `checkpoint` field depending on your wrapper).

## High-level architecture

LimiX represents a dataset as a **2D table** (rows = samples, columns = features) plus an **outcome/target variable**. It embeds each cell value into a latent space and runs **axis-wise attention** over:

- **Feature axis** (column-wise interactions)
- **Sample axis** (row-wise interactions)

### End-to-end data flow

```mermaid
flowchart LR
    A[Raw Tabular Data X y] --> B[Cell wise Embedding MLP LN GELU]
    B --> C[Discriminative Feature Encoding]
    C --> D[Transformer Blocks Axis wise Attention]
    D --> E[Bi level Attention Retrieval Pass 1]
    E --> F[Customized Context Top K samples]
    F --> G[Second Forward Pass Pass 2]
    G --> H[Task Head Classification or Regression]
    H --> I[Predictions and Optional Embeddings]
```

## Inference parameters

### Complete parameter reference (regression)

In TabTune, the pipeline builds a default config for regression LimiX like:

```python
model_params = {
    # Runtime
    "device": "cuda",                      # or "cpu"
    "repo_id": "stableai-org/LimiX-16M",    # HuggingFace repo (TabTune regression default)
    "filename": "LimiX-16M.ckpt",          # checkpoint file

    # Ensemble behavior (TabTune default = 8)
    "n_estimators": 8,                     # number of ensemble members

    # Model architecture (may be overridden by checkpoint config)
    "nlayers": 6,
    "nhead": 6,
    "embed_dim": 192,
    "hid_dim": 768,
    "dropout": 0.1,

    # Preprocessing / ensemble diversity
    "seed": 0,
    "inference_config": None,              # dict/list/path; if None uses built-in defaults
    "features_per_group": 2,               # grouping inside FeaturesTransformer
    "preprocess_variant": "default"        # optional internal knob
}
```

> **Important:** the regression estimator loads the checkpoint first and, if present, uses the **checkpoint's config** to override `nlayers/nhead/embed_dim/hid_dim/dropout` to match the official model initialization behavior.

### Parameter descriptions

| Parameter | Type | Default (TabTune) | Description |
|---|---:|---:|---|
| `device` | str | auto | `"cuda"` if available else `"cpu"` |
| `repo_id` | str | `stableai-org/LimiX-16M` | HuggingFace repository for weights |
| `filename` | str | `LimiX-16M.ckpt` | Checkpoint to download/load |
| `n_estimators` | int | `8` | Regression ensemble size (matches paper setting for regression pipelines) fileciteturn1file11L18-L22 |
| `nlayers` | int | `6` | Transformer blocks in TabTune code (often overridden by checkpoint config) |
| `nhead` | int | `6` | Attention heads (often overridden by checkpoint config) |
| `embed_dim` | int | `192` | Embedding dimension (often overridden by checkpoint config) |
| `hid_dim` | int | `768` | FFN hidden size (often overridden by checkpoint config) |
| `dropout` | float | `0.1` | Dropout (often overridden by checkpoint config) |
| `seed` | int | `0` | Base seed; each estimator uses `seed + i*1000` for diversity |
| `inference_config` | dict/list/str | `None` | Preprocess pipeline configs; can be a JSON path or Python object |
| `features_per_group` | int | `2` | Controls feature grouping inside `FeaturesTransformer` |

## Tasks supported in TabTune

Within TabTune, LimiX is typically used for:

- Classification
- Regression

Advanced capabilities (depending on wrapper exposure):

- Embedding extraction
- Missing-value imputation
- Out-of-distribution evaluation
- Representation learning

## Fine-tuning with LimiX in TabTune (regression)

The LimiX paper notes retrieval can also support **efficient inference-time ensemble and fine-tuning**. fileciteturn1file2L9-L12

In TabTune, **regression fine-tuning is implemented as episodic training** in `TuningManager._finetune_limix_regression`:

### Fine-tuning parameters (TabTune)

```python
tuning_params = {
    "device": "cuda",
    "epochs": 3,
    "steps_per_epoch": 100,

    # Episode sizes
    "support_size": 256,   # aka context_size
    "query_size": 64,

    # Optimizer
    "lr": 1e-5,
    "weight_decay": 0.01,
    "clip_grad_norm": 1.0,

    # Misc
    "seed": 42,
    "show_progress": True
}
```

### How episodic fine-tuning works

For each episode:
1. Randomly sample `support_size + query_size` rows.
2. Build a single episode tensor `[1, S+Q, F]`.
3. Provide true `y` for support rows, and **zeros for query y** (prevents label leakage).
4. Forward call: `torch_model(X_episode, y_episode, eval_pos=S, task_type="reg")`
5. Optimize MSE on query predictions.

After fine-tuning, TabTune calls `model.fit(X_train, y_train)` again to refresh ensemble state for inference.

## Usage patterns

### Regression: inference (default)

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="Limix",
    task_type="regression",
    tuning_strategy="inference",   # default path
    model_params={
        "repo_id": "stableai-org/LimiX-16M",
        "filename": "LimiX-16M.ckpt",
        "n_estimators": 8,
    },
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
```

### Regression: episodic fine-tuning

```python
pipeline = TabularPipeline(
    model_name="Limix",
    task_type="regression",
    tuning_strategy="finetune",
    tuning_params={
        "device": "cuda",
        "epochs": 3,
        "steps_per_epoch": 100,
        "support_size": 256,
        "query_size": 64,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "clip_grad_norm": 1.0,
        "seed": 42,
        "show_progress": True,
    },
    model_params={
        "repo_id": "stableai-org/LimiX-16M",
        "filename": "LimiX-16M.ckpt",
        "n_estimators": 8,
    },
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
```
