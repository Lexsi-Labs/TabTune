# OrionMSP v1.5: Hierarchical Multi-Scale Sparse Attention for Tabular In-Context Learning

OrionMSP v1.5 (a.k.a. `orionmsp_v15` in TabTune) is an in-context learning (ICL) foundation model for **tabular classification** that combines:
- **Column-wise distribution-aware embedding**
- **Hierarchical sparse row interaction**
- **Perceiver-style cross-component memory**
- **ICL predictor with optional thinking tokens**
- **Inference-time acceleration via KV caching**


---

## 1. Introduction

**What is OrionMSP v1.5?**

OrionMSP is designed for tabular tasks in the *in-context* setting: given a **support/context set** of labeled rows and a **query set** of unlabeled rows, the model predicts query labels in a single forward pass by learning dataset-specific decision rules from the context.

OrionMSP is a four-stage pipeline:  
1) Column-wise embedder → 2) Multi-scale sparse row interaction → 3) Cross-component memory → 4) ICL predictor.  

---

## 2. Architecture

### 2.1 High-Level Design (as implemented in TabTune)

```mermaid
flowchart LR
    A[Raw tabular rows: numeric + categorical] --> B[Column-wise embedder]
    B --> C[Cell/feature embeddings E]
    C --> D[RowInteraction: sparse attention over feature axis]
    D --> E[[CLS] summaries -> row embeddings H]
    E --> F[Cross-component Perceiver memory]
    F --> G[Enhanced row reps R]
    G --> H[ICL predictor (split attention)]
    H --> I[Logits for query rows]
```

### 2.2 Core Components 

1) **Column-wise embedder**
   - Encodes each cell value while conditioning on the column distribution (paper §3.2).
   - Uses a shared set-based column encoder across all columns (paper notes the shared Set Transformer approach).

2) **RowInteraction: hierarchical sparse attention**
   - Prepends **[CLS]** and **[GLOBAL]** tokens, then performs sparse attention over the feature axis at one or more resolutions (paper Figure 1 description).
   - Sparse attention uses a small local window plus global routing tokens to keep compute close to linear in the number of features (paper Appendix B.2 complexity discussion).

3) **Perceiver-style memory (optional, but enabled by default in v1.5 settings)**
   - Compresses the *training/context* rows into a fixed number of latent vectors, then lets all rows read from these latents (paper Algorithm 2).
   - Preserves ICL causality because *writes* use context rows only (paper Appendix B.2).

4) **ICL predictor (split-attention transformer)**
   - Injects labels only into context rows (paper Eq. 31–32).
   - Uses a split self-attention mask: context (+ thinking tokens) attend freely; query rows attend only to the context region (paper Eq. 33–35).
   - Produces logits for query rows only.


## 3. Inference Parameters

### 3.1 Complete Parameter Reference 

```python
model_params = {
    # Embedding / column encoder
    "embed_dim": 256,
    "dropout": 0.0,
    "activation": "gelu",
    "norm_first": True,

    # Column interaction (set-based / inducing points)
    "col_num_blocks": 3,
    "col_nhead": 4,
    "col_num_inds": 128,

    # NEW in v1.5: feature positional embeddings
    "feature_pos_emb": "subspace",   # "subspace", "learned", or None

    # Row interaction (sparse TFrow)
    "row_num_blocks": 9,             # v1.5 default (deeper)
    "row_nhead": 8,
    "row_num_cls": 4,
    "row_rope_base": 100000,
    "row_num_global": 2,
    "row_scales": (1,),              # v1.5 default (single scale)
    "row_window": 4,
    "row_num_random": 0,
    "row_group_mode": "pma",

    # NEW in v1.5: grouping + scale aggregation
    "features_per_group": 2,
    "scale_combine_method": "enhanced_attention",

    # Perceiver memory (cross-component memory)
    "perc_num_latents": 32,          # v1.5 default (increased)
    "perc_layers": 2,

    # NEW in v1.5: enhanced memory + thinking interface
    "num_memory_heads": 4,
    "use_memory_gating": True,
    "num_thinking_tokens": 0,

    # ICL predictor
    "icl_num_blocks": 12,
    "icl_nhead": 4,

    # Task shaping
    "max_classes": 10,               # typical cap in TabTune usage
}
```

### 3.2 Inference Manager Reference (v1.5)

```python
inference_manager_params = {
    "min_batch_size": 1,
    "safety_factor": 0.8,
    "offload": "auto",               # or True/False
    "auto_offload_pct": 0.5,
    "device": "cuda",
    "use_amp": True,
    "verbose": False,

    # NEW in v1.5
    "enable_kv_cache": False,
    "cache_trainset_representation": False,
}
```

### 3.3 Parameter Descriptions (key deltas)

| Parameter | Type | v1.0 Default | v1.5 Default | What it does |
|---|---:|---:|---:|---|
| `row_num_blocks` | int | 3 | 9 | Depth of row sparse transformer blocks |
| `row_scales` | tuple[int] | (1,4,8) | (1,) | Multi-scale feature resolutions |
| `row_group_mode` | str | contiguous | pma | How features are grouped before sparse attention |
| `feature_pos_emb` | str/None | None | subspace | Adds feature-position signal to embeddings |
| `features_per_group` | int | — | 2 | Grouping granularity for features |
| `scale_combine_method` | str | — | enhanced_attention | How outputs across scales are combined |
| `perc_num_latents` | int | 16 | 32 | Memory latent capacity |
| `num_memory_heads` | int | — | 4 | Multi-head memory reads |
| `use_memory_gating` | bool | — | True | Dataset-conditioned memory gating |
| `num_thinking_tokens` | int | — | 0 | Inserts learnable tokens between context and query |
| `enable_kv_cache` | bool | — | False | Cache KV pairs for repeated predictions |
| `cache_trainset_representation` | bool | — | False | Cache training-set representations |

---

## 4. Fine-Tuning with OrionMSP v1.5

OrionMSP is primarily an **ICL model** (context + query). In TabTune usage, you typically:
- **fit** the pipeline (stores training data as context pool / support),
- then **predict** on query rows.


```python
tuning_params = {
    "device": "cuda",
    "epochs": 3,
    "learning_rate": 2e-5,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "gradient_clip_value": 1.0,
    "show_progress": True,
}
```

**Tip:** OrionMSP-style models are sensitive to context length. If you increase the number of support rows too aggressively, you can increase memory cost and latency significantly. Consider resampling/context sampling strategies already supported in TabTune.

---

## 5. LoRA / PEFT Target Modules (practical guidance)

If you apply PEFT/LoRA in your TabTune setup, start by targeting:
- attention projections in column encoder,
- row interaction attention blocks,
- ICL predictor blocks,
- output head / label encoder.

A common safe starting configuration:

```python
peft_config = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": None  # use your project defaults or map to attention/MLP blocks
}
```

---

## 6. Usage Patterns

### 6.1 Inference-only usage (typical OrionMSP flow)

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="OrionMSPv1.5",   
    tuning_strategy="inference",
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
```

### 6.2 Faster repeated prediction (KV cache)

Use KV caching when:
- you call `predict()` multiple times on the same fitted dataset,
- or you run large batch inference where reuse helps.

```python
# Pseudocode: depends on how your TabTune pipeline exposes the manager config
pipeline = TabularPipeline(
    model_name="OrionMSPv1.5",
    tuning_strategy="inference",
    tuning_params={
        "inference_manager_params": {
            "enable_kv_cache": True,
            "cache_trainset_representation": True,
        }
    }
)
```

---


## 7. Best Practices

### ✅ Do’s
- ✅ Start with v1.5 defaults (deeper RowInteraction + memory improvements).
- ✅ Use moderate context sizes; scale carefully with compute.
- ✅ Turn on KV caching if you do repeated inference on the same fitted dataset.
- ✅ Consider minimal thinking tokens (e.g., 2) if you need more intermediate reasoning capacity.

### ❌ Don’ts
- ❌ Don’t increase context length without checking latency/memory.
- ❌ Don’t set large thinking-token buffers by default.
- ❌ Don’t disable memory blindly—v1.5’s memory heads + gating are designed to help.

---

## 8. OrionMSP v1.5 vs v1.0 (quick comparison)

| Dimension | OrionMSP v1.0 (`orion_msp`) | OrionMSP v1.5 (`orionmsp_v1.5`) |
|---|---|---|
| RowInteraction depth | 3 blocks | 9 blocks (default) |
| Multi-scale default | (1,4,8) | (1,) |
| Feature positional embeddings | — | `feature_pos_emb` (default: subspace) |
| Feature grouping | contiguous | `pma` + `features_per_group` |
| Scale aggregation | baseline | `scale_combine_method="enhanced_attention"` |
| Perceiver memory latents | 16 | 32 |
| Memory reads | single-head | multi-head (`num_memory_heads`) |
| Memory gating | — | enabled (`use_memory_gating=True`) |
| Thinking tokens | — | supported (`num_thinking_tokens`) |
| Inference optimization | baseline | KV caching options |



## 9. Next Steps

- Add this file under your docs **models** section (same place as the existing `orion-msp.md`), e.g.:
  - `docs/models/orionmsp1.5.md` (or your chosen filename)
- Add it to your docs navigation / sidebar index if your docs build uses an explicit nav list.

---

OrionMSP v1.5 is a practical, scalable ICL foundation model for tabular data with improved memory, grouping, and inference acceleration. Use it when you want strong cross-dataset generalization without per-task retraining.
