# Release Notes

------------------------------------------------------------------------

## Release — 2nd April 2026

### 🎯 Major Highlights

- **TabPFNv2.6 Integration** — Full support for PriorLabs' latest TabPFN release, covering classification and regression with inference and fine-tuning. Includes a dedicated **native fine-tuning mode** (`finetune_mode='native'`) backed by `FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor` with bar distribution loss, cosine LR scheduling with warmup, mixed-precision (AMP), early stopping, and validation-based model selection.
- **TabICLv2 Integration** — Full support for TabICLv2 for both classification and regression, with inference and episodic fine-tuning for both tasks. Regression fine-tuning uses turn-by-turn episodic MSE training.

## Release Notes -> 26th Feb 2026

**TabTune** marks the first production-ready release of the unified
tabular foundation model framework.

### 🎯 Major Highlights

-   Fully unified `TabularPipeline` API (`fit`, `predict`, `evaluate`,
    `save`, `load`)
-   Model-aware `DataProcessor` for automated preprocessing
-   `TuningManager` with three strategies:
    -   `inference` (zero-shot)
    -   `finetune` (full fine-tuning)
    -   `peft` (LoRA-based parameter-efficient fine-tuning)
-   `TabularLeaderboard` for benchmarking and model comparison

------------------------------------------------------------------------

### 🧠 Supported Models (9 Total)

-   TabPFN-v2\
-   TabICL\
-   OrionMSP v1.0\
-   **OrionMSP v1.5 (New)**\
-   OrionBix\
-   TabDPT\
-   Mitra\
-   ContextTab\
-   **LimiX (New)**

------------------------------------------------------------------------

### 🆕 New Additions 

#### ✅ TabPFN v2.6 (`model_name='TabPFNv26'`)

- Classification: inference, meta-learning FT, SFT, native FT
- Regression: inference, turn-by-turn FT, native FT
- `finetune_mode='native'` uses `FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor` with:
  - Bar distribution loss (regression)
  - Cosine LR with warmup
  - Mixed-precision (AMP)
  - Early stopping with patience
  - Validation-based model selection
  - Gradient clipping
  - Activation checkpointing
- New `tuning_params` keys: `early_stopping`, `early_stopping_patience`, `validation_split_ratio`, `n_estimators_finetune`, `n_estimators_validation`, `n_estimators_final_inference`, `grad_clip_value`, `use_lr_scheduler`, `use_activation_checkpointing`

#### ✅ TabICLv2 (`model_name='TabICLv2'`)

- Classification: inference + finetune (episodic meta-learning)
- Regression: inference + finetune (episodic turn-by-turn MSE)
- Regression FT uses AdamW with gradient clipping and post-finetune re-fit for inference cache rebuild
  
------------------------------------------------------------------------

### ⚙️ Improvements

-   Cleaner modular architecture
-   Better memory management
-   Improved gradient stability for MSP models
-   Colab compatibility enhancements
-   Expanded serialization support

------------------------------------------------------------------------

### 🛠 Developer Experience

-   Modular structure for adding new models
-   Improved documentation for contributions
-   Extended API reference coverage
-   Updated project structure clarity

------------------------------------------------------------------------

## 0.1.0 --- Alpha Release

-   Initial alpha release
-   Introduced:
    -   `TabularPipeline`
    -   `DataProcessor`
    -   `TuningManager`
    -   `TabularLeaderboard`
-   Basic documentation:
    -   Getting Started
    -   User Guide
    -   Models
    -   API Reference

------------------------------------------------------------------------

**TabTune** establishes a complete foundation for tabular model
inference, fine-tuning, benchmarking, regression workflows, and
resampling-aware meta-learning.
