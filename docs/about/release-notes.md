# Release Notes

------------------------------------------------------------------------

## Release Notes -> 26th Feb 2026

**TabTune** marks the first production-ready release of the unified
tabular foundation model framework.

### 🎯 Major Highlights

-   Fully unified `TabularPipeline` API (`fit`, `predict`, `evaluate`,
    `save`, `load`)
-   Model-aware `DataProcessor` for automated preprocessing
-   `TuningManager` with three strategies:
    -   `inference` (zero-shot)
    -   `base-ft` (full fine-tuning)
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

#### ✅ OrionMSP v1.5 and Limix Model Support

#### ✅ Regression Framework

-   Unified regression training pipeline
-   Standardized evaluation metrics
-   Benchmark-ready structure

#### ✅ Resampling Module

-   Context-aware support/query sampling
-   Episodic batching utilities
-   ICL model optimization support

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
