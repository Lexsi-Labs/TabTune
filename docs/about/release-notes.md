# Release Notes

------------------------------------------------------------------------

## v0.2.0

This is the largest release since the first public version: **seven new models**, and four new
subsystems aimed at getting a tabular foundation model into production rather than onto a
benchmark table.

### New models (16 total, up from 9)

| Model | Family | Notes |
|---|---|---|
| **EXAONE Tabular** (`EXAONETabular` / `EXAONE`) | Cross-axis ICL (LG AI Research) | Cross-axis Summary Transformer (CAST). TabTune vendors the full inference runtime: ECOC decomposition for >10 classes, attention-based feature selector, CUDA execution planner. ~21M params, 8-member default ensemble. |
| **xRFM** (`XRFM`) | Kernel / feature learning | Recursive Feature Machine with AGOP feature learning and tree-partitioned EigenPro. **No pretrained weights** — trains from scratch, so it is the only bundled model that works air-gapped out of the box. |
| **iLTM** (`ILTM`) | Hypernetwork | A hypernetwork generates MLP ensembles conditioned on dataset embeddings, combining GBDT tree embeddings with retrieval. Apache-2.0 weights, ungated. |
| **TabFM** (`TabFM`) | Hybrid-attention ICL (Google) | Alternating row/column attention → CLS row compression → causal ICL transformer. Full PEFT support. |
| **TabPFN v3** (`TabPFNv3`) | PFN / ICL | 160 classes, 20,000 features, ~200M cell budget. Native, meta-learning, SFT and PEFT fine-tuning for both tasks. |
| **TabICLv2** (`TabICLv2`) | Scalable ICL | QASSMax normalisation plus a native quantile regression head. |
| **TabPFN v2.6** (`TabPFNv26`) | PFN / ICL | Prior Labs' native fine-tuning API with bar-distribution loss. |

### Model registry 

A **torch-free** registry recording each checkpoint's **capability envelope** and **weight
licence**, both checked before any weights load.

- `list_models()`, `list_model_names()`, `models_dataframe()`, `get_model_spec()`
- `resolve_model_name()` — case-, hyphen- and dot-insensitive alias resolution
- `check_envelope()` — class, feature, row and cell limits; hard limits raise, soft limits warn
- `check_license()` — tri-state `commercial_use_ok`; unverified terms warn rather than block
- `register_model()` — add your own `ModelSpec`
- New exceptions: `TabTuneError`, `ConfigError`, `ModelNotFoundError`, `UnsupportedTaskError`, `UnsupportedStrategyError`, `EnvelopeError`, `LicenseError`

New `TabularPipeline` keyword-only parameters: `envelope_mode`, `license_mode`, `validate`.

See [Model Registry](../user-guide/registry.md).

### Uncertainty quantification 

`tabtune.uncertainty` adds the two standard fixes for TFM overconfidence. Both consume only
`predict_proba` / `predict`, so they work for pipelines, ensembles, distilled students and
plain scikit-learn estimators.

- `ConformalClassifier` — `'lac'` and `'aps'` methods, `predict_set()`, `set_sizes()`, `coverage()`
- `ConformalRegressor` — `'absolute'` for any model, `'cqr'` where native quantiles exist
- `Recalibrator` — post-hoc temperature scaling; composes around the pipeline without mutating it
- `uncertainty_report()` / `pipeline.uncertainty_report()` — ECE, MCE, Brier, coverage, set sizes and **size-stratified coverage (SSCS)**
- `size_stratified_coverage()`

Calibration rows must be disjoint from training: a re-used training frame is detected by
fingerprint and **raises** rather than silently voiding the guarantee.

See [Uncertainty Quantification](../user-guide/uncertainty.md).

### Shift-aware evaluation 

- `TemporalSplit` — forward chaining with optional embargo `gap`; never trains on the future
- `GroupedSplit` — leave-groups-out
- `StratifiedGroupedSplit` — grouped **and** class-balanced
- `ShiftEvaluator` / `ShiftReport` / `FoldResult` / `shift_gap()` — reports the IID-to-shift gap
- `drop_split_columns` prevents the model reading the column that defines the split
- Shared metrics moved to `tabtune.evaluation.metrics`, so the pipeline, leaderboard, benchmark CSV and shift report agree by construction

See [Shift-Aware Evaluation](../user-guide/shift-evaluation.md).

### Typed configuration

Pydantic schemas for every knob, with YAML/JSON round-tripping.

- `PipelineConfig`, `TuningConfig`, `PeftConfig`, `ProcessorConfig`, `ContextSamplingConfig`
- `load_config()`, `save_config()`, `dump_config()`, `config_from_mapping()`
- `cfg.validate_against_registry()` — fails before any weights load
- `cfg.resolved_finetune_mode()`
- `strict=True` turns an unknown key into an error, for CI

Plain dicts still work everywhere. Unknown keys are still forwarded to the model — the
difference is they now **warn** instead of vanishing.

See [Configuration](../user-guide/configuration.md).

### Prediction caching 

`cache='memory'` or `'disk'` collapses `evaluate()`'s three redundant forward passes into
one. Entries are keyed on a fingerprint covering the fitted model *and* the input data, so
refitting or changing the data invalidates automatically.

- `pipeline.cache.stats`, `pipeline.clear_cache()`
- `PredictionCache`, `CacheStats`, `make_cache()`, `fingerprint_data()`

See [Caching](../user-guide/caching.md).

### Causal inference

`CausalAnalysis` with six estimators (DML, S/T/X/R-Learner, Causal Forest), formal
identification, a refutation framework (placebo, random common cause, subset stability,
sensitivity), proxy-attribute auditing, counterfactual fairness and `CausalLeaderboard`.
Install with `pip install "tabtune[causal]"`.

See [Causal Inference](../user-guide/causal.md).

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
