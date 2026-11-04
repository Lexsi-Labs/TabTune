# Changelog

All notable changes to TabTune. The format follows [Keep a Changelog](https://keepachangelog.com/).

## 0.2.2

### Added

- **LimiX-2** (`LimiXv2`): Stable AI's 400M-parameter LimiX-2, vendored from LimiX
  `inference/v2_0` and `model/v2_0`. Classification, regression, episodic fine-tuning and
  LoRA. LimiX v1 stays `Limix`.
- **NVIDIA Kumo Tabular** (`KumoTabular`): small, medium and large checkpoints of NVIDIA
  structured-data-models. Classification, regression, episodic fine-tuning and LoRA.
- **TabAbax** (`TabAbax`): BaiRong's TabICLv2-derived model, vendored from the TabAbax
  source tree. Classification, regression, episodic fine-tuning and LoRA through the
  model's own training branch.
- **Timestral** (`Timestral`): Mistral AI's Decoupled Horizon Forecasting models, 5M to
  209M. Multivariate targets, past, known and categorical covariates, 21 quantiles,
  fine-tuning and LoRA, on `TimeSeriesPipeline`.
- **T0** (`T0`): The Forecasting Company's t0-alpha, -beta and -gamma. Multivariate
  targets, known covariates, any quantile level, fine-tuning and LoRA.
- A validated episodic fine-tuning driver (`tabtune.TuningManager.episodic_finetune`) used
  by Kumo, LimiX-2 and TabAbax: held-out validation scored through the real inference path, the pretrained
  weights as the epoch-0 baseline, warmup and cosine schedule, stratified support sets sized
  like the inference context, gradient accumulation, early stopping, best weights restored
  (pretrained weights as the fallback), and a `finetune_report_` on the estimator. Replaces
  the fixed-schedule loops under which every fine-tuned run scored worse than zero-shot on
  the models' own benchmark.
- `pyarrow` as a core dependency (Kumo's tensor backend), and `typeguard>=2.13.3,<3`, which
  the vendored TabFM estimator imports through jaxtyping.

### Changed

- A regression pipeline without an explicit `finetune_mode` uses `meta-learning` for models
  that register no `turn_by_turn` mode (Kumo, LimiX-2, TabAbax, Causilo, TabLDM) instead of
  warning.

## 0.2.1

### Added

- **Time series on `TimeSeriesPipeline`.** The same pipeline pattern as tabular:
  `TimeSeriesPipeline(model_name, task_type, tuning_strategy, tuning_params, model_params,
  forecast_params)` with `fit`, `predict`, `evaluate`, `save` and `load`. Tasks are
  forecasting, anomaly detection, imputation and embeddings; strategies are zero-shot
  `inference`, `finetune` and `peft`. Conformal `calibrate()`, rolling-origin `backtest()`,
  context-length tuning and `select()`, plus `TimeSeriesLeaderboard`, `TimeSeriesEnsemble`
  and `TimeSeriesBenchmark`.
- Nine vendored time series foundation models: Chronos, Chronos-Bolt, Chronos-2, TimesFM 3,
  Toto 1, Toto 2, TiRex, TiRex-2, Time-MoE.
- **Five new tabular models**: TabPFN v3.5 and v3.5-fast (Prior Labs), Causilo (Nums AI),
  TabLDM (Xiaomi) and Mitra v2 (AWS), in the registry with envelopes and weight licences.
  Their preprocessing needs `skrub`, now a core dependency.
- Tabular models as forecasters: `TabPFN-TS` on the vendored TabPFN, `TabularTS-<name>`
  for every tabular model with a regression head, and the weight-free `TabularTS-GBM`.
- `tabtune.bridge.SeriesFeaturizer`: leak-safe features of an entity's history for any
  tabular model.
- The `tabtune timeseries` command line group.
- One logger: `setup_logger` configures Rich, plain or JSON output and rotating files once.

### Removed

- The `tabtune.tsfm` facade.

## 0.2.0

### Added

- Seven models: EXAONE Tabular, xRFM, iLTM, TabFM, TabPFN v3, TabICL v2, TabPFN v2.6.
- A torch-free model registry: `ModelSpec` per checkpoint with a capability envelope and a
  weight licence, `list_models`, `get_model_spec`, `check_envelope`, `check_license`,
  `register_model`, and the exception hierarchy (`TabTuneError`, `ConfigError`,
  `ModelNotFoundError`, `UnsupportedTaskError`, `UnsupportedStrategyError`,
  `EnvelopeError`, `LicenseError`). New `TabularPipeline` keyword-only parameters
  `envelope_mode`, `license_mode`, `validate`.
- Uncertainty: `ConformalClassifier` (`lac`, `aps`), `ConformalRegressor` (`absolute`,
  `cqr`), `Recalibrator`, `uncertainty_report()` with ECE, MCE, Brier, coverage, set sizes
  and size-stratified coverage. A calibration frame that overlaps training is detected by
  fingerprint and raises.
- Shift-aware evaluation: `TemporalSplit` (with embargo `gap`), `GroupedSplit`,
  `StratifiedGroupedSplit`, `ShiftEvaluator` reporting the IID-to-shift gap. Shared
  metrics in `tabtune.evaluation.metrics`.
- Typed configuration: `PipelineConfig`, `TuningConfig`, `PeftConfig`, `ProcessorConfig`,
  `ContextSamplingConfig`, YAML round trips, `validate_against_registry()`. Unknown keys in
  plain dicts now warn instead of vanishing.
- Prediction caching: `cache="memory"` or `"disk"`.
- Causal inference: `CausalAnalysis` with DML, S/T/X/R-learners and causal forests,
  identification, refutation, proxy auditing and counterfactual fairness.
- Distillation: `TabDistiller` into LightGBM, XGBoost, CatBoost or MLP students, single and
  multi-teacher, ONNX export.
- Lazy imports: `import tabtune` no longer pulls in torch or transformers.

## 0.1.1 (26 February 2026)

### Added

- OrionMSP v1.5 and LimiX.
- The regression framework: unified training, standard metrics, benchmark structure.
- The resampling module: context-aware support/query sampling and episodic batching.

### Changed

- `base-ft` renamed `finetune`.
- Better memory management, gradient stability for the MSP models, Colab compatibility,
  wider serialisation.

## 0.1.0

- First release: `TabularPipeline`, `DataProcessor`, `TuningManager`,
  `TabularLeaderboard`, nine models, the first documentation site.
