# Features

TabTune is two pipelines that share one habit: the library adapts to the model, not the
other way round.

1. **`TabularPipeline`** wraps 24 tabular foundation models. You name a model, a task and a
   tuning strategy; preprocessing, the training loop and the licence check are chosen for
   you.
2. **`TimeSeriesPipeline`** does the same for eleven time series foundation models, every
   tabular model as a forecaster, and five statistical baselines.

Every row below links to the guide that explains it and, where one exists, to the script
under `examples/` that runs it. The scripts need the real checkpoints, so most of them
want a GPU or patience.

## Tabular

| Feature | What it does | Guide | Example |
|---|---|---|---|
| Unified pipeline | `fit`, `predict`, `predict_proba`, `evaluate`, `save`, `load` over every model. | [TabularPipeline](user-guide/pipeline.md) | `01_unified_api.py` |
| Model-aware preprocessing | `DataProcessor` picks imputation, scaling and encoding per model; special preprocessors for models that want raw frames. | [Data Processing](user-guide/data-processing.md) | `02_automated_preprocessing.py` |
| Zero-shot inference | In-context prediction with the training set as context, no gradient step. | [Tuning Strategies](user-guide/tuning-strategies.md) | `01_unified_api.py` |
| Fine-tuning modes | `meta-learning` (episodic support/query), `sft` (one fixed split), `turn_by_turn` (regression), `native` (the vendor's own loop, TabPFN v2.6+). | [Tuning Strategies](user-guide/tuning-strategies.md) | `03_finetuning_strategies.py` |
| Validated episodic fine-tuning | Held-out validation, pretrained weights as the epoch-0 baseline, warmup and cosine schedule, early stopping, best weights restored. Used by Kumo, LimiX-2 and TabAbax. | [Validated Fine-Tuning](user-guide/fine-tuning.md) | `22_new_models_finetune_effect.py` |
| PEFT with LoRA | Adapters on each model's attention and feed-forward layers, with per-model target tables. | [PEFT & LoRA](advanced/peft-lora.md) | `06_advanced_usage.py` |
| Regression | Native heads where the model has one, bar-distribution or quantile heads elsewhere, `predict_quantiles` and `predict_intervals`. | [Regression](user-guide/regression.md) | `10_regression_model_comparison.py`, `11_regression_finetuning.py` |
| Model registry | Torch-free `ModelSpec` per checkpoint: tasks, strategies, capability envelope, weight licence. Checked before any download. | [Model Registry](user-guide/registry.md) | `14_registry_and_licensing.py` |
| Leaderboard | Fit many configurations on one split and rank them, with HTML and Pareto reports. | [Leaderboard](user-guide/leaderboard.md) | `04_model_comparison.py`, `09_benchmarking.py` |
| Ensembling | Six strategies: averaging, weighted averaging, voting, stacking, deep ensembles, dynamic selection. | [Ensembling](user-guide/ensembling.md) | notebook 14 |
| Distillation | Any teacher pipeline into a LightGBM, XGBoost, CatBoost or MLP student, single or multi-teacher, ONNX export. | [Distillation](user-guide/distillation.md) | notebook 15 |
| Uncertainty | Split conformal sets and intervals, temperature recalibration, `uncertainty_report()`. | [Uncertainty](user-guide/uncertainty.md) | `18_uncertainty.py` |
| Shift-aware evaluation | Temporal, grouped and stratified-grouped splits; `ShiftEvaluator` reports the IID-to-shift gap. | [Shift-Aware Evaluation](user-guide/shift-evaluation.md) | `15_shift_aware_evaluation.py` |
| Causal inference | `CausalAnalysis` with DML, S/T/X/R-learners and causal forests, refutation and proxy auditing. | [Causal Inference](user-guide/causal.md) | notebook 16 |
| Typed configuration | Pydantic schemas for every knob, YAML round trips, `validate_against_registry()`. | [Typed Configuration](user-guide/configuration.md) | |
| Prediction caching | `cache="memory"` or `"disk"` collapses `evaluate()`'s repeated forward passes into one. | [Prediction Caching](user-guide/caching.md) | |
| Resampling | Context and support/query sampling for episodic training and oversized tables. | [Resampling](user-guide/resampling.md) | `07_data_sampling.py` |
| Fairness and calibration reports | `evaluate_fairness`, `evaluate_calibration`, `evaluate_interval_calibration`, residual analysis. | [TabularPipeline](user-guide/pipeline.md) | `08_evaluation_metrics.py` |
| One logger | `setup_logger` configures Rich, plain or JSON output and rotating files once. | [Logging](user-guide/logging.md) | `logging_showcase.py` |

## Time series

| Feature | What it does | Guide | Example |
|---|---|---|---|
| Forecasting | Point and quantile forecasts from a long frame and a `TimeSeriesSchema`. | [Time Series Overview](timeseries/overview.md) | `timeseries/quickstart.py` |
| Eleven vendored foundation models | Chronos, Chronos-Bolt, Chronos-2, TimesFM 3, Toto 1, Toto 2, TiRex, TiRex-2, Time-MoE, Timestral, T0. No model package installed. | [Time Series Models](timeseries/models.md) | `19_chronos_family.py`, `20_timesfm3_family.py`, `21_toto_family.py` |
| Fine-tuning and LoRA | `finetune` and `peft` for all eleven models. | [Time Series Fine-Tuning](timeseries/fine-tuning.md) | `timeseries/fine_tuning.py` |
| Anomaly detection, imputation, embeddings | Three more tasks from the same fitted forecaster. | [Tasks](timeseries/tasks.md) | `timeseries/anomaly_detection.py`, `timeseries/imputation.py` |
| Conformal calibration and backtesting | `calibrate()` widens or narrows intervals per horizon step; `backtest()` scores rolling origins. | [Time Series Overview](timeseries/overview.md) | `timeseries/probabilistic_and_calibration.py` |
| Covariates and multivariate targets | Past and known-future covariates, several targets per item, with a clear error when a model cannot read them. | [Time Series Overview](timeseries/overview.md) | `timeseries/covariates_and_multivariate.py` |
| Leaderboard, ensemble, benchmark, select | Compare models on backtests, combine them level by level, aggregate across datasets fev-bench style. | [Compare, Combine, Select](timeseries/comparison.md) | `timeseries/leaderboard_and_ensemble.py`, `timeseries/benchmark.py` |
| Tabular models as forecasters | `TabPFN-TS` and `TabularTS-<name>` for every tabular model with a regression head. | [Tabular Models as Forecasters](timeseries/tabular-forecasters.md) | `timeseries/tabular_forecasting.py` |
| Series histories as features | `SeriesFeaturizer` adds leak-safe summary statistics and embeddings of an entity's history to each table row. | [Series Histories as Features](timeseries/series-features.md) | `timeseries/embeddings_and_series_features.py` |
| Command line | `tabtune timeseries forecast | evaluate | anomalies | impute | embed | benchmark`. | [CLI](reference/cli.md) | |

## Possible use cases

| Use case | How |
|---|---|
| A strong baseline on a new table in a minute | `TabularPipeline("TabICL")` or `TabularPipeline("TabPFN")` in `inference` mode. |
| Beat the baseline with your own data | `tuning_strategy="peft"` with the validated loop; it keeps the pretrained weights if nothing beats them. |
| Pick a model you are allowed to ship | `license_mode="commercial"` plus `list_models(commercial_ok=True)`. |
| Compare twelve models before lunch | `TabularLeaderboard` with `cache="disk"`. |
| Serve without a GPU | `TabDistiller` into LightGBM, then ONNX export. |
| Forecast with honest intervals | `TimeSeriesPipeline("ChronosBolt").calibrate(windows=4)`. |

## Honest notes

- PEFT is experimental for TabPFN v2 and ConTextTab. Use `finetune` there if a LoRA run
  misbehaves.
- EXAONE's projections are raw parameters, so its `peft` run wraps zero adapters and
  proceeds as a full fine-tune. The registry marks this.
- Every example script downloads real weights on first use. Nothing in `tests/` does;
  the suite runs on CPU with tiny randomly initialised networks.
- See [Known Issues](community/known-issues.md) for the full list.
