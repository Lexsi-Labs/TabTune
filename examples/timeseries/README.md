# Time series examples

Short scripts for `tabtune.TimeSeries`. Each one builds seeded synthetic data with
`make_panel`, runs offline on a CPU in well under two minutes, and prints its results.

| Script | Shows |
|---|---|
| `quickstart.py` | `TimeSeriesPipeline` fit, predict, `evaluate` on held-out data, rolling `backtest` |
| `covariates_and_multivariate.py` | past and known covariates with `future_df`; two targets forecast jointly by Chronos-2 |
| `probabilistic_and_calibration.py` | quantile levels, interval coverage, `calibrate()` with `cqr`, `absolute` and `signed` |
| `fine_tuning.py` | Time-MoE `finetune` and `peft` (LoRA), early stopping, the training report, `save` and `load` |
| `anomaly_detection.py` | the three scoring methods, conformal flags, `AnomalyResult.evaluate`, a rolling median baseline |
| `imputation.py` | bidirectional and forward gap filling, compared with linear interpolation |
| `embeddings_and_series_features.py` | series embeddings; `SeriesFeaturizer` history features for a churn classifier, with a leakage check |
| `leaderboard_and_ensemble.py` | `TimeSeriesLeaderboard` in backtest and holdout mode, `TimeSeriesEnsemble`, `TimeSeriesPipeline.select` |
| `tabular_forecasting.py` | `TabularTS-GBM`, `TabularTS-XRFM`, the `time` and `lags` designs, a custom scikit-learn backend |
| `benchmark.py` | `TimeSeriesBenchmark`: skill, win rate, paired tests, your own datasets, saved reports |

## Running

From the repository root, with TabTune installed (`pip install -e .`):

```bash
python examples/timeseries/quickstart.py
```

No extra package is needed: the model code for every registered model is part of TabTune.

## What runs offline, and what the numbers mean

The scripts use models that need no download:

- statistical baselines (`SeasonalNaive`, `Naive`, `Drift`, `WindowAverage`);
- `TabularTS-GBM` (scikit-learn gradient boosting) and `TabularTS-XRFM` (trains from scratch);
- tiny, randomly initialised Time-MoE and Chronos-2 checkpoints that the scripts build and save
  to a temporary directory. These exercise the real model code, but their forecasts and
  embeddings carry no learned structure, so do not read anything into their scores.

## Using a foundation model

Change the model name. For example, in `quickstart.py` set `MODEL = "Chronos2"`. In the scripts
that build a tiny checkpoint, remove `model_params={"checkpoint": ...}` to load the released
weights (`amazon/chronos-2`, `Maple728/TimeMoE-50M`). Weights download from the Hugging Face Hub
on first use and are cached; `HF_HUB_OFFLINE=1` then reads the cache without network calls.

Check the license before you use a model's weights: `tabtune timeseries list-models` prints it,
and `license_mode="commercial"` makes the pipeline refuse weights that forbid commercial use. See
`docs/timeseries/models.md`.

## Logging

TabTune logs each fit and forecast at `INFO`. The scripts lower that to `WARNING` with
`logging.getLogger("tabtune").setLevel(logging.WARNING)`; remove the line to see the log.
