# Time Series

`TimeSeriesPipeline` is `TabularPipeline` for sequences: name a model, a task and a tuning
strategy, call `fit`, then `predict`. The data is a long frame with one row per series and
timestamp, and a `TimeSeriesSchema` names the role of each column. Eleven time series
foundation models are vendored, every tabular model with a regression head doubles as a
forecaster, and five statistical baselines give every comparison a floor.

```python
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

df = make_panel(n_series=5, length=400, freq="h")           # or your own long frame
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)

pipe = TimeSeriesPipeline("ChronosBolt", forecast_params={"prediction_length": 24})
forecast = pipe.fit(history, schema).predict()
forecast.to_pandas()      # item_id, timestamp, target, point, 0.1, 0.5, 0.9
pipe.evaluate(actual)     # MAE, RMSE, sMAPE, MASE, pinball, WQL, coverage
pipe.backtest(windows=3)  # rolling-origin scores
pipe.calibrate(windows=4) # conformal intervals, for any model
```

Swap `"ChronosBolt"` for `"TiRex"`, `"TabPFN-TS"` or `"SeasonalNaive"` and nothing else
changes.

## The input contract

Long format: one row per `(item, timestamp)` with the target in its own column. Extra
targets and covariates are extra columns, named by the schema.

```python
schema = TimeSeriesSchema(
    target=["load", "price"],          # one column, or several for a multivariate forecast
    timestamp="timestamp",             # anything pandas.to_datetime accepts
    item_id="site",                    # None treats the frame as a single series
    freq="h",                          # None infers it and requires every series to agree
    past_covariates=["temperature"],   # observed over the history only
    known_covariates=["promo"],        # also known over the horizon; supplied via future_df
)
pipe = TimeSeriesPipeline("Chronos2", forecast_params={"prediction_length": 24})
pipe.fit(history, schema, future_df=future_promo)
pipe.evaluate(actual)["per_target"]
```

A model that cannot read a schema feature rejects it with the models that can named in the
error. `NaN` in the history is allowed for every model except Time-MoE, which the registry
records.

## The constructor

```python
TimeSeriesPipeline(
    model_name,                      # registry name or alias
    task_type="forecasting",         # "anomaly_detection" | "imputation" | "embedding"
    tuning_strategy="inference",     # "finetune" | "peft"
    tuning_params=None,              # epochs, learning_rate, batch_size, peft_config ...
    model_params=None,               # checkpoint, device, adapter options
    forecast_params=None,            # prediction_length (required), quantile_levels, context_length
    task_params=None,                # anomaly / imputation options
    cache=None, envelope_mode="warn", license_mode="research", validate=True,
)
```

`model_params["checkpoint"]` picks the weights: one of the validated checkpoint ids for the
model (the registry lists them) or a local directory. `forecast_params["quantile_levels"]`
asks for quantiles; each adapter maps them onto what its model produces, interpolating
inside the grid, clamping outside it with a warning, or refusing a level the model never
produced. Quantiles are sorted before they reach you, and
`forecast.metadata["point_forecast"]` says whether the point is a mean or a median.

## What the pipeline guarantees

- **Checks before downloads.** The registry confirms the model supports the task and the
  strategy, `license_mode="commercial"` refuses non-commercial weights, and
  `envelope_mode` warns or errors when the horizon exceeds what the checkpoint was
  documented for.
- **No configuration is scored on data it was fitted on.** `backtest`, the leaderboard and
  the ensemble train without the scored windows.
- **Covariates a model cannot read are an error**, never silently dropped.
- **Quantiles mean what they say**, as above.

## Methods

| Method | What it does |
|---|---|
| `fit(df, schema, future_df=None)` | Validate the history, load the model, train it for `finetune` and `peft`. |
| `predict(df=None, future_df=None)` | The task's result for the fitted history, or for a new `df`. |
| `evaluate(df_actual, forecast=None, history=None)` | MAE, RMSE, sMAPE, MASE, pinball loss, WQL and coverage against observed values; `per_target` for multivariate schemas. |
| `backtest(windows, step=None)` | Rolling-origin forecasts over the end of the history, scored. |
| `calibrate(method="cqr", windows=4)` | Split conformal calibration of the quantiles, one threshold per horizon step. Methods `cqr`, `absolute`, `signed`. |
| `tune_context_length(candidates, windows)` | Choose `context_length` by backtest and keep the best. |
| `select(df, schema, forecast_params, candidates, ...)` | Class method: backtest the candidates and return the best one, fitted. |
| `save(path)` / `load(path)` | Persist the fitted pipeline, adapters included. |

Conformal coverage on time series is approximate because consecutive windows are not
exchangeable; `calibrate()` says so in its metadata and `evaluate()` reports the realised
coverage so you can check.

## Where to next

- [Models](models.md): the eleven foundation models, the tabular forecasters and the
  baselines, with context limits, covariate support and licences.
- [Tasks](tasks.md): anomaly detection, imputation and embeddings from the same fitted
  forecaster.
- [Fine-Tuning](fine-tuning.md): full and LoRA fine-tuning of the foundation models.
- [Compare, Combine, Select](comparison.md): leaderboard, ensemble, benchmark and
  `select()`.
- [Tabular Models as Forecasters](tabular-forecasters.md) and
  [Series Histories as Features](series-features.md): the two bridges between the tabular
  and the time series sides.
- [CLI](../reference/cli.md): the same operations from the shell.
