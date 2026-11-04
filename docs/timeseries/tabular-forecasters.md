# Tabular Models as Forecasters

A tabular foundation model with a regression head can forecast: build a table where each
row is one timestamp of one series with features that describe it, fit the model on the
history, predict the horizon rows. TabTune ships two designs for that table and registers
every regression-capable tabular model with both.

## `TabPFN-TS`

A re-implementation of Prior Labs' TabPFN-TS recipe on TabTune's vendored TabPFN v3.5. Each
series is a separate regression problem; the features are calendar components and Fourier
terms of the timestamp plus any known covariates. Quantiles come from the model's native
regression head.

```python
pipe = TimeSeriesPipeline("TabPFN-TS", forecast_params={"prediction_length": 24,
                                                        "quantile_levels": [0.1, 0.5, 0.9]})
pipe.fit(history, schema, future_df=future_known).predict()
```

Context up to 32,768 rows, known covariates (numeric and categorical) read, past-only
covariates not. The TabPFN 3.5 weight licence applies: non-commercial.

## `TabularTS-<name>`

Every tabular model with a regression head, as a forecaster: `TabularTS-TabPFN`,
`TabularTS-TabICLv2`, `TabularTS-Mitra`, `TabularTS-KumoTabular`, `TabularTS-LimiXv2`,
`TabularTS-TabAbax` and so on, 20 in all. Two feature designs:

```python
TimeSeriesPipeline("TabularTS-TabICLv2", model_params={"features": "time"})   # TabPFN-TS's time features, per series
TimeSeriesPipeline("TabularTS-Mitra",    model_params={"features": "lags"})   # pooled direct multi-horizon lag design, M5 style
```

With `"lags"` the series are pooled into one table of lagged values and one model predicts
every horizon step directly. Models with native quantiles (TabPFN family, TabICL v2) return
them; the others return point forecasts, and `calibrate()` can wrap intervals around those.
Each forecaster inherits its tabular model's weight licence and `commercial_use_ok`, so
`list_time_series_models(commercial_ok=True)` filters them correctly.

These are inference-only as forecasters; to fine-tune a tabular model on your series,
build the table yourself and use `TabularPipeline`.

## `TabularTS-GBM` and scikit-learn

`TabularTS-GBM` is gradient boosting on the pooled lag design, one quantile-loss model per
level, with no weights to download. It is a sensible second baseline after seasonal naive,
and it is what the lag design looks like without a foundation model. Any scikit-learn
regressor drops into the same place through `SklearnBackend`.

## When to reach for these

- Many short series with rich calendar structure: TabPFN-TS does well.
- Strong known covariates (promotions, prices, holidays): the tabular designs read them
  as columns; several foundation models cannot.
- A floor for a leaderboard that is better than seasonal naive: `TabularTS-GBM`.

The script `examples/timeseries/tabular_forecasting.py` runs the comparison.
