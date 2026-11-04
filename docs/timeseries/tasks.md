# Time Series Tasks

One fitted forecaster serves four tasks. `task_type` picks the task; `predict()` returns a
`ForecastResult`, an `AnomalyResult`, an `ImputationResult` or an `EmbeddingResult`.

```python
from tabtune.TimeSeries import TimeSeriesPipeline

TimeSeriesPipeline("ChronosBolt", task_type="anomaly_detection",
                   task_params={"method": "interval"}).fit(df, schema).predict()
TimeSeriesPipeline("ChronosBolt", task_type="imputation").fit(df_with_gaps, schema).predict()
TimeSeriesPipeline("TiRex", task_type="embedding").fit(df, schema).predict().embeddings
```

## Forecasting

The default. `forecast_params["prediction_length"]` is required; `quantile_levels` asks
for quantiles; `context_length` caps the history each forecast reads. The result's
`to_pandas()` is the stable public layout: one row per `(item, target, timestamp)` with
`point` and one column per quantile level.

```python
pipe = TimeSeriesPipeline("Chronos2", forecast_params={"prediction_length": 24,
                                                       "quantile_levels": [0.05, 0.5, 0.95]})
result = pipe.fit(history, schema).predict()
result.point            # [row, horizon], read-only
result.quantiles        # [row, horizon, level]
result.metadata["point_forecast"]   # "mean" or "median"
```

## Anomaly detection

Each observation is scored against rolling forecasts made from the history before it, and
the scores become conformal p-values against a reference period. `alpha` is the false
alarm rate when the reference is clean.

| `task_params` key | Default | Meaning |
|---|---|---|
| `method` | `"forecast_error"` | `"forecast_error"`, `"interval"` (distance outside the forecast interval) or `"likelihood"`. |
| `stride` | 1 | Steps between forecast origins. Larger is faster and coarser. |
| `min_context` | model-specific | Observations before scoring starts. |
| `coverage` | 0.98 | Interval coverage for `"interval"`. |
| `alpha` | 0.01 | False alarm rate for the conformal threshold. |
| `threshold` | `"conformal"` | `"conformal"`, `"contamination"` or a number. |
| `contamination` | - | Expected anomaly fraction for the `"contamination"` threshold. |
| `reference_fraction` | - | Share of the history used as the clean reference. |

`forecast_params["prediction_length"]` is the number of steps forecast from each origin
(default 1).

```python
res = TimeSeriesPipeline("ChronosBolt", task_type="anomaly_detection",
                         task_params={"alpha": 0.02}).fit(df, schema).predict()
res.anomalies()                   # only the flagged rows
res.evaluate(labels=df["is_anomaly"])   # AUROC, AUPRC, precision, recall, F1
```

Evaluate on labelled data before trusting a detector. `make_anomalous_panel()` makes a
seeded panel with known point anomalies for exactly that.

## Imputation

Each gap is filled by forecasting forward from the values before it and, where there are
enough, backward from the values after it (the reversed series), blending the two across
the gap. `method="forecast"` uses the forward pass only. The result keeps the schema's
columns with the targets filled, plus an `imputed` flag and, for a single target, a 10 to
90 percent band of the fill.

```python
filled = TimeSeriesPipeline("Chronos2", task_type="imputation").fit(df_with_gaps, schema).predict()
filled.to_pandas()[["item_id", "timestamp", "target", "imputed", "lower", "upper"]]
```

## Embeddings

One vector per `(item, target)` series from the model's encoder.

```python
emb = TimeSeriesPipeline("TiRex", task_type="embedding").fit(df, schema).predict()
emb.embeddings          # [row, dim]
emb.similarity()        # [row, row] cosine similarity
emb.to_pandas()         # item column, target, emb_0 ... emb_<dim-1>
```

These are what [`SeriesFeaturizer`](series-features.md) adds to a tabular frame.

## From the command line

```bash
tabtune timeseries anomalies --data sensors.csv --model ChronosBolt --alpha 0.02
tabtune timeseries impute    --data sales.csv   --model Chronos2 --output filled.csv
tabtune timeseries embed     --data sales.csv   --model TiRex    --output emb.csv
```

See [CLI](../reference/cli.md).
