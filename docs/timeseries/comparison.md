# Compare, Combine, Select

Three objects and one class method, mirroring the tabular side. Members are
`TimeSeriesPipeline` configurations, so foundation models, tabular forecasters and
statistical baselines mix freely.

## `TimeSeriesLeaderboard`

```python
from tabtune.TimeSeries import TimeSeriesLeaderboard

board = TimeSeriesLeaderboard(history, schema, forecast_params={"prediction_length": 24}, windows=2)
board.add_models(["SeasonalNaive", "Chronos2", "TiRex", "TabularTS-GBM"])
board.add_model("TiRex", tuning_strategy="peft", tuning_params={"epochs": 2}, label="TiRex LoRA")
board.add_all(commercial_ok=True)        # every registered model that can read the schema
board.run(rank_by="WQL")
board.best()
board.to_markdown("leaderboard.md")
```

Two modes. **Backtest** (default): every configuration is fitted on `df` without its last
`windows` forecast windows and scored on rolling-origin forecasts over them. **Holdout**:
pass `df_actual`, and every configuration is fitted on `df` and scored on the observations
that follow it. Each entry records the metrics, fit and predict time, the weight licence
and, when it failed, the error and the traceback; failed entries rank last instead of
hiding.

## `TimeSeriesEnsemble`

```python
from tabtune.TimeSeries import TimeSeriesEnsemble

ens = TimeSeriesEnsemble(["ChronosBolt", "TiRex", "SeasonalNaive"],
                         ensemble_strategy="greedy_selection",
                         forecast_params={"prediction_length": 24}, validation_windows=3)
ens.fit(history, schema).predict()
ens.weights_                 # by member label
ens.get_leaderboard()        # each member's validation loss, weight and fit time
```

Quantiles are combined level by level (Vincentization): the ensemble's q-quantile is a
weighted average of the members' q-quantiles. Weights are learned on validation windows at
the end of the history.

| Strategy | Weights |
|---|---|
| `mean`, `median` | Uniform, or the per-level median. No fitting. |
| `weighted_averaging` | Proportional to one over the validation loss. |
| `greedy_selection` | Caruana et al. ensemble selection with replacement, as in AutoGluon-TimeSeries. The default. |
| `stacking` | Non-negative least squares on the point forecasts. |
| `best` | All weight on the best member. |

The validation loss is WQL when quantiles are available and WAPE otherwise.

## `TimeSeriesBenchmark`

Several configurations over several datasets, aggregated the way fev-bench does:

- **skill** = `1 - geomean(clip(error / baseline error, 0.01, 100))` over tasks, relative
  to seasonal naive, with a bootstrap interval;
- **win rate**: the fraction of tasks a model beats each other model, ties counting one
  half, averaged over opponents;
- **pairwise tests**: Wilcoxon signed-rank tests on task errors with Holm's correction.

A task counts towards the ranking only if every model succeeded on it.

```python
from tabtune.TimeSeries import TimeSeriesBenchmark

bench = TimeSeriesBenchmark(["Chronos2", "TiRex", "SeasonalNaive"], datasets=None)  # None: built-in synthetic sets
results = bench.run()
results.leaderboard(metric="WQL")
results.pairwise_tests()
results.save("bench_out")       # raw.csv, leaderboard.csv, report.md, config.json
```

The built-in datasets are seeded synthetic panels for smoke tests, not a substitute for
GIFT-Eval or fev-bench. Pass your own as
`{"name": {"df": ..., "schema": ..., "prediction_length": ...}}`.

## `TimeSeriesPipeline.select`

```python
best = TimeSeriesPipeline.select(history, schema, {"prediction_length": 24},
                                 candidates=["Chronos2", "TiRex", "SeasonalNaive"],
                                 windows=3, rank_by="WQL", license_mode="commercial")
best.predict()
```

Backtests the candidates and returns the best one already fitted on the full history.
`license_mode="commercial"` drops candidates you could not ship before spending time on
them.

## From the command line

```bash
tabtune timeseries benchmark --models Chronos2,TiRex,SeasonalNaive --data a.csv --data b.csv --horizon 24 --output bench_out
```
