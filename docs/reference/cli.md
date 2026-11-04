# CLI

The `tabtune` console script, installed with the package. Today it has one group,
`timeseries`; the tabular side is Python-only.

```bash
tabtune timeseries list-models [--task TASK] [--strategy STRATEGY] [--commercial] [--json]
tabtune timeseries info NAME
tabtune timeseries forecast  --data DATA --horizon H [--model M] [--quantiles 0.1,0.5,0.9] [--output OUT.csv]
tabtune timeseries evaluate  --data DATA --horizon H [--model M] [--quantiles ...] [--json]
tabtune timeseries anomalies --data DATA [--model M] [--alpha A] [--coverage C] [--output OUT.csv]
tabtune timeseries impute    --data DATA [--model M] [--output OUT.csv]
tabtune timeseries embed     --data DATA [--model M] [--output OUT.csv]
tabtune timeseries benchmark --models A,B,C [--data X.csv ...] [--horizon H] [--windows W] [--metric WQL] [--output DIR]
```

`-v` / `--verbose` on the group shows TabTune's progress log.

## Data options

Every data command reads a long-format CSV and shares these options:

| Option | Meaning |
|---|---|
| `--data` | The CSV. One row per item and timestamp. |
| `--target` | Target column or columns, comma-separated. Defaults to `target`. |
| `--timestamp` | Timestamp column. Defaults to `timestamp`. |
| `--item-id` | Item column. Defaults to `item_id` when present; otherwise the file is one series. |
| `--freq` | Pandas frequency. Inferred when omitted. |
| `--model` | Registry name or alias. |
| `--checkpoint` | A validated checkpoint id or a local directory. |
| `--device` | `cuda`, `cuda:0`, `cpu` or `auto`. |

## Commands

**`list-models`** prints the catalogue with tasks, strategies, context limits and licences.
`--commercial` keeps only weights cleared for commercial use. `--json` for scripts.

**`info NAME`** prints one model's registry entry: checkpoints, limits, covariate support,
licence, paper.

**`forecast`** fits the model on the whole file and writes `--horizon` steps per item,
with the requested quantiles as columns. `--output` writes a CSV; stdout otherwise.

**`evaluate`** holds out the last `--horizon` steps of every series, forecasts them from the
rest, and prints MAE, RMSE, sMAPE, MASE, pinball loss, WQL and coverage.

**`anomalies`** scores every observation and writes the score, p-value and flag per row.
`--alpha` is the false alarm rate; `--coverage` the interval coverage.

**`impute`** fills missing target values and writes the completed frame with an `imputed`
column.

**`embed`** writes one embedding row per series.

**`benchmark`** runs every model on every `--data` file (or the built-in synthetic sets when
none is given) in backtest mode and writes `raw.csv`, `leaderboard.csv` and `report.md` to
`--output`.

## Examples

```bash
tabtune timeseries list-models --commercial
tabtune timeseries forecast  --data sales.csv --model Chronos2 --horizon 14 --output forecast.csv
tabtune timeseries evaluate  --data sales.csv --model TiRex --horizon 14
tabtune timeseries anomalies --data sensors.csv --model ChronosBolt
tabtune timeseries benchmark --models Chronos2,TiRex,SeasonalNaive --data a.csv --data b.csv --horizon 24
```

Everything the CLI does is a thin layer over [`TimeSeriesPipeline`](../timeseries/overview.md)
and [`TimeSeriesBenchmark`](../timeseries/comparison.md); when you need options the CLI
does not expose, use the Python API.
