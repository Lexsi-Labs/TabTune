"""Command line interface: ``tabtune timeseries <command>``.

Commands:

* ``list-models``: registered models with tasks, strategies and license
* ``info NAME``: one model's registry entry as JSON
* ``forecast``: forecast a long-format CSV and write the forecasts as CSV
* ``evaluate``: hold out the last ``--horizon`` steps of a CSV and score them
* ``anomalies``: score every observation of a CSV and write the scores
* ``impute``: fill the missing target values of a CSV
* ``embed``: one embedding vector per series of a CSV
* ``benchmark``: compare models on CSVs or the built-in synthetic datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

__all__ = ["main", "build_parser"]


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _data_args(parser: argparse.ArgumentParser, *, horizon: bool = True) -> None:
    parser.add_argument("--data", required=True, help="long-format CSV")
    parser.add_argument("--target", default="target", help="target column(s), comma-separated")
    parser.add_argument("--timestamp", default="timestamp", help="timestamp column")
    parser.add_argument("--item-id", dest="item_id", help="item column (default: item_id if present)")
    parser.add_argument("--freq", help="pandas frequency (inferred if omitted)")
    parser.add_argument("--model", default="SeasonalNaive")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    if horizon:
        parser.add_argument("--horizon", type=int, required=True)
        parser.add_argument("--quantiles", type=_floats, default=None, help="e.g. 0.1,0.5,0.9")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabtune timeseries", description="Time series models in TabTune.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show TabTune's progress log")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-models", help="list registered models")
    p.add_argument("--task")
    p.add_argument("--strategy")
    p.add_argument("--commercial", action="store_true", help="only weights cleared for commercial use")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("info", help="show one model's registry entry")
    p.add_argument("name")

    p = sub.add_parser("forecast", help="forecast a CSV")
    _data_args(p)
    p.add_argument("--output", help="output CSV (stdout if omitted)")

    p = sub.add_parser("evaluate", help="hold out the last --horizon steps of a CSV and score")
    _data_args(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("anomalies", help="score every observation of a CSV")
    _data_args(p, horizon=False)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--coverage", type=float, help="interval coverage (default: 0.98, or the widest the model predicts)")
    p.add_argument("--output", help="output CSV (stdout if omitted)")

    p = sub.add_parser("impute", help="fill missing target values of a CSV")
    _data_args(p, horizon=False)
    p.add_argument("--output", help="output CSV (stdout if omitted)")

    p = sub.add_parser("embed", help="embed every series of a CSV")
    _data_args(p, horizon=False)
    p.add_argument("--output", help="output CSV (stdout if omitted)")

    p = sub.add_parser("benchmark", help="compare models on CSVs or built-in datasets")
    p.add_argument("--models", type=_names, required=True, help="comma-separated model names")
    p.add_argument("--data", action="append", default=[], help="CSV path (repeatable)")
    p.add_argument("--horizon", type=int, help="horizon for --data CSVs")
    p.add_argument("--target", default="target")
    p.add_argument("--timestamp", default="timestamp")
    p.add_argument("--item-id", dest="item_id")
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--metric", default="mase")
    p.add_argument("--output", help="directory for raw.csv, leaderboard.csv and report.md")
    return parser


def _read(args: argparse.Namespace) -> tuple[Any, Any]:
    import pandas as pd

    from .schema import TimeSeriesSchema

    try:
        columns = list(pd.read_csv(args.data, nrows=0).columns)
        item_id = args.item_id or ("item_id" if "item_id" in columns else None)
        frame = pd.read_csv(args.data, converters={item_id: str} if item_id else None)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SystemExit(f"cannot read {args.data}: {exc}") from exc
    targets = _names(args.target)
    missing = [c for c in (*targets, args.timestamp, *([item_id] if item_id else [])) if c not in frame.columns]
    if missing:
        raise SystemExit(f"{args.data} has no column(s) {missing}; columns are {list(frame.columns)}")
    schema = TimeSeriesSchema(
        target=targets[0] if len(targets) == 1 else targets,
        timestamp=args.timestamp,
        item_id=item_id,
        freq=getattr(args, "freq", None),
    )
    return frame, schema


def _pipeline(args: argparse.Namespace, task: str, **forecast: Any) -> Any:
    from .pipeline import TimeSeriesPipeline

    model_params = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    params = {k: v for k, v in forecast.items() if v is not None}
    return TimeSeriesPipeline(
        args.model,
        task_type=task,
        tuning_params={"device": args.device},
        model_params=model_params,
        forecast_params=params or None,
    )


def _write(frame: Any, output: str | None) -> None:
    if output:
        frame.to_csv(output, index=False)
        print(f"wrote {len(frame)} rows to {output}", file=sys.stderr)
    else:
        frame.to_csv(sys.stdout, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    logger = logging.getLogger("tabtune")
    level = logger.level
    if not args.verbose:
        logger.setLevel(logging.WARNING)
    try:
        return _run(args)
    finally:
        logger.setLevel(level)


def _run(args: argparse.Namespace) -> int:

    if args.command == "list-models":
        import pandas as pd

        from ..registry.TimeSeries import list_time_series_models

        specs = list_time_series_models(
            task=args.task, strategy=args.strategy, commercial_ok=True if args.commercial else None
        )
        rows = [
            {
                "model": s.name,
                "family": s.family,
                "tasks": ",".join(sorted(s.tasks)),
                "strategies": ",".join(sorted(s.strategies)),
                "multivariate": s.supports_multivariate,
                "covariates": s.supports_covariates,
                "license": s.license.name,
                "commercial": s.license.badge,
            }
            for s in specs
        ]
        frame = pd.DataFrame(rows)
        print(frame.to_json(orient="records", indent=2) if args.json else frame.to_string(index=False))
        return 0

    if args.command == "info":
        from ..registry.TimeSeries import get_time_series_model_spec

        print(json.dumps(get_time_series_model_spec(args.name).to_dict(), indent=2))
        return 0

    if args.command == "benchmark":
        from .benchmark import TimeSeriesBenchmark
        from .schema import TimeSeriesSchema

        datasets = None
        if args.data:
            if args.horizon is None:
                print("--horizon is required with --data", file=sys.stderr)
                return 2
            import pandas as pd

            datasets = {}
            for path in args.data:
                frame = pd.read_csv(path)
                item_id = args.item_id or ("item_id" if "item_id" in frame.columns else None)
                schema = TimeSeriesSchema(target=args.target, timestamp=args.timestamp, item_id=item_id)
                datasets[path] = {"df": frame, "schema": schema, "prediction_length": args.horizon}
        results = TimeSeriesBenchmark(args.models, datasets, windows=args.windows).run()
        print(results.leaderboard(args.metric).to_string(index=False))
        if args.output:
            results.save(args.output, metric=args.metric)
            print(f"wrote results to {args.output}", file=sys.stderr)
        return 0

    frame, schema = _read(args)
    if args.command == "forecast":
        pipe = _pipeline(args, "forecasting", prediction_length=args.horizon, quantile_levels=args.quantiles)
        _write(pipe.fit(frame, schema).predict().to_pandas(), args.output)
        return 0
    if args.command == "evaluate":
        from .data import split_horizon

        history, actual = split_horizon(frame, schema, args.horizon)
        pipe = _pipeline(args, "forecasting", prediction_length=args.horizon, quantile_levels=args.quantiles)
        metrics = pipe.fit(history, schema).evaluate(actual, output_format="json" if args.json else "rich")
        if not args.json:
            from .pipeline import _metric_label

            for key, value in metrics.items():
                if key != "per_target":
                    print(f"{_metric_label(key):>20}: {value:.4f}")
            for name, target_metrics in metrics.get("per_target", {}).items():
                print(f"\nTarget '{name}'")
                for key, value in target_metrics.items():
                    print(f"{_metric_label(key):>20}: {value:.4f}")
        return 0
    if args.command == "anomalies":
        from .pipeline import TimeSeriesPipeline

        pipe = TimeSeriesPipeline(
            args.model,
            task_type="anomaly_detection",
            tuning_params={"device": args.device},
            model_params={"checkpoint": args.checkpoint} if args.checkpoint else {},
            task_params={"alpha": args.alpha, **({"coverage": args.coverage} if args.coverage else {})},
        )
        _write(pipe.fit(frame, schema).predict().to_pandas(), args.output)
        return 0
    if args.command == "embed":
        pipe = _pipeline(args, "embedding")
        _write(pipe.fit(frame, schema).predict().to_pandas(), args.output)
        return 0
    pipe = _pipeline(args, "imputation")
    _write(pipe.fit(frame, schema).predict().to_pandas(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
