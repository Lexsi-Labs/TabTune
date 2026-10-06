"""Offline presentation fixtures; displayed scores are illustrative, not benchmarks.

Run from a checkout with PYTHONPATH=. python examples/logging_showcase.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabtune.logger import get_logger, log_context, log_event, log_metrics, log_table, setup_logger


def showcase(domain: str) -> None:
    """Emit fixed records through the same renderer used by both pipelines."""
    ts = domain == "timeseries"
    logger = get_logger("TimeSeries.pipeline" if ts else "TabularPipeline.pipeline")
    model = "ChronosBolt" if ts else "TabICLv2"
    with log_context(domain=domain, model=model, run_id="demo", operation="fit"):
        log_event(logger, "stage_started", "Fit started")
        if ts:
            log_event(
                logger,
                "data_validated",
                "History validated",
                series=24,
                frequency="D",
                context_length=512,
                horizon=28,
            )
            log_event(
                logger, "stage_completed", "Fit complete", elapsed_s=2.84, training="zero-shot"
            )
        else:
            log_event(logger, "data_received", "Tabular data received", rows=12000, features=36)
            log_event(logger, "stage_completed", "Fit complete", elapsed_s=1.62)
    with log_context(domain=domain, model=model, run_id="demo", operation="evaluate"):
        if ts:
            log_metrics(
                logger, {"MASE": 0.7312, "RMSE": 12.418, "WQL": 0.0826}, title="Forecast evaluation"
            )
        else:
            log_metrics(
                logger,
                {"ROC AUC": 0.9412, "Weighted F1": 0.8973, "Accuracy": 0.9125},
                title="Classification evaluation",
            )
    if ts:
        with log_context(domain=domain, model=model, run_id="demo", operation="backtest"):
            log_event(
                logger, "progress", "Backtest windows scored", completed=5, total=5, elapsed_s=8.43
            )
    with log_context(domain=domain, run_id="demo", operation="leaderboard"):
        log_table(
            logger,
            "Leaderboard / illustrative values",
            ["Model", "Status", "MASE" if ts else "ROC AUC", "fit_s"],
            [
                (model, "ok", 0.7312 if ts else 0.9412, 2.84 if ts else 1.62),
                (
                    "SeasonalNaive" if ts else "TabPFNv35",
                    "ok",
                    1.0 if ts else 0.9298,
                    0.01 if ts else 2.71,
                ),
            ],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["tabular", "timeseries", "both"], default="both")
    parser.add_argument("--format", choices=["auto", "rich", "plain", "json"], default="rich")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--export", type=Path, help="Save actual Rich output as HTML and SVG using this path stem"
    )
    args = parser.parse_args()
    if args.export and args.format != "rich":
        parser.error("--export requires --format rich")
    logger = setup_logger(console_format=args.format, log_file=args.log_file, file_format="json")
    rich_console = next((h.console for h in logger.handlers if hasattr(h, "console")), None)
    if args.export:
        if rich_console is None:
            parser.error("Rich is required for --export")
        rich_console.record = True
        rich_console.width = 92
    logger.warning(
        "Presentation demo: all model scores and timings below are illustrative fixtures."
    )
    for domain in ["tabular", "timeseries"] if args.domain == "both" else [args.domain]:
        showcase(domain)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        rich_console.save_html(str(args.export.with_suffix(".html")), clear=False)
        rich_console.save_svg(
            str(args.export.with_suffix(".svg")),
            title="TabTune logging / presentation demo",
            clear=False,
        )


if __name__ == "__main__":
    main()
