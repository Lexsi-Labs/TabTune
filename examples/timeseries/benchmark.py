"""Benchmark forecasters over several datasets: skill against seasonal naive, win rates and paired tests.

Uses the built-in synthetic datasets, which only check that the machinery works; they say nothing
about real-data accuracy. Add "Chronos2" or "ChronosBolt" to MODELS to benchmark foundation models.
"""

import tempfile
from pathlib import Path

from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.TimeSeries import TimeSeriesBenchmark, TimeSeriesSchema, make_panel

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES BENCHMARK: Skill, Win Rates and Paired Tests")
logger.info("=" * 80)

MODELS = ["Naive", "Drift", {"model_name": "TabularTS-GBM", "model_params": {"max_iter": 50}, "label": "GBM"}]

logger.info("\n1️⃣  Benchmarking on the built-in synthetic datasets...")
results = TimeSeriesBenchmark(MODELS, windows=2).run()
log_table(logger, "📊 Leaderboard (MASE)", list((results.leaderboard("mase")).columns), (results.leaderboard("mase")).itertuples(index=False, name=None))
log_table(logger, "📊 Paired tests (MASE)", list((results.pairwise_tests("mase")).columns), (results.pairwise_tests("mase")).itertuples(index=False, name=None))
log_table(logger, "📊 Per-dataset scores (MASE)", list((results.task_scores("mase").round(3).reset_index()).columns), (results.task_scores("mase").round(3).reset_index()).itertuples(index=False, name=None))

# Your own data: any long frame with a schema and a horizon per dataset.
own = {
    "daily_trend": {
        "df": make_panel(n_series=6, length=200, freq="D", kind="trend", season=7, seed=4),
        "schema": TimeSeriesSchema(target="target", item_id="item_id"),
        "prediction_length": 14,
    }
}
logger.info("\n2️⃣  Benchmarking on your own dataset...")
single = TimeSeriesBenchmark(["Naive", "Drift"], own, windows=2).run()
log_table(logger, "Single-dataset results", list((single.raw[["dataset", "model", "status", "mase", "wql"]]).columns), (single.raw[["dataset", "model", "status", "mase", "wql"]]).itertuples(index=False, name=None))

with tempfile.TemporaryDirectory() as directory:
    results.save(directory, metric="mase")
    logger.info(f"\n💾 Saved files: {sorted(path.name for path in Path(directory).iterdir())}")
