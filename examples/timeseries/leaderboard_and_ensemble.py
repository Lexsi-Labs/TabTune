"""Rank models by backtest, combine them in an ensemble, and let select() pick one.

Uses baselines and TabularTS-GBM so it runs offline. Add "Chronos2", "ChronosBolt" or
"TimesFM3" to the model lists to compare foundation models; their weights download on first use.
"""


from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.TimeSeries import (
    TimeSeriesEnsemble,
    TimeSeriesLeaderboard,
    TimeSeriesPipeline,
    TimeSeriesSchema,
    make_panel,
    split_horizon,
)

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES LEADERBOARD, ENSEMBLE AND MODEL SELECTION")
logger.info("=" * 80)

df = make_panel(n_series=5, length=24 * 20, freq="h", kind="multi_seasonal", seed=0)
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)
forecast_params = {"prediction_length": 24}

# Backtest mode: each model is fitted without the last 2 windows and scored on them.
logger.info("\n1️⃣  Ranking models by backtest with TimeSeriesLeaderboard...")
board = TimeSeriesLeaderboard(history, schema, forecast_params=forecast_params, windows=2)
board.add_models(["SeasonalNaive", "Naive", "Drift", "WindowAverage"])
board.add_model("TabularTS-GBM", model_params={"max_iter": 50})
board.run()

logger.info("\n2️⃣  Combining models with TimeSeriesEnsemble (greedy selection)...")
ensemble = TimeSeriesEnsemble(
    ["SeasonalNaive", "Naive", "WindowAverage", {"model_name": "TabularTS-GBM", "model_params": {"max_iter": 50}}],
    ensemble_strategy="greedy_selection",
    forecast_params=forecast_params,
)
ensemble.fit(history, schema)
log_table(logger, "Ensemble leaderboard", list((ensemble.get_leaderboard()).columns), (ensemble.get_leaderboard()).itertuples(index=False, name=None))
ensemble.evaluate(actual)

logger.info("\n3️⃣  Letting TimeSeriesPipeline.select() pick a model...")
best = TimeSeriesPipeline.select(
    history,
    schema,
    forecast_params,
    candidates=["SeasonalNaive", "Naive", "Drift", "WindowAverage"],
    windows=2,
)
logger.info(f"\n✅ select() chose {best.model_name}")
log_table(logger, "Selected models", list((best.selection_[["Rank", "Model", "wql", "mase"]]).columns), (best.selection_[["Rank", "Model", "wql", "mase"]]).itertuples(index=False, name=None))
best.evaluate(actual)
