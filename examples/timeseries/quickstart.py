"""Forecast a panel of hourly series, score the forecast on held-out data and backtest it.

Runs offline with the SeasonalNaive baseline. Set MODEL to "Chronos2", "ChronosBolt" or
"TimesFM3" to use a foundation model; its weights download on first use.
"""


from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

MODEL = "SeasonalNaive"

logger.info("=" * 80)
logger.info("TIME SERIES QUICKSTART: Forecast, Evaluate and Backtest")
logger.info("=" * 80)

df = make_panel(n_series=4, length=24 * 14, freq="h", seed=0)
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)

logger.info(f"\n📊 Data: {df['item_id'].nunique()} hourly series, {len(history)} history rows, {len(actual)} held-out rows")

logger.info("\n1️⃣  Fitting and forecasting...")
pipe = TimeSeriesPipeline(MODEL, forecast_params={"prediction_length": 24})
pipe.fit(history, schema)
forecast = pipe.predict()
log_table(logger, "Forecast (first rows)", list((forecast.to_pandas().head()).columns), (forecast.to_pandas().head()).itertuples(index=False, name=None))
logger.info(f"   Forecast metadata: {forecast.metadata}")

logger.info("\n2️⃣  Scoring the forecast on held-out data...")
pipe.evaluate(actual, forecast=forecast)

logger.info("\n3️⃣  Backtesting over 3 rolling windows...")
pipe.backtest(windows=3)
