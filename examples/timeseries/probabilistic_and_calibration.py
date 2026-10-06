"""Quantile forecasts, interval coverage, and conformal calibration of the intervals.

TabularTS-GBM's quantile models under-cover on these random walks; calibrate() rescales the
intervals from backtest errors. Replace "TabularTS-GBM" with "Chronos2" for native quantiles.
"""


from tabtune.logger import get_logger, setup_logger
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES PROBABILISTIC FORECASTS AND CONFORMAL CALIBRATION")
logger.info("=" * 80)

LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]

df = make_panel(n_series=8, length=24 * 15, freq="h", kind="random_walk", seed=3)
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)

pipe = TimeSeriesPipeline(
    "TabularTS-GBM",
    forecast_params={"prediction_length": 24, "quantile_levels": LEVELS},
    model_params={"max_iter": 50},
)
pipe.fit(history, schema)
results = []


def report(label: str) -> None:
    forecast = pipe.predict()
    frame = forecast.to_pandas()
    width = (frame["0.95"] - frame["0.05"]).mean()
    metrics = pipe.evaluate(actual, forecast=forecast)
    results.append((label, metrics["coverage_90"], metrics["wql"], width))


logger.info("\n1️⃣  Scoring the raw quantile forecast...")
report("raw")
# 8 series x 4 windows = 32 calibration rows; a 90% interval needs at least 10.
logger.info("\n2️⃣  Calibrating with each conformal method...")
for method in ("cqr", "absolute", "signed"):
    pipe.calibrate(method=method, windows=4)
    report(method)

logger.info("\n" + "=" * 80)
logger.info("📊 Calibration Comparison")
logger.info("=" * 80)
for label, coverage, wql, width in results:
    logger.info(f"   {label:<10} coverage_90={coverage:.3f}  wql={wql:.4f}  width={width:.2f}")
logger.info(f"\n   Active calibration: {pipe.predict().metadata['calibration']}")
