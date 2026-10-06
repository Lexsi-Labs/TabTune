"""Tabular models as forecasters: TabularTS-GBM, TabularTS-XRFM and a custom scikit-learn regressor.

Everything here trains from scratch on the context, so nothing is downloaded. Replace
"TabularTS-XRFM" with "TabularTS-TabICLv2" or "TabPFN-TS" to use a tabular foundation model.
"""

import logging

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from tabtune.logger import get_logger, setup_logger
from tabtune.models.TimeSeries.tabular import SklearnBackend
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")
# TabularTS-XRFM fits a TabularPipeline on every forecast; keep its per-fit logs quiet.
for name in ("tabtune.TabularPipeline", "tabtune.models.regression"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger.info("=" * 80)
logger.info("TABULAR MODELS AS TIME SERIES FORECASTERS")
logger.info("=" * 80)

df = make_panel(n_series=5, length=24 * 10, freq="h", seed=0)
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)
forecast_params = {"prediction_length": 24}
results = []


def score(label: str, name: str, **model_params) -> TimeSeriesPipeline:
    pipe = TimeSeriesPipeline(name, forecast_params=forecast_params, model_params=model_params)
    forecast = pipe.fit(history, schema).predict()
    metrics = pipe.evaluate(actual, forecast=forecast)
    kind = "quantiles" if forecast.quantiles is not None else "point only"
    results.append((label, metrics["mase"], kind))
    return pipe


logger.info("\n1️⃣  Fitting each tabular forecaster...")
score("GBM, lag design", "TabularTS-GBM")
score("GBM, time design", "TabularTS-GBM", features="time")
xrfm = score("xRFM, time design", "TabularTS-XRFM")
# Lag features have gaps at the start of each series; Ridge needs them imputed.
ridge = SklearnBackend(lambda: make_pipeline(SimpleImputer(), Ridge(alpha=1.0)))
score("Ridge, lag design", "TabularTS-GBM", backend=ridge)

logger.info("\n" + "=" * 80)
logger.info("📊 Tabular Forecaster Comparison")
logger.info("=" * 80)
for label, mase, kind in results:
    logger.info(f"   {label:<22} mase={mase:.3f}  {kind}")

logger.info("\n2️⃣  Calibrating xRFM intervals from backtest windows...")
# xRFM has no quantile head, so intervals come from conformal calibration on backtest windows.
xrfm.calibrate(method="absolute", windows=4)
forecast = xrfm.predict()
xrfm.evaluate(actual, forecast=forecast)
logger.info(f"   Active calibration: {forecast.metadata['calibration']}")
