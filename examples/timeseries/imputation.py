"""Fill gaps in hourly series from forecasts, and compare with linear interpolation.

Runs offline with SeasonalNaive. Replace it with "Chronos2" or "ChronosBolt" to fill gaps
from a foundation model's forecasts.
"""


import numpy as np

from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES IMPUTATION: Forecast-Based Gap Filling vs Linear Interpolation")
logger.info("=" * 80)

truth = make_panel(n_series=3, length=24 * 14, freq="h", seed=1)
gappy = truth.copy()
rng = np.random.default_rng(0)
for rows in gappy.groupby("item_id").indices.values():
    for start in rng.choice(rows[48:-24], size=2, replace=False):
        gappy.loc[start : start + 11, "target"] = np.nan
missing = gappy["target"].isna().to_numpy()
logger.info(f"\n📊 {missing.sum()} of {len(gappy)} values removed")

schema = TimeSeriesSchema(target="target", item_id="item_id")
logger.info("\n1️⃣  Filling the gaps with each imputation method...")
for method in ("bidirectional", "forecast"):
    pipe = TimeSeriesPipeline("SeasonalNaive", task_type="imputation", task_params={"method": method})
    result = pipe.fit(gappy, schema).predict()
    filled = result.to_pandas()
    error = np.abs(filled["target"].to_numpy() - truth["target"].to_numpy())[missing].mean()
    logger.info(f"   {method:<14} {result}  MAE on the gaps {error:.3f}")

logger.info("\n2️⃣  Imputed rows:")
log_table(logger, "Imputed rows (first rows)", list((filled[filled["imputed"]].head()).columns), (filled[filled["imputed"]].head()).itertuples(index=False, name=None))

linear = gappy.groupby("item_id")["target"].transform(lambda s: s.interpolate(limit_direction="both"))
error = np.abs(linear.to_numpy() - truth["target"].to_numpy())[missing].mean()
logger.info("\n3️⃣  Linear interpolation baseline")
logger.info(f"   {'linear':<14} MAE on the gaps {error:.3f}")
