"""Past and known covariates with TabularTS-GBM, then a joint forecast of two targets with Chronos-2.

The Chronos-2 part builds a tiny randomly initialised checkpoint so that it runs offline, so its
numbers mean nothing. Remove model_params to use the released amazon/chronos-2 weights.
"""

import tempfile

import torch

from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.models.chronos import Chronos2Model
from tabtune.models.chronos.chronos2.config import Chronos2CoreConfig
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES COVARIATES AND MULTIVARIATE FORECASTING")
logger.info("=" * 80)


def tiny_chronos2_checkpoint(directory: str) -> str:
    """Save a 2-layer Chronos-2 with random weights in the Hugging Face layout."""
    config = Chronos2CoreConfig(d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)
    config.chronos_config = {
        "context_length": 64,
        "input_patch_size": 8,
        "input_patch_stride": 8,
        "output_patch_size": 8,
        "max_output_patches": 2,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "use_reg_token": True,
        "use_arcsinh": True,
    }
    config.architectures = ["Chronos2Model"]
    torch.manual_seed(0)
    Chronos2Model(config).save_pretrained(directory)
    return directory


# "promo" is a 0/1 flag known in advance; "temperature" is only observed up to the forecast origin.
df = make_panel(n_series=4, length=24 * 21, freq="h", seed=0, covariates=["promo", "temperature"])
with_covariates = TimeSeriesSchema(
    target="target",
    item_id="item_id",
    past_covariates=["temperature"],
    known_covariates=["promo"],
)
target_only = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, with_covariates, prediction_length=24)
future_df = actual[["item_id", "timestamp", "promo"]]

logger.info("\n1️⃣  TabularTS-GBM on the target alone...")
plain = TimeSeriesPipeline("TabularTS-GBM", forecast_params={"prediction_length": 24})
plain.fit(history[["item_id", "timestamp", "target"]], target_only)
plain_mase = plain.evaluate(actual)["mase"]

logger.info("\n2️⃣  TabularTS-GBM with past and known covariates...")
informed = TimeSeriesPipeline("TabularTS-GBM", forecast_params={"prediction_length": 24})
informed.fit(history, with_covariates, future_df=future_df)
informed_mase = informed.evaluate(actual)["mase"]
logger.info(f"\n✅ MASE: target only {plain_mase:.3f}, with covariates {informed_mase:.3f}")

logger.info("\n3️⃣  Chronos-2 joint forecast of two targets...")

panel = make_panel(n_series=3, length=24 * 10, freq="h", seed=1, n_targets=2, covariates=["promo"])
multivariate = TimeSeriesSchema(target=["target_0", "target_1"], item_id="item_id", known_covariates=["promo"])
history, actual = split_horizon(panel, multivariate, prediction_length=12)

workdir = tempfile.TemporaryDirectory()
pipe = TimeSeriesPipeline(
    "Chronos2",
    forecast_params={"prediction_length": 12},
    model_params={"checkpoint": tiny_chronos2_checkpoint(workdir.name)},
)
pipe.fit(history, multivariate, future_df=actual[["item_id", "timestamp", "promo"]])
forecast = pipe.predict()
log_table(logger, "Forecast rows per item and target", list((forecast.to_pandas().groupby(["item_id", "target"]).size().reset_index(name="rows")).columns), (forecast.to_pandas().groupby(["item_id", "target"]).size().reset_index(name="rows")).itertuples(index=False, name=None))
pipe.evaluate(actual, forecast=forecast)
