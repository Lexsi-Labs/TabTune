"""TimesFM 3.0: Google's family, and what it declares up front.

TimesFM 3.0 is the second time series family TabTune ships.
It produces the whole horizon in one non-autoregressive pass, 
so it has no maximum horizon at all, and it reads a 
15360-step context, the longest of any model here.

It also has three hard edges, and every one of them is *declared* rather than
discovered halfway through a forecast:

* covariates are unnamed numeric channels, so a string column raises before any
  weights load;
* it predicts nine fixed quantile levels (the deciles). Asking for 0.05 raises
  rather than interpolating a number the model never produced;
* the weights are released under a non-commercial license while the code is
  Apache-2.0, so `license_mode="commercial"` refuses them.

Section 4 triggers all three on purpose. One quirk worth knowing while reading
the numbers: TimesFM fits and removes a linear trend before the transformer sees
anything, so a perfectly linear series returns its own extrapolation whatever
the weights say. The panel below has seasonality and noise for that reason.

Run:
    python examples/20_timesfm3_family.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tabtune import TimeSeriesPipeline
from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.registry.errors import LicenseError
from tabtune.TimeSeries import TimeSeriesSchema

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")


def section(title: str) -> None:
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)


rng = np.random.default_rng(0)
HORIZON = 14
DAYS = 180

dates = pd.date_range("2024-01-01", periods=DAYS + HORIZON, freq="D")
t = np.arange(len(dates))

frames = []
for store, level, trend in (("A", 100.0, 0.0), ("B", 60.0, 0.3)):
    promo = (rng.random(len(t)) < 0.2).astype(float)
    weather = rng.choice(["sun", "rain"], size=len(t), p=[0.7, 0.3])
    footfall = 500 + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 10, len(t))
    sales = (
        level
        + trend * t
        + 15 * np.sin(2 * np.pi * t / 7)
        + 25 * promo                      
        - 8 * (weather == "rain")          
        + 0.02 * footfall
        + rng.normal(0, 3, len(t))
    )
    frames.append(
        pd.DataFrame(
            {
                "store": store,
                "date": dates,
                "sales": sales,
                "traffic": 10 * sales + rng.normal(0, 40, len(t)),
                "footfall": footfall,
                "promo": promo,
                "weather": weather,
                "is_rain": (weather == "rain").astype(float),
            }
        )
    )

data = pd.concat(frames, ignore_index=True)
history = data.groupby("store").head(DAYS)
future = data.groupby("store").tail(HORIZON)
planned = future[["store", "date", "promo", "is_rain"]]

MODEL = "TimesFM3"

TARGET_ONLY = TimeSeriesSchema(target="sales", timestamp="date", item_id="store")
MULTI = TimeSeriesSchema(
    target=("sales", "traffic"),
    timestamp="date",
    item_id="store",
    known_covariates=("promo", "is_rain"),
)


def run(schema, *, future_df=None, levels=(0.1, 0.5, 0.9), model=MODEL, **model_params):
    columns = ["store", "date", *schema.target_names, *schema.covariate_names]
    pipe = TimeSeriesPipeline(
        model,
        model_params=model_params,
        forecast_params={"prediction_length": HORIZON, "quantile_levels": list(levels)},
        tuning_params={"device": "cpu"},
    )
    pipe.fit(history[columns], schema, future_df=future_df)
    started = time.perf_counter()
    forecast = pipe.predict()
    elapsed = time.perf_counter() - started
    return pipe.evaluate(future, forecast=forecast), forecast, elapsed


section("1. What the registry declares")

spec = get_time_series_model_spec(MODEL)
horizon = "none (single pass)" if spec.max_horizon is None else str(spec.max_horizon)
logger.info(f"  {'family':<22} {spec.family}")
logger.info(f"  {'checkpoints':<22} {', '.join(spec.checkpoints)}")
logger.info(f"  {'max context':<22} {spec.max_context}")
logger.info(f"  {'max horizon':<22} {horizon}")
logger.info(f"  {'native missing values':<22} {spec.native_missing}")
logger.info(f"  {'multivariate targets':<22} {spec.supports_multivariate}")
logger.info(f"  {'covariates':<22} {spec.supports_covariates}")
logger.info(f"  {'categorical covariates':<22} {spec.supports_categorical_covariates}")
logger.info(f"  {'weights license':<22} {spec.license.name}")
logger.info(f"  {'commercial use ok':<22} {spec.license.commercial_use_ok}")
logger.info(f"  {'dependency extra':<22} {spec.dependency_extra}")

section("2. A univariate forecast")

naive = history.groupby("store")["sales"].last()
naive_mae = (future["sales"] - future["store"].map(naive)).abs().mean()

metrics, forecast, elapsed = run(TARGET_ONLY)
logger.info(f"  {'model':<12} {'MAE':>8} {'pinball':>9} {'seconds':>9} {'point':>8}")
logger.info(
    f"  {MODEL:<12} {metrics['mae']:8.3f} {metrics['mean_pinball_loss']:9.3f} "
    f"{elapsed:9.2f} {forecast.metadata['point_forecast']:>8}"
)
logger.info(f"  {'naive':<12} {naive_mae:8.3f} {'-':>9} {'-':>9} {'-':>8}")

frame = forecast.to_pandas()
merged = frame.merge(future, on=["store", "date"])
inside = merged["sales"].between(merged["0.1"], merged["0.9"])
logger.info(f"\n  {inside.mean():.0%} of held-out actuals fall inside the 10-90% band\n")
log_table(logger, "Forecast (first rows per store)", list((frame.groupby("store").head(3)).columns), (frame.groupby("store").head(3)).itertuples(index=False, name=None))

section("3. Covariates as unnamed numeric channels")

SCHEMAS = {
    "target only": TARGET_ONLY,
    "+ past covariate": TimeSeriesSchema(
        target="sales", timestamp="date", item_id="store", past_covariates=("footfall",)
    ),
    "+ known future": TimeSeriesSchema(
        target="sales",
        timestamp="date",
        item_id="store",
        past_covariates=("footfall",),
        known_covariates=("promo", "is_rain"),
    ),
}

logger.info(f"  {'schema':<18} {'MAE':>8} {'pinball':>9}")
for label, schema in SCHEMAS.items():
    scores, _, _ = run(schema, future_df=planned if schema.known_covariates else None)
    logger.info(f"  {label:<18} {scores['mae']:8.3f} {scores['mean_pinball_loss']:9.3f}")
logger.info(f"  {'naive':<18} {naive_mae:8.3f} {'-':>9}")

logger.info("\n  Covariates are concatenated onto the target as extra float channels:")
logger.info("  past-only ones span the history, known-future ones span history and")
logger.info("  horizon in one array. They carry no names into the model, so the order")
logger.info("  is the schema's -- which is also why a string column cannot be passed.")

section("4. Three declared refusals")

categorical = TimeSeriesSchema(
    target="sales", timestamp="date", item_id="store", known_covariates=("promo", "weather")
)
try:
    run(categorical, future_df=future[["store", "date", "promo", "weather"]])
    logger.info("  categorical covariate: accepted")
except ConfigError as exc:
    logger.info(f"  categorical covariate: {str(exc).splitlines()[0]}")

try:
    run(TARGET_ONLY, levels=(0.05, 0.5, 0.95))
except ConfigError as exc:
    logger.info(f"\n  off-grid quantile:     {str(exc).splitlines()[0]}")

try:
    TimeSeriesPipeline(
        MODEL,
        forecast_params={"prediction_length": HORIZON},
        license_mode="commercial",
    )
except LicenseError as exc:
    logger.info(f"\n  commercial license:    {str(exc).splitlines()[0]}")

logger.info("\n  The license check runs in __init__ and the categorical one in fit, so")
logger.info("  neither ever loads the weights; the quantile grid is checked when the")
logger.info("  forecast is requested. The refusal is a deliberate choice: TimesFM")
logger.info("  produces nine deciles, and interpolating a 0.05 would mean reporting a")
logger.info("  number the model never predicted.")

section("5. Two targets, forecast jointly")

metrics, forecast, _ = run(MULTI, future_df=planned)
logger.info(f"  panel rows (items x targets): {forecast.point.shape[0]}")
logger.info(f"  targets per row: {forecast.targets}")
logger.info("\n  per-target metrics (different scales, so never averaged blindly):")
for name, scores in metrics["per_target"].items():
    logger.info(f"    {name:<10} mae={scores['mae']:9.3f}  rmse={scores['rmse']:9.3f}")

frame = forecast.to_pandas()
logger.info(f"\n  {len(frame)} rows = 2 items x 2 targets x {HORIZON} steps")
log_table(logger, "Store A (first rows)", list((frame.query("store == 'A'").head(6)).columns), (frame.query("store == 'A'").head(6)).itertuples(index=False, name=None))

logger.info("\n  Reminder: the vendored TimesFM 3 code is Apache-2.0, but the weights")
logger.info(f"  are {spec.license.name}. For commercial work the registry")
logger.info(f"  points at: {', '.join(spec.commercial_alternatives)}.")
