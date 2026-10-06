"""The Chronos family: three models, one pipeline.

* **Chronos v1** is a T5 language model over tokenised series. It *samples*
  paths, so it needs a seed to be repeatable and its point forecast is the mean
  of those paths. Context 512.
* **Chronos-Bolt** patches the history and reads all nine quantiles off one
  forward pass. Nothing is sampled, so it is deterministic and much faster, with
  a 2048-step context.
* **Chronos-2** reads everything about an item at once: several targets forecast
  jointly, covariates observed over the history, and covariates whose future
  values you already know. Context 8192.

Swapping between them is a one-word change: the schema, the result type, the
metrics and the cache are identical. So this example runs all three over one
panel and compares them at the end. Sections 4 and 5 show what only Chronos-2
adds, and how v1 and Bolt refuse that schema rather than quietly ignoring it.

Run:
    python examples/19_chronos_family.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tabtune import TimeSeriesPipeline
from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.registry import ConfigError, get_time_series_model_spec
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
            }
        )
    )

data = pd.concat(frames, ignore_index=True)
history = data.groupby("store").head(DAYS)
future = data.groupby("store").tail(HORIZON)

planned = future[["store", "date", "promo", "weather"]]

CHECKPOINTS = {
    "Chronos": "amazon/chronos-t5-tiny",
    "ChronosBolt": "amazon/chronos-bolt-tiny",
    "Chronos2": "amazon/chronos-2",
}
MODELS = tuple(CHECKPOINTS)

TARGET_ONLY = TimeSeriesSchema(target="sales", timestamp="date", item_id="store")
MULTI = TimeSeriesSchema(
    target=("sales", "traffic"),
    timestamp="date",
    item_id="store",
    known_covariates=("promo", "weather"),
)


def run(model, schema, *, future_df=None, seed=0, **model_params):
    columns = ["store", "date", *schema.target_names, *schema.covariate_names]
    pipe = TimeSeriesPipeline(
        model,
        model_params={"checkpoint": CHECKPOINTS[model], **model_params},
        forecast_params={"prediction_length": HORIZON, "quantile_levels": [0.1, 0.5, 0.9]},

        tuning_params={"seed": seed} if seed is not None else {},
    )
    pipe.fit(history[columns], schema, future_df=future_df)
    started = time.perf_counter()
    forecast = pipe.predict()
    elapsed = time.perf_counter() - started
    return pipe.evaluate(future, forecast=forecast), forecast, elapsed


section("1. Zero-shot: fit only validates the history and loads the weights")

pipe = TimeSeriesPipeline(
    "Chronos",
    model_params={"checkpoint": CHECKPOINTS["Chronos"]},
    forecast_params={"prediction_length": HORIZON, "quantile_levels": [0.1, 0.5, 0.9]},
    tuning_params={"seed": 0},
)
pipe.fit(history[["store", "date", "sales"]], TARGET_ONLY)
logger.info("\n%s", pipe)
logger.info(f"  training occurred: {pipe.training_occurred_}")

result = pipe.predict()
logger.info(f"  point forecast = {result.metadata['point_forecast']} of the sampled paths\n")
frame = result.to_pandas()
log_table(logger, "Forecast (first rows per store)", list((frame.groupby("store").head(3)).columns), (frame.groupby("store").head(3)).itertuples(index=False, name=None))

merged = frame.merge(future, on=["store", "date"])
inside = merged["sales"].between(merged["0.1"], merged["0.9"])
logger.info(f"\n  {inside.mean():.0%} of held-out actuals fall inside the 10-90% band")

section("2. The same univariate panel, all three models")

naive = history.groupby("store")["sales"].last()
naive_mae = (future["sales"] - future["store"].map(naive)).abs().mean()

univariate = {}
logger.info(f"  {'model':<12} {'MAE':>8} {'pinball':>9} {'seconds':>9} {'point':>8} {'context':>8}")
for name in MODELS:
    metrics, forecast, elapsed = run(name, TARGET_ONLY)
    univariate[name] = metrics
    spec = get_time_series_model_spec(name)
    logger.info(
        f"  {name:<12} {metrics['mae']:8.3f} {metrics['mean_pinball_loss']:9.3f} "
        f"{elapsed:9.2f} {forecast.metadata['point_forecast']:>8} {spec.max_context:8}"
    )
logger.info(f"  {'naive':<12} {naive_mae:8.3f} {'-':>9} {'-':>9} {'-':>8} {'-':>8}")

section("3. Sampled or quantile-native: which runs need a seed")

for name in MODELS:
    _, seeded, _ = run(name, TARGET_ONLY, seed=0)
    _, unseeded, _ = run(name, TARGET_ONLY, seed=None)  
    identical = np.array_equal(seeded.point, unseeded.point)
    kind = "quantile-native (seed ignored)" if identical else "sampled (seed decides)"
    logger.info(f"  {name:<12} unseeded rerun reproduces the seeded run: {str(identical):<5}  -> {kind}")

logger.info("\n  Only v1 draws sample paths, which is also why only v1 can average them")
logger.info("  into a mean. Bolt and Chronos-2 return the 0.5 quantile as their point.")

section("4. Covariates: only Chronos-2 reads them")

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
        known_covariates=("promo", "weather"),
    ),
}

logger.info(f"  {'schema':<18} {'MAE':>8} {'pinball':>9}   (Chronos2)")
for label, schema in SCHEMAS.items():
    metrics, _, _ = run(
        "Chronos2", schema, future_df=planned if schema.known_covariates else None
    )
    logger.info(f"  {label:<18} {metrics['mae']:8.3f} {metrics['mean_pinball_loss']:9.3f}")
logger.info(f"  {'naive':<18} {naive_mae:8.3f} {'-':>9}")
logger.info("\n  'promo' and 'weather' really drive these sales, so telling the model what")
logger.info("  is planned should beat forecasting from the target history alone. Note that")
logger.info("  'weather' is a string column: Chronos-2 encodes categories natively.\n")

for name in ("Chronos", "ChronosBolt"):
    try:
        run(name, SCHEMAS["+ known future"], future_df=planned)
        logger.info(f"  {name:<12} read the covariates")
    except ConfigError as exc:
        logger.info(f"  {name:<12} refuses that schema -> {str(exc).splitlines()[0]}")

logger.info("\n  Declared, not discovered: without the capability flags, v1 would have")
logger.info("  returned a plausible forecast that ignored every covariate column.")

section("5. Two targets, forecast jointly by Chronos-2")

metrics, forecast, _ = run("Chronos2", MULTI, future_df=planned)
logger.info(f"  panel rows (items x targets): {forecast.point.shape[0]}")
logger.info(f"  targets per row: {forecast.targets}")
logger.info("\n  per-target metrics (different scales, so never averaged blindly):")
for name, scores in metrics["per_target"].items():
    logger.info(f"    {name:<10} mae={scores['mae']:9.3f}  rmse={scores['rmse']:9.3f}")

frame = forecast.to_pandas()
logger.info("\n  the long result, one row per (item, target, timestamp):")
log_table(logger, "Store A (first rows)", list((frame.query("store == 'A'").head(6)).columns), (frame.query("store == 'A'").head(6)).itertuples(index=False, name=None))
logger.info(f"\n  {len(frame)} rows = 2 items x 2 targets x {HORIZON} steps")

section("6. The family side by side")

logger.info(
    f"  {'model':<12} {'checkpoint':<24} {'context':>8} {'horizon':>8} "
    f"{'covars':>7} {'multi':>6} {'MAE':>8}"
)
for name in MODELS:
    spec = get_time_series_model_spec(name)
    horizon = "none" if spec.max_horizon is None else str(spec.max_horizon)
    logger.info(
        f"  {name:<12} {CHECKPOINTS[name]:<24} {spec.max_context:>8} {horizon:>8} "
        f"{str(spec.supports_covariates):>7} {str(spec.supports_multivariate):>6} "
        f"{univariate[name]['mae']:8.3f}"
    )
logger.info(f"  {'naive':<12} {'-':<24} {'-':>8} {'-':>8} {'-':>7} {'-':>6} {naive_mae:8.3f}")

logger.info("\n  The MAE column is the univariate panel from section 2, so it compares")
logger.info("  like with like. All three checkpoints here are the smallest of their")
logger.info("  line; the larger ones trade speed for accuracy. Every Chronos weight is")
logger.info("  Apache-2.0, so any of them is usable commercially.")
