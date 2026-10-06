"""The Toto family: two generations that are two different models.

TabTune ships both of Datadog's Toto models. Toto 2.0 predicts a fixed grid 
of nine deciles in a single deterministic pass. Toto 1.0 draws sample paths, 
which changes three things a caller can feel:

* **any quantile level works.** 1.0 reads quantiles off its samples, so 0.025 and
  0.975 are as real as 0.1. 2.0 refuses anything off its trained grid rather than
  interpolating a number it never produced. Section 3 shows both answers.
* **a mean forecast exists.** 1.0 can average its paths; 2.0 has no paths to
  average, so `point_forecast="mean"` raises. Section 4.
* **the seed matters.** 1.0 varies run to run unless seeded; 2.0 is deterministic
  and ignores the seed entirely. Section 5.

Sections 6 and 7 cover covariates. The measurements are in
docs/models/toto-1.md and docs/models/toto-2.md.

Toto 1.0 works with a plain `pip install tabtune`. Toto 2.0's backend is not
fully vendored, so it needs an optional install:
    pip install 'tabtune[toto2]'     # Toto 2.0 (unit scaling libraries)

The first run downloads Datadog/Toto-Open-Base-1.0 (~605 MB) and
Datadog/Toto-2.0-22m (~88 MB).

Run:
    python examples/21_toto_family.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tabtune import TimeSeriesPipeline
from tabtune.logger import get_logger, setup_logger
from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.TimeSeries import TimeSeriesSchema

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")


def section(title: str) -> None:
    logger.info(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


rng = np.random.default_rng(0)
HORIZON = 24
STEPS = 512

dates = pd.date_range("2024-01-01", periods=STEPS + HORIZON, freq="h")
t = np.arange(len(dates))

frames = []
for host, level in (("web-1", 40.0), ("web-2", 65.0)):
    promo = (rng.random(len(t)) < 0.2).astype(float)
    cpu = (
        level
        + 12 * np.sin(2 * np.pi * t / 24)
        + 6 * np.sin(2 * np.pi * t / 168)
        + 15 * promo
        + rng.normal(0, 2, len(t))
    )
    frames.append(
        pd.DataFrame(
            {
                "host": host,
                "ts": dates,
                "cpu": cpu,
                "memory": 3 * cpu + rng.normal(0, 8, len(t)),
                "deploys": promo,
            }
        )
    )

data = pd.concat(frames, ignore_index=True)
history = data.groupby("host").head(STEPS)
future = data.groupby("host").tail(HORIZON)
planned = future[["host", "ts", "deploys"]]

MODELS = ("Toto1", "Toto2")
UNIVARIATE = TimeSeriesSchema(target="cpu", timestamp="ts", item_id="host")
MULTIVARIATE = TimeSeriesSchema(target=("cpu", "memory"), timestamp="ts", item_id="host")
WITH_COVARIATES = TimeSeriesSchema(
    target="cpu",
    timestamp="ts",
    item_id="host",
    past_covariates=("memory",),
    known_covariates=("deploys",),
)


SAMPLING = {"num_samples": 128, "samples_per_batch": 64}


def run(model, schema, *, levels=(0.1, 0.5, 0.9), future_df=None, seed=7, **model_params):
    columns = ["host", "ts", *schema.target_names, *schema.covariate_names]
    if model == "Toto1":
        model_params = {**SAMPLING, **model_params}
    pipe = TimeSeriesPipeline(
        model,
        model_params=model_params,
        forecast_params={"prediction_length": HORIZON, "quantile_levels": list(levels)},
        tuning_params={"device": "cpu", "seed": seed},
    )
    pipe.fit(history[columns], schema, future_df=future_df)
    started = time.perf_counter()
    forecast = pipe.predict()
    elapsed = time.perf_counter() - started
    return pipe.evaluate(future, forecast=forecast), forecast, elapsed


section("1. What the registry declares about each generation")

logger.info(f"  {'model':<7} {'checkpoints':>11} {'context':>8} {'covars':>7} {'extra':>6}  paper")
for name in MODELS:
    spec = get_time_series_model_spec(name)
    logger.info(
        f"  {spec.name:<7} {len(spec.checkpoints):>11} {spec.max_context:>8} "
        f"{str(spec.supports_covariates):>7} {str(spec.dependency_extra):>6}  {spec.paper}"
    )

logger.info("\n  Both max_context values are TabTune's choice, not an upstream limit:")
logger.info("  1.0 documents none at all, and 2.0's is derived by inverting the")
logger.info("  residual attention ratio its configs share.")

section("2. The same panel, both generations")

naive = history.groupby("host")["cpu"].last()
naive_mae = (future["cpu"] - future["host"].map(naive)).abs().mean()

logger.info(f"  {'model':<7} {'MAE':>8} {'pinball':>9} {'seconds':>9}")
for name in MODELS:
    metrics, _, elapsed = run(name, UNIVARIATE)
    logger.info(f"  {name:<7} {metrics['mae']:8.3f} {metrics['mean_pinball_loss']:9.3f} {elapsed:9.2f}")
logger.info(f"  {'naive':<7} {naive_mae:8.3f} {'-':>9} {'-':>9}")

section("3. Off-grid quantile levels: served by 1.0, refused by 2.0")

levels = (0.025, 0.5, 0.975)
for name in MODELS:
    try:
        _, forecast, _ = run(name, UNIVARIATE, levels=levels)
        lo, hi = forecast.quantiles[0, :, 0], forecast.quantiles[0, :, 2]
        truth = future.head(HORIZON)["cpu"].to_numpy()
        covered = ((truth >= lo) & (truth <= hi)).mean()
        logger.info(
            f"  {name:<7} serves {levels} -> mean 95% interval width "
            f"{np.mean(hi - lo):.2f}, covers {covered:.0%} of web-1's truth"
        )
    except ConfigError as exc:
        logger.info(f"  {name:<7} refuses {levels}:")
        logger.info(f"          {str(exc).splitlines()[0]}")

logger.info("\n  1.0's quantiles are order statistics of its sample paths, so nothing")
logger.info("  is interpolated. 2.0 would have to invent a number, so it declines.")

section("4. A mean point forecast: available only from sample paths")

for name in MODELS:
    try:
        metrics, _, _ = run(name, UNIVARIATE, point_forecast="mean")
        median, _, _ = run(name, UNIVARIATE, point_forecast="median")
        logger.info(
            f"  {name:<7} mean MAE {metrics['mae']:.3f} vs median MAE {median['mae']:.3f}"
        )
    except ConfigError as exc:
        logger.info(f"  {name:<7} has no mean: {str(exc).splitlines()[0]}")

logger.info("\n  The median is 1.0's default, and upstream's own docstring warns that")
logger.info("  the mean of heavy-tailed sample paths is the worse point forecast.")

section("5. Repeatability: 1.0 needs a seed, 2.0 does not")

for name in MODELS:
    _, a, _ = run(name, UNIVARIATE, seed=7)
    _, b, _ = run(name, UNIVARIATE, seed=7)
    _, c, _ = run(name, UNIVARIATE, seed=8)
    same_seed = np.array_equal(a.point, b.point)
    other_seed = np.array_equal(a.point, c.point)
    verdict = "deterministic (seed ignored)" if other_seed else "sampled (seed decides)"
    logger.info(f"  {name:<7} same seed identical={same_seed}  seed 7 == seed 8={other_seed}  -> {verdict}")

logger.info("\n  1.0 seeds inside torch.random.fork_rng, so forecasting never disturbs")
logger.info("  the global RNG. `samples_per_batch` changes the draws too, so pin it")
logger.info("  alongside the seed when comparing runs.")

section("6. Covariates: accepted by 1.0, refused by 2.0, used by neither")

for name in MODELS:
    try:
        metrics, _, _ = run(name, WITH_COVARIATES, future_df=planned)
        target_only, _, _ = run(name, UNIVARIATE)
        logger.info(
            f"  {name:<7} accepts them -> MAE {metrics['mae']:.3f} "
            f"against {target_only['mae']:.3f} target-only"
        )
    except ConfigError as exc:
        logger.info(f"  {name:<7} refuses them -> {str(exc).splitlines()[0]}")

logger.info("\n  1.0 drives upstream's exogenous path faithfully -- the channel really")
logger.info("  does reach the model as the last variate -- but the released weights do")
logger.info("  not respond to it, measured across one, two and three decode blocks. 2.0")
logger.info("  refuses rather than quietly ignoring. For covariates that move a")
logger.info("  forecast, use Chronos2 or TimesFM3.")

section("7. The 'extra target' workaround, and its limits")

metrics, forecast, _ = run("Toto2", MULTIVARIATE)
alone, _, _ = run("Toto2", UNIVARIATE)
logger.info("  Toto forecasts all variates jointly, so the usual advice for conditioning")
logger.info("  on a related series is to name it as an extra target rather than a")
logger.info("  covariate. Here `memory` is 3x cpu plus noise, so it carries real signal:")
logger.info(
    f"    cpu as one of two targets -> rows {forecast.point.shape[0]}, "
    f"cpu mae={metrics['per_target']['cpu']['mae']:.3f}"
)
logger.info(f"    cpu on its own            -> rows 2, cpu mae={alone['mae']:.3f}")
logger.info("\n  Treat that as unproven, not as advice: the same workaround was measured")
logger.info("  on 1.0 against a driver the target genuinely depends on, and it did not")
logger.info("  condition the forecast either (9.532 as an extra target against 9.533")
logger.info("  target-alone). It has never been measured on 2.0's weights.")

section("8. Multivariate output is long for both")

for name in MODELS:
    _, forecast, _ = run(name, MULTIVARIATE)
    frame = forecast.to_pandas()
    logger.info(
        f"  {name:<7} {forecast.point.shape[0]} rows x {HORIZON} steps -> "
        f"{len(frame)} long rows, targets {sorted(set(frame['target']))}"
    )

logger.info("\n  Both generations' weights are Apache-2.0; only Toto 2.0's backend")
logger.info("  libraries are an optional install.")
