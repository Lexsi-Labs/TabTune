"""Flag point anomalies from forecast residuals, and compare with a rolling median detector.

Runs offline with SeasonalNaive. Replace it with "Chronos2" or "ChronosBolt" to score against
a foundation model's forecasts.
"""


from sklearn.metrics import roc_auc_score

from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema
from tabtune.TimeSeries.data import make_anomalous_panel

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES ANOMALY DETECTION: Forecast Residuals vs Rolling Median")
logger.info("=" * 80)

df = make_anomalous_panel(n_series=3, length=400, n_anomalies=5, seed=0)
labels = df[["item_id", "timestamp", "is_anomaly"]]
data = df.drop(columns="is_anomaly")
schema = TimeSeriesSchema(target="target", item_id="item_id")

logger.info("\n1️⃣  Scoring anomalies with each detection method...")
for method in ("forecast_error", "interval", "likelihood"):
    pipe = TimeSeriesPipeline(
        "SeasonalNaive",
        task_type="anomaly_detection",
        task_params={"method": method, "alpha": 0.01},
    )
    result = pipe.fit(data, schema).predict()
    scores = result.evaluate(labels)
    logger.info(f"   {method:<15} {result}  " + "  ".join(f"{k}={v:.3f}" for k, v in scores.items() if k not in ("n", "n_anomalies")))

logger.info("\n2️⃣  Top flagged points:")
log_table(logger, "Anomalies (first rows)", list((result.anomalies[["item_id", "timestamp", "value", "expected", "score", "p_value"]].head()).columns), (result.anomalies[["item_id", "timestamp", "value", "expected", "score", "p_value"]].head()).itertuples(index=False, name=None))


def rolling_median_score(values):
    median = values.rolling(25, center=True, min_periods=5).median()
    spread = (values - median).abs().rolling(25, center=True, min_periods=5).median()
    return (values - median).abs() / (1.4826 * spread)


# The window is centred, so this baseline also looks at later values; the forecast-based
# scores use only the history before each point.
baseline = data.groupby("item_id")["target"].transform(rolling_median_score).to_numpy()
scored = result.to_pandas()["score"].notna().to_numpy()
logger.info("\n3️⃣  Rolling median baseline")
logger.info(f"   AUROC: {roc_auc_score(labels['is_anomaly'][scored], baseline[scored]):.3f}")
