"""Series embeddings from Time-MoE, and customer histories as features for a tabular churn model.

A tiny randomly initialised Time-MoE is built first so the script runs offline; its embeddings
carry no learned structure. Remove the checkpoint to use Maple728/TimeMoE-50M.
"""

import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from tabtune import TabularPipeline
from tabtune.bridge import SeriesFeaturizer
from tabtune.logger import get_logger, log_table, setup_logger
from tabtune.models.time_moe import TimeMoeConfig, TimeMoeForPrediction
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES EMBEDDINGS AND SERIES FEATURES FOR TABULAR MODELS")
logger.info("=" * 80)


def tiny_timemoe_checkpoint(directory: str) -> str:
    """Save a 2-layer, 4-expert Time-MoE with random weights in the Hugging Face layout."""
    config = TimeMoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        horizon_lengths=[1, 4, 8],
        max_position_embeddings=128,
        use_cache=False,
    )
    torch.manual_seed(0)
    TimeMoeForPrediction(config).save_pretrained(directory)
    return directory


workdir = tempfile.TemporaryDirectory()
checkpoint = tiny_timemoe_checkpoint(workdir.name)

logger.info("\n1️⃣  Embedding series with Time-MoE...")
panel = make_panel(n_series=4, length=24 * 7, freq="h", seed=0)
encoder = TimeSeriesPipeline("TimeMoE", task_type="embedding", model_params={"checkpoint": checkpoint})
embeddings = encoder.fit(panel, TimeSeriesSchema(target="target", item_id="item_id")).predict()
log_table(logger, "Embeddings (first 6 columns)", list((embeddings.to_pandas().iloc[:, :6]).columns), (embeddings.to_pandas().iloc[:, :6]).itertuples(index=False, name=None))
logger.info("   Cosine similarity:\n%s", np.round(embeddings.similarity(), 3))


def churn_data(n_customers: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Churners' daily spend trends down; age carries no signal."""
    rng = np.random.default_rng(seed)
    histories, rows = [], []
    for customer in range(n_customers):
        churn = int(rng.random() < 0.5)
        days = np.arange(90)
        spend = rng.uniform(20, 60) + (-0.3 if churn else 0.05) * days + rng.normal(0, 3, 90)
        histories.append(
            pd.DataFrame({"customer": customer, "date": pd.date_range("2024-01-01", periods=90), "spend": spend})
        )
        as_of = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(45, 90)))
        rows.append({"customer": customer, "age": int(rng.integers(18, 80)), "as_of": as_of, "churn": churn})
    table = pd.DataFrame(rows)
    return table, pd.concat(histories, ignore_index=True), table.pop("churn")


logger.info("\n2️⃣  Customer spend histories as features for a churn model...")
table, spend, churn = churn_data(300)
train, test, y_train, y_test = train_test_split(table, churn, test_size=0.3, random_state=0, stratify=churn)

# cutoff="as_of": each row sees only the observations strictly before its own as-of date.
featurizer = SeriesFeaturizer(
    spend,
    id_col="customer",
    time_col="date",
    cutoff="as_of",
    model="TimeMoE",
    checkpoint=checkpoint,
    n_components=4,
)
X_train = featurizer.fit_transform(train).drop(columns=["customer", "as_of"])
X_test = featurizer.transform(test).drop(columns=["customer", "as_of"])
logger.info(f"   {X_train.shape[1]} columns, e.g. {list(X_train.columns[1:5])} ... {list(X_train.columns[-2:])}")

for label, columns in (("age only", ["age"]), ("age + history", list(X_train.columns))):
    model = TabularPipeline("XRFM", task_type="classification").fit(X_train[columns], y_train)
    accuracy = np.mean(model.predict(X_test[columns]) == y_test.to_numpy())
    logger.info(f"   {label:<14} accuracy {accuracy:.3f}")

logger.info("\n3️⃣  Leakage check: corrupting every value at or after the cutoff...")
corrupted = spend.merge(table[["customer", "as_of"]], on="customer")
corrupted.loc[corrupted["date"] >= corrupted["as_of"], "spend"] = 1e9
corrupted = corrupted.drop(columns="as_of")
clean = SeriesFeaturizer(spend, id_col="customer", time_col="date", cutoff="as_of").fit_transform(test)
dirty = SeriesFeaturizer(corrupted, id_col="customer", time_col="date", cutoff="as_of").fit_transform(test)
unchanged = np.allclose(clean.filter(like="ts_"), dirty.filter(like="ts_"), equal_nan=True)
logger.info(f"   {'✅' if unchanged else '❌'} Summaries unchanged: {unchanged}")
