"""Full fine-tuning and LoRA (peft) of Time-MoE, with early stopping and a save/load round trip.

A tiny randomly initialised Time-MoE is built first so the script runs offline. Remove the
checkpoint from model_params to start from Maple728/TimeMoE-50M instead.
"""

import os
import tempfile

import numpy as np
import torch

from tabtune.logger import get_logger, setup_logger
from tabtune.models.time_moe import TimeMoeConfig, TimeMoeForPrediction
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel, split_horizon

setup_logger(use_rich=True)
logger = get_logger("TimeSeries.examples")

logger.info("=" * 80)
logger.info("TIME SERIES FINE-TUNING: Full Fine-Tuning, LoRA and Save/Load")
logger.info("=" * 80)


def tiny_timemoe_checkpoint(directory: str) -> str:
    """Save a 2-layer, 4-expert Time-MoE with random weights in the Hugging Face layout."""
    config = TimeMoeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        horizon_lengths=[1, 8, 24],
        max_position_embeddings=256,
        use_cache=False,
    )
    torch.manual_seed(0)
    TimeMoeForPrediction(config).save_pretrained(directory)
    return directory


workdir = tempfile.TemporaryDirectory()
checkpoint = tiny_timemoe_checkpoint(os.path.join(workdir.name, "timemoe"))

df = make_panel(n_series=8, length=24 * 14, freq="h", seed=0)
schema = TimeSeriesSchema(target="target", item_id="item_id")
history, actual = split_horizon(df, schema, prediction_length=24)
forecast_params = {"prediction_length": 24}
model_params = {"checkpoint": checkpoint}

logger.info("\n1️⃣  Zero-shot baseline...")
zero_shot = TimeSeriesPipeline("TimeMoE", model_params=model_params, forecast_params=forecast_params)
zero_shot.fit(history, schema)
zero_shot_mase = zero_shot.evaluate(actual)["mase"]

logger.info("\n2️⃣  Full fine-tuning with early stopping...")
finetuned = TimeSeriesPipeline(
    "TimeMoE",
    tuning_strategy="finetune",
    model_params=model_params,
    forecast_params=forecast_params,
    tuning_params={
        "epochs": 6,
        "steps_per_epoch": 25,
        "batch_size": 32,
        "learning_rate": 3e-3,
        "early_stopping": True,
        "early_stopping_patience": 2,
        "seed": 0,
    },
)
finetuned.fit(history, schema)
report = finetuned.training_report_
logger.info(f"   Validation loss per epoch: {[round(h['validation_loss'], 3) for h in report['history']]}")
finetune_mase = finetuned.evaluate(actual)["mase"]

logger.info("\n3️⃣  LoRA (peft) fine-tuning...")
lora = TimeSeriesPipeline(
    "TimeMoE",
    tuning_strategy="peft",
    model_params=model_params,
    forecast_params=forecast_params,
    tuning_params={
        "epochs": 2,
        "steps_per_epoch": 25,
        "batch_size": 32,
        "learning_rate": 1e-2,
        "peft_config": {"r": 4, "lora_alpha": 8},
        "seed": 0,
    },
)
lora.fit(history, schema)
# LoRA adapts a pretrained model; on random base weights it has little to adapt.
peft_mase = lora.evaluate(actual)["mase"]

logger.info("\n" + "=" * 80)
logger.info("📊 MASE Comparison")
logger.info("=" * 80)
logger.info(f"   zero-shot  {zero_shot_mase:.3f}")
logger.info(f"   finetune   {finetune_mase:.3f}")
logger.info(f"   peft       {peft_mase:.3f}")

logger.info("\n4️⃣  Save/load round trip...")
for name, pipe in (("finetune", finetuned), ("peft", lora)):
    path = os.path.join(workdir.name, f"{name}.joblib")
    pipe.save(path)
    restored = TimeSeriesPipeline.load(path)
    same = np.allclose(restored.predict().point, pipe.predict().point, atol=1e-5)
    logger.info(f"   {'✅' if same else '❌'} {name}: saved {os.path.getsize(path) / 1024:.0f} KiB, reloaded forecast identical: {same}")
