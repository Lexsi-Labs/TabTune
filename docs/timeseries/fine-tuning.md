# Time Series Fine-Tuning

All eleven foundation models support `tuning_strategy="finetune"` (every weight) and
`"peft"` (LoRA adapters on the linear layers each adapter names). Training windows are
sampled from the history: a context of `context_length` steps followed by
`prediction_length` steps to predict, so the model trains on the shape it will be asked
for.

```python
pipe = TimeSeriesPipeline(
    "TiRex",
    tuning_strategy="peft",
    forecast_params={"prediction_length": 24},
    tuning_params={
        "epochs": 3,
        "steps_per_epoch": 50,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "validation_split": 0.1,          # any value above 0 holds out the last horizon of every series
        "early_stopping": True,
        "early_stopping_patience": 2,
        "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    },
)
pipe.fit(history, schema)
pipe.backtest(windows=3)
pipe.save("tirex-lora.joblib")       # adapters and the fitted state travel with the file
```

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `epochs` | 1 | Passes over `steps_per_epoch` batches. |
| `steps_per_epoch` | 50 | One step is one batch of sampled windows. |
| `learning_rate` | 1e-5 | AdamW learning rate. |
| `batch_size` | 32 | Windows per step. |
| `validation_split` | 0 | Above 0 holds out the last `prediction_length` steps of every series for early stopping. |
| `early_stopping`, `early_stopping_patience` | off, 1 | Stop after this many epochs without validation improvement. |
| `gradient_clip_norm` | 1.0 | Gradient-norm clip. |
| `device`, `seed` | auto, 0 | Also the only two keys zero-shot inference reads. |
| `peft_config` | - | `r`, `lora_alpha`, `lora_dropout`, `target_modules` (overrides the adapter's defaults). |

`forecast_params["context_length"]` sets the training context too; `None` uses the model's
limit. A horizon longer than the checkpoint was documented for is trained at the limit and
rolled out at prediction time, with a warning.

## What a fine-tuned pipeline carries

`save()` writes the tuned state: every weight for `finetune`, only the adapters for `peft`,
plus the schema and the configuration. `load()` rebuilds the base model from its checkpoint
and applies the state, so the saved file stays small for LoRA runs.

## When it helps

Zero-shot is strong on most public benchmarks, and a fine-tuned run can lose to it. Always
`backtest()` both, or put both on a `TimeSeriesLeaderboard` with a `label`:

```python
board = TimeSeriesLeaderboard(history, schema, forecast_params={"prediction_length": 24}, windows=3)
board.add_model("TiRex")
board.add_model("TiRex", tuning_strategy="peft", tuning_params={"epochs": 2}, label="TiRex LoRA")
board.run()
```

A fine-tuned member on the leaderboard or in an ensemble is trained without the scored
windows and refitted on the whole history afterwards, so its score is honest.
