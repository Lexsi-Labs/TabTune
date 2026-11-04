# Causilo

Causilo is Nums AI's in-context model: column attention, then row mixing, then a pooled
prediction stage over a fixed pretrained context. It handles more than ten classes through
an error-correcting output code (ECOC) decomposition with balanced active/rest codes and
streaming probability reconstruction. TabTune vendors `causilo` 1.0.2 unmodified (every
import was already relative) and adds the estimator wrappers, the fine-tuning path and LoRA
in `tabtune_support.py`.

| | |
|---|---|
| `model_name` | `"Causilo"` (aliases `Causilo-v1`, `causilo`) |
| Classification | `inference`, `finetune`, `peft` |
| Regression | `inference`, `finetune`, `peft` |
| `finetune_mode` | `meta-learning` only |
| Limits | none declared; ECOC above ten classes |
| Reads natively | NaN, categorical |
| Weights | [`nums-ai/causilo`](https://huggingface.co/nums-ai/causilo), pinned to a release commit by `checkpoints.py` |
| Licence | Apache-2.0 code, Causilo License v1.0 weights (non-commercial) |

## Usage

```python
from tabtune import TabularPipeline

pipe = TabularPipeline("Causilo", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)

reg = TabularPipeline("Causilo", task_type="regression", tuning_strategy="peft",
                      tuning_params={"epochs": 5, "peft_config": {"r": 8}})
reg.fit(X_train, y_train)
reg.predict_quantiles(X_test)
```

## Why fine-tuning needed its own path

Causilo ships as an inference library. `Engine.fit` prepares a context without touching
weights, and every prediction path runs under `torch.inference_mode()`, so nothing the
public API produces can take part in autograd. The network underneath is ordinary, though:
`ModelRunner.predict(table, targets)` maps `(B, T+Q, features)` plus `(B, T)` training
targets to `(B, Q, outputs)` and carries gradients when it is not called under inference
mode. TabTune builds support/query episodes and drives that entry point directly, the same
episodic scheme the other in-context models use.

## Fine-tuning parameters

`epochs`, `steps_per_epoch` (alias `episodes_per_epoch`), `learning_rate`, `support_size`,
`query_size`, `grad_clip` (alias `grad_clip_value`), `seed` (alias `random_state`), and for
`peft` a `peft_config`. LoRA targets the `projection`, `output`, `value` and `gate` layers
and leaves the `head` alone.

## Limits

- Above ten classes the model runs one forward per ECOC codebook row, so a 30-class
  problem costs several forwards per prediction.
- Pretrained on synthetic tables; the vendor documents performance on tables up to the
  tens of thousands of rows.

## Licence

The code is Apache-2.0. The weights carry the separate Causilo License v1.0, which does
not allow commercial use; the registry records `commercial_use_ok=False`.
