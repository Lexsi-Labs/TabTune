# TabPFN v2.6

TabPFN v2.6 is Prior Labs' follow-up to TabPFN v2 with the same prior-fitted network idea
and the same limits, plus a native fine-tuning API that trains with the bar-distribution
loss the model was pretrained on. TabTune vendors the Prior Labs source as
`tabtune/models/tabpfnv26` with imports rewritten and nothing else changed.

| | |
|---|---|
| `model_name` | `"TabPFNv26"` (aliases `TabPFN-v2.6`, `TabPFN2.6`, `TabPFNv2.6`, `TabPFN26`) |
| Classification | `inference`, `finetune` |
| Regression | `inference`, `finetune` |
| `finetune_mode` | `meta-learning`, `sft`, `turn_by_turn`, `native` |
| Limits | 10 classes, 500 features, 10,000 rows (soft; `ignore_pretraining_limits` is set) |
| Reads natively | NaN |
| Weights | [`Prior-Labs/tabpfn_2_5`](https://huggingface.co/Prior-Labs/tabpfn_2_5), non-commercial licence |
| Paper | [arXiv:2511.08667](https://arxiv.org/abs/2511.08667) |

## Usage

```python
from tabtune import TabularPipeline

# Zero-shot
pipe = TabularPipeline("TabPFNv26", task_type="classification", tuning_strategy="inference",
                       model_params={"n_estimators": 8})
pipe.fit(X_train, y_train)

# Native fine-tuning: Prior Labs' own loop, bar-distribution loss, its own validation
pipe = TabularPipeline("TabPFNv26", task_type="classification", tuning_strategy="finetune",
                       finetune_mode="native",
                       tuning_params={"epochs": 5, "learning_rate": 1e-5})
pipe.fit(X_train, y_train)

# Regression, same checkpoint family
reg = TabularPipeline("TabPFNv26", task_type="regression", tuning_strategy="finetune",
                      finetune_mode="native")
reg.fit(X_train, y_train)
reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

## Model parameters

The constructor of the vendored `TabPFNClassifier` / `TabPFNRegressor`. The ones that
matter day to day:

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | 8 | Ensemble members, each a different feature and class permutation. More is slower and a little better. |
| `softmax_temperature` | 0.9 | Lower is sharper. |
| `balance_probabilities` | `False` | Rebalance class probabilities to a uniform prior. |
| `average_before_softmax` | `False` | Average logits across members instead of probabilities. |
| `inference_precision` | `"auto"` | `"auto"`, `"autocast"` or a torch dtype. |
| `memory_saving_mode` | `"auto"` | Batch the estimators to fit the device memory. |
| `random_state` | 0 | Seed for the permutations. |
| `model_path` | `"auto"` | Local checkpoint file; `"auto"` uses the Hugging Face cache. |
| `inference_config` | `None` | Dict of upstream `InferenceConfig` overrides, such as `{"SUBSAMPLE_SAMPLES": 5000}`. |

## Fine-tuning

- `native` runs Prior Labs' `FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor`. It
  has its own train/validation split, its own early stopping, and the regression head
  trains on the bar distribution rather than on MSE. This is the mode to try first.
- `meta-learning`, `sft` and `turn_by_turn` are TabTune's loops, shared with TabPFN v2,
  described on [Tuning Strategies](../user-guide/tuning-strategies.md).
- There is no LoRA target table for v2.6, so `peft` is not offered. Use v3 or v3.5 if you
  want adapters on a PFN.

## Limits

TabTune passes `ignore_pretraining_limits=True`, so a table beyond 10,000 rows or 500
features runs and gets worse rather than raising. For big tables use
`inference_config={"SUBSAMPLE_SAMPLES": ...}` or move to [TabPFN v3](tabpfnv3.md). More
than ten classes raises; the head has ten slots.

## Licence

Code is Apache-2.0 with attribution. The v2.6 weights are under the TabPFN-2.6 licence,
which does not allow commercial use; the registry records `commercial_use_ok=False` and
names Mitra, TabICL v2 and the Orion family as commercial alternatives.
