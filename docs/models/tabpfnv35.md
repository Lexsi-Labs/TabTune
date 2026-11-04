# TabPFN v3.5 and v3.5-fast

TabPFN v3.5 is a new architecture, not a new checkpoint for v3, and it is the first Prior
Labs release where one multitask checkpoint carries both the classification and the
regression head. It reads text and datetime columns natively. v3.5-fast is a separately
trained, lower-cost model on the same runtime, not a re-export. TabTune vendors the Prior
Labs v9.0.0 source as `tabtune/models/tabpfnv35`, imports rewritten, model code untouched.

| | |
|---|---|
| `model_name` | `"TabPFNv35"` (aliases `TabPFN-v3.5`, `TabPFN3.5`) and `"TabPFNv35Fast"` (aliases `TabPFN-v3.5-fast`, `TabPFN3.5Fast`) |
| Classification | `inference`, `finetune`, `peft` |
| Regression | `inference`, `finetune` |
| `finetune_mode` | `meta-learning`, `sft`, `turn_by_turn`, `native` |
| Limits | none declared; the runtime subsamples and warns |
| Reads natively | NaN, text, categorical, datetime |
| Weights | [`Prior-Labs/tabpfn_3_5`](https://huggingface.co/Prior-Labs/tabpfn_3_5), non-commercial licence |
| Docs | [docs.priorlabs.ai/models](https://docs.priorlabs.ai/models) |

## Usage

```python
from tabtune import TabularPipeline

pipe = TabularPipeline("TabPFNv35", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)            # pass the raw frame; strings and dates are fine

fast = TabularPipeline("TabPFNv35Fast", task_type="regression", tuning_strategy="inference")
fast.fit(X_train, y_train)

# Native fine-tuning, pinned to the v3.5 checkpoint
pipe = TabularPipeline("TabPFNv35", task_type="classification", tuning_strategy="finetune",
                       finetune_mode="native", tuning_params={"epochs": 5})
pipe.fit(X_train, y_train)

# LoRA on the attention projections, the embedding and the ICL blocks
pipe = TabularPipeline("TabPFNv35", task_type="classification", tuning_strategy="peft",
                       tuning_params={"epochs": 5, "peft_config": {"r": 8}})
pipe.fit(X_train, y_train)
```

## What is different from v3

- One checkpoint, two heads. Loading code differs, which is one reason this is a separate
  vendored tree rather than a patch to `tabpfnv3`.
- Text and datetime columns go through the model's own preprocessing (it imports `skrub`,
  now a core dependency of TabTune for that reason).
- The runtime no longer registers the legacy `tabpfn.model` module aliases upstream
  installs for AutoGluon, so a real `tabpfn` package can be co-installed without the two
  fighting over `sys.modules`.
- Both v3.5 names run on the same vendored tree; the model version is pinned explicitly in
  each class so a later upstream default cannot silently change what TabTune loads.

## Model parameters

The same constructor as [TabPFN v2.6](tabpfnv26.md) (`n_estimators`, `softmax_temperature`,
`inference_precision`, `memory_saving_mode`, `inference_config`, `random_state`), plus
`model_path`, which is pinned to the v3.5 or v3.5-fast checkpoint and should normally be
left at `"auto"`.

For large tables use per-estimator subsampling:

```python
model_params={"inference_config": {"SUBSAMPLE_SAMPLES": 50_000}}
```

## Fine-tuning

`native` wraps Prior Labs' finetuning classes and pins the model version so a fine-tuned
v3.5-fast stays v3.5-fast. `meta-learning`, `sft` and `turn_by_turn` are TabTune's loops.
`peft` targets `q_projection`, `k_projection`, `v_projection`, `out_projection`, `x_embed`
and the `icl_blocks`.

The `tbt` alias routes regression to `turn_by_turn`.

## Licence

The v3.5 weights are under the TabPFN 3.5 licence v1.0, non-commercial. The registry records
`commercial_use_ok=False`. Downloading them requires accepting Prior Labs' terms on the
Hub, so log in with a token first.
