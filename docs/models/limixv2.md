# LimiX-2

LimiX-2 is StableAI's 400M-parameter successor to LimiX. It is a contextual mechanism network pretrained with context-conditional masked modelling on structural-causal-model priors, and one model covers classification, regression and imputation. TabTune vendors LimiX `inference/v2_0` and `model/v2_0` in `tabtune/models/limixv2/_upstream`; preprocessing and ensembling run unchanged through upstream's `LimiXPredictor`. Episodic fine-tuning and LoRA are TabTune additions in `tabtune_support.py`.

LimiX v1 remains available as `model_name="Limix"` ([LimiX](limix.md)); it is a separate registry entry with its own strategies.

| | |
|---|---|
| `model_name` | `"LimiXv2"` (aliases: `"LimiX-2"`, `"LimiX2"`, `"LimiXV2"`, `"Limix-2"`) |
| Classification strategies | `inference`, `finetune`, `peft` |
| Regression strategies | `inference`, `finetune`, `peft` |
| `finetune_mode` | `meta-learning` only |
| Weights | [`stableai-org/LimiX-2`](https://huggingface.co/stableai-org/LimiX-2), file `LimiX-2.ckpt` |
| Paper | [arXiv:2609.17488](https://arxiv.org/abs/2609.17488) |

## Usage

```python
from tabtune import TabularPipeline

# Classification, zero-shot
pipe = TabularPipeline(
    model_name="LimiXv2",
    task_type="classification",
    tuning_strategy="inference",
    model_params={"softmax_temperature": 0.9},
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Classification, LoRA fine-tuning
pipe = TabularPipeline(
    model_name="LimiXv2",
    task_type="classification",
    tuning_strategy="peft",
    tuning_params={"epochs": 5, "peft_config": {"r": 8, "lora_alpha": 16}},
)
pipe.fit(X_train, y_train)

# Regression, full fine-tuning
reg = TabularPipeline(
    model_name="LimiXv2",
    task_type="regression",
    tuning_strategy="finetune",
    finetune_mode="meta-learning",
    tuning_params={"epochs": 5, "support_size": 512, "query_size": 256},
)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
```

LimiX-2 implements only `"meta-learning"`, for regression too; when `finetune_mode` is not given, the pipeline selects it.

## Model parameters (`model_params`)

Constructor parameters of `LimiXV2Classifier` / `LimiXV2Regressor`. TabTune resolves `device` and sets it after applying `model_params`.

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `None` | Local `LimiX-2.ckpt`. Must exist if given. |
| `repo_id` | `None` | Hugging Face repo. `None` tries `stableai-org/LimiX-2`, then `stable-ai/LimiX-2`. |
| `filename` | `"LimiX-2.ckpt"` | Checkpoint file in the repo. |
| `cache_dir` | `None` | Hugging Face cache directory. |
| `inference_config` | `None` | Path to a JSON inference config or a dict. `None` uses the bundled `cls_default_noretrieval_v2.json` / `reg_default_noretrieval_v2.json`. |
| `device` | `None` | `None`/`"auto"` selects CUDA when available. |
| `mix_precision` | `True` | Mixed-precision inference. |
| `softmax_temperature` | `0.9` | Softmax temperature. |
| `seed` | `0` | Seed. |
| `test_batch_size` | `16384` | Query rows per inference batch. |
| `deterministic` | `False` | Deterministic inference. |
| `gpu_ids` | `None` | GPUs for upstream's multi-GPU pipeline pool. |
| `enable_preprocess_parallel` | `True` | Parallel preprocessing. |
| `preprocess_num_jobs` | `16` | Preprocessing workers. |

## Fine-tuning parameters (`tuning_params`)

Both `finetune` and `peft` run the validated episodic loop in `tabtune.TuningManager.episodic_finetune` (model defaults in `tabtune.models.limixv2.tabtune_support.FINETUNE_DEFAULTS`). Before any training step the loop holds out a validation split of the training rows (stratified for classification), scores the **pretrained** weights on it through the model's real inference path (full ensemble, full preprocessing, held-out rows removed from the context), and then trains on episodes drawn from the training part only. After every epoch it re-scores the split, keeps the best weights seen (epoch 0 = pretrained) and stops early when nothing improves for `patience` epochs. At the end the best weights and the full training context are restored, so a run that does not beat zero-shot inference leaves the model exactly as it was. The outcome is on `pipe.model.finetune_report_` (`.summary()`, `.history`, `.kept`, `.baseline`, `.best`).

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `10` | Maximum passes of the episode loop. |
| `steps_per_epoch` | `16` | Optimiser steps per epoch. Alias: `episodes_per_epoch`. |
| `episodes_per_step` | `1` | Episodes accumulated into one optimiser step. Alias: `accumulation_steps`. |
| `learning_rate` | `None` | Peak AdamW learning rate; `None` picks `1e-5` for full fine-tuning and `1e-4` for LoRA. Alias: `lr`. |
| `weight_decay` | `0.0` | AdamW weight decay. |
| `support_size` | `None` | Context rows per episode. `None` uses as many training-part rows as the query leaves, capped by `max_support_size`, so episodes resemble the inference context. Alias: `context_size`. |
| `max_support_size` | `1024` | Cap on the support rows per episode (memory bound). |
| `query_size` | `256` | Query rows per episode (at most a quarter of the training part). |
| `grad_clip` | `1.0` | Gradient-norm clip; `0` disables it. Alias: `grad_clip_value`. |
| `warmup_fraction` | `0.1` | Share of optimiser steps spent on linear warmup; the rest decays with a cosine to `min_lr_fraction` of the peak. |
| `min_lr_fraction` | `0.1` | Floor of the cosine decay. |
| `validation_fraction` | `0.2` | Share of training rows held out for model selection. `0` disables validation (not recommended). Alias: `val_fraction`. |
| `min_validation_rows` | `32` | Below this many held-out rows the loop trains without validation and keeps the final weights, with a warning. |
| `patience` | `3` | Epochs without a validation improvement before stopping. |
| `min_delta` | `1e-3` | Relative improvement that counts as better. |
| `eval_every` | `1` | Validate every this many epochs. |
| `keep_best` | `True` | Restore the best validation weights (pretrained ones included). `False` keeps the final weights. |
| `l2sp` | `0.0` | L2 penalty towards the pretrained weights (L2-SP); for LoRA this is weight decay on the adapters. |
| `seed` | `0` | Split and episode sampling seed. Alias: `random_state`. |
| `amp` | `True` | bfloat16 autocast on CUDA devices that support it. |
| `log_every` | `1` | Log interval in epochs. |
| `peft_config` | `None` | LoRA settings: `r` (8), `lora_alpha` (16), `lora_dropout` (0.05), `target_modules`. |

Each episode samples one ensemble pipeline, preprocesses the support/query split exactly as `predict` does and trains through the model's forward pass. Losses: cross-entropy on the query rows for classification, cross-entropy over the model's regression bucket grid for regression. The validation metric is log-loss (classification) or RMSE in target-standard-deviation units (regression), computed with `LimiXPredictor.predict` on the held-out rows. Support sets are stratified so every class is in the context.

LoRA targets the attention and gated-MLP projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`); `scale_linear` is excluded.

Before the first validation pass, frozen-model reuse is disabled so predictions use the trained weights. If upstream's multi-GPU pipeline pool is active (`gpu_ids`), TabTune closes it and runs the fine-tuned model on a single device, with a `RuntimeWarning`, because pool workers reload weights from the checkpoint file.

```python
pipe = TabularPipeline(model_name="LimiXv2", task_type="classification", tuning_strategy="peft",
                       tuning_params={"epochs": 10, "patience": 3, "peft_config": {"r": 8}})
pipe.fit(X_train, y_train)
print(pipe.model.finetune_report_.summary())
# [LimiXv2] lora fine-tuning: 6 epoch(s), stopped early, kept epoch 3 weights: validation logloss 0.4710 -> 0.4587 (-2.61%); ...
```

## Limits

- Classification supports 2 to 10 classes. The registry declares `max_classes=10` as a hard limit, and the classifier's `fit` raises `ValueError` outside that range.
- Categorical encoding, datetime expansion, missing values and per-member normalisation run inside the model's own pipeline. TabTune's `limixv2_special` preprocessor passes features through untouched; classification targets are label-encoded.

## License

- Weights: LimiX Non-Commercial License. The registry records `commercial_use_ok=False`, so `license_mode="commercial"` fails fast.
- Code: StableAI LimiX License v1.0, which is Apache-2.0 terms plus Section 10. Redistributors must show "Built with StableAI LimiX", and distributed fine-tuned or distilled models must have names starting with "LimiX". See `tabtune/models/limixv2/_upstream/LICENSE.txt` and [the upstream repository](https://github.com/limix-ldm-ai/LimiX).

TabTune's LimiX-2 integration is Built with StableAI LimiX.

## Weights

On the first `fit()` the checkpoint is fetched with `hf_hub_download`, trying `repo_id` if given and otherwise the two default repos in order. To run offline, point at a local file:

```python
model_params={"model_path": "/models/LimiX-2.ckpt"}
```
