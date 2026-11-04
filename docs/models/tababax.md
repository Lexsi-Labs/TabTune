# TabAbax

TabAbax is BaiRong's in-context tabular model, a research derivative of TabICLv2. It replaces TabICL's linear input mapping with learnable Fourier feature embeddings, keeps every feature token and the CLS tokens through the first row interaction, and adds a second column and row interaction round before the in-context head. The default estimator runs eight dataset views. TabTune vendors the TabAbax source tree in `tabtune/models/tababax` with its absolute imports rewritten; inference goes through upstream's estimators unchanged, and fine-tuning and LoRA are TabTune additions in `tabtune_support.py`.

| | |
|---|---|
| `model_name` | `"TabAbax"` (aliases `"Tab-Abax"`, `"TabAbax-v1"`, `"BaiRong-TabAbax"`) |
| Classification strategies | `inference`, `finetune`, `peft` |
| Regression strategies | `inference`, `finetune`, `peft` |
| `finetune_mode` | `meta-learning` only; the pipeline selects it |
| Weights | [`tababax-ai/TabAbax`](https://huggingface.co/tababax-ai/TabAbax): `tababax-classifier.ckpt`, `tababax-regressor.ckpt` |
| Report | [arXiv:2611.00562](https://arxiv.org/abs/2611.00562) |

## Usage

```python
from tabtune import TabularPipeline

pipe = TabularPipeline(model_name="TabAbax", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)

pipe = TabularPipeline(
    model_name="TabAbax",
    task_type="regression",
    tuning_strategy="peft",
    tuning_params={"epochs": 10, "peft_config": {"r": 8, "lora_alpha": 16}},
)
pipe.fit(X_train, y_train)
print(pipe.model.finetune_report_.summary())
```

## Model parameters (`model_params`)

Upstream's `TabAbaxClassifier` / `TabAbaxRegressor` parameters. TabTune resolves `device`.

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `8` | Dataset views in the ensemble. `1` is a single forward pass. |
| `batch_size` | `1` | Ensemble members forwarded together. `None` sends all of them at once. |
| `model_path` | `None` | Local checkpoint. `None` downloads `checkpoint_version` from `tababax-ai/TabAbax` into the Hugging Face cache. |
| `allow_auto_download` | `True` | `False` refuses a download. |
| `checkpoint_version` | `tababax-classifier.ckpt` / `tababax-regressor.ckpt` | File on the Hub. |
| `checkpoint_revision` | `None` | Commit, tag or branch. Pin it when a benchmark must keep loading the bytes it measured. |
| `norm_methods` | `["none", "power"]` | Normalisations spread across the views. |
| `feat_shuffle_method`, `class_shuffle_method` | `"latin"`, `"shift"` | Feature and class permutations per view (class shift is classification only). |
| `outlier_threshold` | `4.0` | Z-score clipping threshold. |
| `softmax_temperature`, `average_logits` | `0.9`, `True` | Classification output. |
| `support_many_classes` | `True` | Mixed-radix ensembling and hierarchical classification above ten classes. |
| `use_amp`, `use_fa3` | `"auto"`, `False` | Mixed precision; FlashAttention-3 only when the runtime provides it. |
| `offload_mode`, `disk_offload_dir` | `"auto"`, `None` | Places the column embedding in GPU, CPU or disk memory by what is free. |
| `random_state` | `42` | Ensemble shuffling. |

## Fine-tuning parameters (`tuning_params`)

`finetune` and `peft` run the validated episodic loop in `tabtune.TuningManager.episodic_finetune`, with TabAbax defaults in `tabtune.models.tababax.tabtune_support.FINETUNE_DEFAULTS`. The loop holds out a validation split (stratified for classification), scores the pretrained weights on it through `predict_proba` / `predict` with the held-out rows removed from the context, trains on episodes from the training part only, re-scores after every epoch, stops early and restores the best weights seen, the pretrained ones included. The outcome is on `pipe.model.finetune_report_`.

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `10` | Maximum passes. |
| `steps_per_epoch` | `30` | Optimiser steps per epoch. |
| `episodes_per_step` | `2` | Episodes accumulated per step. |
| `learning_rate` | `None` | `1e-5` full, `1e-4` LoRA. |
| `support_size`, `max_support_size` | `None`, `2048` | Context rows per episode, capped. |
| `query_size` | `256` | Query rows per episode. |
| `validation_fraction`, `min_validation_rows` | `0.2`, `32` | Model selection split. |
| `patience`, `min_delta` | `3`, `1e-3` | Early stopping. |
| `warmup_fraction`, `min_lr_fraction` | `0.1`, `0.1` | Linear warmup, cosine decay. |
| `l2sp` | `0.0` | Penalty towards the pretrained weights. |
| `peft_config` | `None` | `r`, `lora_alpha`, `lora_dropout`, `target_modules`. |

Each episode takes one fitted ensemble view (normalisation, feature permutation and class shift), a support/query split of the training part, and the model's own training branch (`TabAbax._train_forward`). The loss is cross-entropy over the ten class slots for classification and the pinball loss at the regression head's quantile levels for regression. Fine-tuning is limited to ten classes; inference handles more.

LoRA wraps the feed-forward layers (`linear1`, `linear2`), the attention output projections and the second column interaction's projection (`out_proj`), the Fourier embedder's projection and the ICL decoder. `attn.out_proj` and the embedder projection are read as tensors by functional calls, so their adapters live in `weight`. The class-slot `y_encoder` and the QASSMax gates are excluded.

A fine-tuned estimator keeps its weights across later `fit()` calls and through `save()` / `load()`; upstream's pickling, which reloads the checkpoint, is bypassed for it.

## Limits

- Ten native class slots for fine-tuning.
- Large tables rely on upstream's automatic CPU and disk offload.
- The released checkpoints were trained from a TabICLv2-derived implementation; the training and prior code are not public.

## License

Weights: CC-BY-NC-4.0, not licensed for commercial use. Code: TabICL-derived modules BSD-3-Clause, TabR and rotary-embedding portions MIT, PyTabKit portions and TabAbax packaging Apache-2.0. See `tabtune/models/tababax/LICENSE`, `NOTICE`, `PROVENANCE.md` and `LICENSES/`. The registry records `commercial_use_ok=False` and names TabICLv2, Mitra and Kumo Tabular as commercial alternatives.
