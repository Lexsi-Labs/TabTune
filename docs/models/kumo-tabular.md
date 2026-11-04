# NVIDIA Kumo Tabular

Kumo Tabular is NVIDIA's in-context tabular model. An interleaved column/row encoder with induced set attention and Fourier cell embeddings feeds an in-context learning transformer with length-aware attention temperature. It comes in three sizes (small, medium, large) spanning 28M to 215M parameters, and is pretrained only on synthetic tables with contexts of up to 60,000 rows and up to 100 columns. NVIDIA reports it ranking first on TabArena with an ELO of 1950. TabTune vendors the parts of [NVIDIA structured-data-models](https://github.com/NVIDIA/structured-data-models) it needs in `tabtune/models/kumo/_sdm`; inference goes through upstream's `ICLModel.forward` unchanged, and fine-tuning follows upstream's `examples/tabular/finetune.py`.

| | |
|---|---|
| `model_name` | `"KumoTabular"` (aliases: `"Kumo-Tabular"`, `"Kumo"`, `"NVIDIA-Kumo-Tabular"`, `"KumoTab"`) |
| Classification strategies | `inference`, `finetune`, `peft` |
| Regression strategies | `inference`, `finetune`, `peft` |
| `finetune_mode` | `meta-learning` only |
| Weights | [`nvidia/Kumo-Tabular`](https://huggingface.co/nvidia/Kumo-Tabular), revision `v1.0.0` |

## Usage

```python
from tabtune import TabularPipeline

# Classification, zero-shot with the medium checkpoint
pipe = TabularPipeline(
    model_name="KumoTabular",
    task_type="classification",
    tuning_strategy="inference",
    model_params={"size": "medium"},
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Classification, LoRA fine-tuning
pipe = TabularPipeline(
    model_name="KumoTabular",
    task_type="classification",
    tuning_strategy="peft",
    tuning_params={"epochs": 5, "peft_config": {"r": 8, "lora_alpha": 16}},
)
pipe.fit(X_train, y_train)

# Regression, full fine-tuning
reg = TabularPipeline(
    model_name="KumoTabular",
    task_type="regression",
    tuning_strategy="finetune",
    finetune_mode="meta-learning",
    tuning_params={"epochs": 5, "support_size": 256, "query_size": 128},
)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)                    # mean of 999 quantiles
q = reg.model.predict_quantiles(X_test)        # (rows, 999), levels 0.001 ... 0.999
```

Kumo Tabular implements only `"meta-learning"`, for regression too; when `finetune_mode` is not given, the pipeline selects it.

## Model parameters (`model_params`)

Constructor parameters of `KumoTabularClassifier` / `KumoTabularRegressor`. TabTune resolves `device` and sets it after applying `model_params`.

| Parameter | Default | Description |
|---|---|---|
| `size` | `"large"` | `"small"`, `"medium"` or `"large"`. |
| `num_estimators` | `None` | Ensemble size. `None` uses 8 for small and medium, 16 for large (the counts behind NVIDIA's reported results). |
| `estimator_batch_size` | `"auto"` | Estimators per forward. `"auto"`: on CUDA without `max_context_size`, 1 when the table exceeds 3,000 rows or 50,000 cells, otherwise all at once; 1 on CPU or with `max_context_size`. |
| `query_batch_size` | `8192` | Query rows per forward; `None` sends all rows at once. |
| `max_context_size` | `None` | If set and the training set is larger, each estimator uses its own random context subsample of this many rows. |
| `model_path` | `None` | Local checkpoint file, or a snapshot directory containing `{size}/classifier.pt` / `{size}/regressor.pt`. |
| `repo_id` | `"nvidia/Kumo-Tabular"` | Hugging Face repo. |
| `revision` | `"v1.0.0"` | Repo revision. |
| `cache_dir` | `None` | Download cache directory. |
| `device` | `None` | `None`/`"auto"` selects CUDA when available. |
| `autocast` | `True` | float16 autocast on CUDA. |
| `low_cardinality` | `"infer"` | Passed to column-type inference. |
| `seed` | `0` | Seed for ensemble members and context subsampling. |

## Fine-tuning parameters (`tuning_params`)

Both `finetune` and `peft` run the validated episodic loop in `tabtune.TuningManager.episodic_finetune` (model defaults in `tabtune.models.kumo.tabtune_support.FINETUNE_DEFAULTS`). Before any training step the loop holds out a validation split of the training rows (stratified for classification), scores the **pretrained** weights on it through the model's real inference path (full ensemble, full preprocessing, held-out rows removed from the context), and then trains on episodes drawn from the training part only. After every epoch it re-scores the split, keeps the best weights seen (epoch 0 = pretrained) and stops early when nothing improves for `patience` epochs. At the end the best weights and the full training context are restored, so a run that does not beat zero-shot inference leaves the model exactly as it was. The outcome is on `pipe.model.finetune_report_` (`.summary()`, `.history`, `.kept`, `.baseline`, `.best`).

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `10` | Maximum passes of the episode loop. |
| `steps_per_epoch` | `30` | Optimiser steps per epoch. Alias: `episodes_per_epoch`. |
| `episodes_per_step` | `2` | Episodes accumulated into one optimiser step. Alias: `accumulation_steps`. |
| `learning_rate` | `None` | Peak AdamW learning rate; `None` picks `1e-5` for full fine-tuning and `1e-4` for LoRA. Alias: `lr`. |
| `weight_decay` | `0.0` | AdamW weight decay. |
| `support_size` | `None` | Context rows per episode. `None` uses as many training-part rows as the query leaves, capped by `max_support_size`, so episodes resemble the inference context. Alias: `context_size`. |
| `max_support_size` | `2048` | Cap on the support rows per episode (memory bound). |
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

Training uses the default preprocessing recipe without target sign flipping or output post-processing. Losses: cross-entropy for classification, pinball loss over the 999 quantile levels in target-standard-deviation units for regression. The validation metric is log-loss (classification) or RMSE in target-standard-deviation units (regression). Support sets are stratified so every class is in the context. Episodes with a non-finite loss are skipped. Fine-tuning is limited to ten classes (the head's capacity); inference handles more through error-correcting output codes.

LoRA targets the attention and MLP projections of the column, row and ICL blocks (`qkv_lin`, `out_lin`, `mlp`). `cell_embedding`, `y_lin` and `query_scaling` are excluded.

```python
pipe = TabularPipeline(model_name="KumoTabular", task_type="classification", tuning_strategy="peft",
                       tuning_params={"epochs": 10, "patience": 3, "peft_config": {"r": 8}})
pipe.fit(X_train, y_train)
print(pipe.model.finetune_report_.summary())
# [KumoTabular] lora fine-tuning: 6 epoch(s), stopped early, kept epoch 3 weights: validation logloss 0.4710 -> 0.4587 (-2.61%); ...
```

## Limits

- Ten classes per forward pass; more are handled with error-correcting output codes. At least two classes are required.
- The preprocessing recipe keeps the first 500 columns after shuffling.
- Pretraining covered contexts up to 60,000 rows and 100 columns; accuracy may degrade far beyond that. `max_context_size` bounds the context per estimator.
- Only numerical and categorical columns are embedded. Datetime columns are converted to numbers and object columns to strings before type inference; columns of unsupported type are dropped. TabTune's `kumo_special` preprocessor passes features through untouched.
- Regression predicts 999 quantiles; `predict()` returns their mean.

## License

- Weights: OpenMDW License Agreement v1.1, for commercial use. The registry records `commercial_use_ok=True`.
- Code: NVIDIA structured-data-models is Apache-2.0, and the TabFM-derived cell embedding is Apache-2.0 as well.

## Weights

On the first `fit()` the checkpoint `{size}/classifier.pt` or `{size}/regressor.pt` is downloaded from `repo_id` at `revision`. To run offline, pass a local file or a snapshot directory:

```python
model_params={"size": "large", "model_path": "/models/Kumo-Tabular"}   # uses large/classifier.pt
```
