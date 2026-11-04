# Validated Fine-Tuning

Fine-tuning an in-context model can make it worse. On the benchmark that shipped with the
first Kumo and LimiX-2 integrations (five OpenML datasets, log loss and RMSE), no
fine-tuned run beat zero-shot inference: full fine-tuning was 3 to 6 percent worse on
average, LoRA 1 to 2 percent worse. That loop ran a fixed number of episodes at a constant
learning rate and kept whatever weights it ended on. Nothing measured whether they were
better than where it started.

The validated driver in `tabtune.TuningManager.episodic_finetune` replaces it. Kumo
Tabular, LimiX-2 and TabAbax use it for both `finetune` and `peft`; the other models keep
their own loops (see [Tuning Strategies](tuning-strategies.md)). It is worth knowing even
if you only use those three, because the failure modes it guards against apply to every
in-context model.

## What the loop does

1. **Hold out a validation split.** 20 percent of the training rows, stratified for
   classification; singleton classes stay in the context. Below `min_validation_rows`
   (32) the loop trains blind and says so in the log.
2. **Score the pretrained weights first.** The split is scored through the real inference
   path: full ensemble, full preprocessing, held-out rows removed from the context. That
   number is epoch 0, and every later epoch has to beat it.
3. **Train on episodes that look like inference.** Support sets are stratified and as
   large as the training part allows (capped at 2048 rows for Kumo and TabAbax, 1024 for
   LimiX-2), because a model that will see your whole table as context at prediction time
   should not be trained on 256-row contexts. AdamW, linear warmup over the first 10
   percent of steps, cosine decay to 10 percent of the peak, gradient clipping, optional
   accumulation (`episodes_per_step`) and optional L2-SP towards the pretrained weights.
4. **Re-score after every epoch.** The best weights are snapshotted; the run stops after
   `patience` (3) epochs without a relative improvement of `min_delta` (0.1 percent).
5. **Restore the best weights and the full context.** When no epoch beat epoch 0, the
   estimator is left on the pretrained weights. On the validation split, the model the
   pipeline predicts with is never worse than zero-shot.

The outcome is on the estimator as `finetune_report_`:

```python
pipe = TabularPipeline("Kumo", task_type="classification", tuning_strategy="peft",
                       tuning_params={"epochs": 10, "patience": 3})
pipe.fit(X_train, y_train)
report = pipe.model.finetune_report_
print(report.summary())
# epochs_run=6 best_epoch=4 baseline=0.2311 best=0.2104 (-9.0%) kept=finetuned
print(report.history)          # per-epoch validation scores
```

`kept` is `"finetuned"` or `"pretrained"`. If you see `"pretrained"` often, the honest
reading is that zero-shot is already the right answer for that dataset, not that something
failed.

## Parameters

All three models accept these in `tuning_params`. Model-specific defaults are on each
model's page.

| Parameter | Default | Meaning |
|---|---|---|
| `epochs` | 10 | Maximum passes of the episode loop. |
| `steps_per_epoch` | 30 | Optimiser steps per epoch. Alias `episodes_per_epoch`. |
| `episodes_per_step` | 2 | Episodes accumulated into one optimiser step. Alias `accumulation_steps`. |
| `learning_rate` | `None` | `1e-5` for full fine-tuning, `1e-4` for LoRA. Alias `lr`. |
| `weight_decay` | 0.0 | AdamW weight decay. |
| `support_size` | `None` | Context rows per episode. `None` means as many as the training part allows, capped by `max_support_size`. Alias `context_size`. |
| `max_support_size` | model-specific | Cap on support rows per episode. |
| `query_size` | 256 | Query rows per episode, at most a quarter of the training part. |
| `grad_clip` | 1.0 | Gradient-norm clip. Alias `grad_clip_value`. |
| `warmup_fraction` | 0.1 | Share of steps spent in linear warmup. |
| `min_lr_fraction` | 0.1 | Cosine floor as a fraction of the peak learning rate. |
| `validation_fraction` | 0.2 | Held-out share for selection. `0` disables validation. Alias `val_fraction`. |
| `min_validation_rows` | 32 | Below this the loop trains without validation and warns. |
| `patience`, `min_delta` | 3, 1e-3 | Early stopping. |
| `eval_every` | 1 | Epochs between validation passes. |
| `keep_best` | `True` | Restore the best validation weights, pretrained included. |
| `l2sp` | 0.0 | L2 penalty towards the pretrained weights. |
| `seed` | 0 | Split and episode sampling seed. Alias `random_state`. |
| `amp` | `True` | bfloat16 autocast on CUDA devices that support it. |
| `peft_config` | `None` | LoRA settings for `peft`: `r`, `lora_alpha`, `lora_dropout`, `target_modules`. |

The old blind behaviour is `validation_fraction=0, warmup_fraction=0, min_lr_fraction=1.0,
episodes_per_step=1, support_size=256, query_size=128`, which is useful for an A/B.

## Measuring it on your hardware

```bash
python examples/22_new_models_finetune_effect.py --device cuda --verbose
python examples/22_new_models_finetune_effect.py --report-only
```

The script fits inference, `finetune` and `peft` for Kumo, LimiX-2 and TabAbax on a set
of OpenML datasets and writes `fits.csv` with `kept`, `best_epoch`, `epochs_run`,
`ft_val_baseline` and `ft_val_best` per fit. `--no-validation` runs the blind loop on the
same splits.

## Costs

Validation adds one inference pass over the held-out rows per epoch. That is well under a
second for Kumo and TabAbax on a GPU, and the dominant cost of an epoch for LimiX-2,
whose ensemble preprocessing runs on CPU. Full fine-tuning keeps a CPU copy of the trainable
parameters for the snapshot (1.6 GB for LimiX-2); LoRA keeps only the adapters. `l2sp`
keeps a second copy on the device and is off by default.

## Why the other models' loops differ

TabPFN v2.6, v3 and v3.5 have a `native` mode that runs Prior Labs' own fine-tuning code,
which has its own validation. TabICL, Orion-MSP, Orion-BiX, Mitra, TabDPT, TabFM, iLTM and
EXAONE run TabTune's original episodic loop, which is unvalidated: it trains for the
requested epochs and keeps the result. For those models, hold out a test set, compare
against an `inference` pipeline on it, and keep the better one. The
[Leaderboard](leaderboard.md) does that comparison in one call.
