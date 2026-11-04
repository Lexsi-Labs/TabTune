# FAQ

## What is TabTune, in one sentence?

One scikit-learn style pipeline for running and fine-tuning 24 tabular foundation models,
plus a matching pipeline for eleven time series foundation models, with the model code
vendored and the licences checked before anything downloads. See [Home](../index.md).

## Do I need a GPU?

Not for inference on the smaller models (TabPFN v2, TabICL, Orion-MSP, EXAONE,
Kumo small): a few thousand rows predict in seconds on CPU. Fine-tuning anything, or
inference with the large models on large tables, wants a GPU.

## Which Python version?

3.10 or newer. Toto 2 needs 3.12. See [Installation](../getting-started/installation.md).

## Do I have to install each model's package?

No. The model code is vendored under `tabtune/models/`; `pip install -e .` is the whole
install. Checkpoints download from the Hugging Face Hub on the first `fit()`.

## Does TabTune support regression?

Yes, on every model with a regression head: the TabPFN family, TabICL v2, Mitra and Mitra
v2, ConTextTab, TabDPT, LimiX and LimiX-2, TabFM, xRFM, iLTM, EXAONE (local checkpoint),
Causilo, TabLDM and Kumo. TabICL, Orion-MSP and Orion-BiX are classification
only. `list_models(task="regression")` is the live answer. See
[Regression](../user-guide/regression.md).

## `inference`, `finetune` or `peft`?

Start with `inference`; it is a real baseline for an in-context model. Move to `peft` when
you have a GPU and more than a few thousand rows, and to `finetune` when memory allows and
`peft` plateaued. For Kumo, LimiX-2 and TabAbax the fine-tuning loop validates itself
and keeps the pretrained weights if nothing beats them. For the others, hold out a test set
and compare. See [Tuning Strategies](../user-guide/tuning-strategies.md) and
[Validated Fine-Tuning](../user-guide/fine-tuning.md).

## Which models can I use commercially?

The weights decide. `list_models(commercial_ok=True)` returns them, and
`TabularPipeline(..., license_mode="commercial")` refuses the rest at construction time.
The table is on [Models](../models/overview.md).

## Why did fine-tuning make my model worse?

Because an in-context model is already adapted to your data at prediction time, and a
training loop that does not measure itself against that baseline can drift away from it.
The [Validated Fine-Tuning](../user-guide/fine-tuning.md) page explains the causes and the
fix. If `finetune_report_.kept` says `"pretrained"`, zero-shot won and that is the honest
result.

## How do I pick a model for my dataset?

Rows, columns, classes, text, licence, hardware, in that order; then a
[Leaderboard](../user-guide/leaderboard.md) over the survivors. The checklist is at the
bottom of [Models](../models/overview.md) and the long version is
[Model Selection](../user-guide/model-selection.md).

## My dataset has 50,000 rows and TabPFN warns. Is the result wrong?

No, it is a soft limit: TabTune passes `ignore_pretraining_limits=True`, so the model runs
and accuracy degrades past 10,000 rows. For big tables use TabPFN v3 or v3.5, TabICL v2,
Kumo or TabDPT, or subsample the context. `envelope_mode="error"` turns the warning into
an exception if you would rather be stopped.

## Can I use my own preprocessing?

Yes. `processor_params` overrides imputation, scaling and encoding;
[Custom Preprocessing](../advanced/custom-preprocessing.md) shows how to put your own
transformer in the chain. Models that preprocess internally (Kumo, LimiX-2, TabAbax,
TabPFN v3.5, Causilo, TabLDM) receive the raw frame regardless.

## Does the time series pipeline fine-tune?

All eleven foundation models: `finetune` for every weight, `peft` for LoRA adapters. The
tabular forecasters and the baselines are inference-only. See
[Time Series Fine-Tuning](../timeseries/fine-tuning.md).

## Where do I report a bug or ask a question?

[GitHub Issues](https://github.com/Lexsi-Labs/TabTune/issues) for bugs and requests,
[Discord](https://discord.gg/dSB62Q7A) for questions. [Contributing](contributing.md) has
the workflow, [Security](security.md) is for vulnerabilities.
