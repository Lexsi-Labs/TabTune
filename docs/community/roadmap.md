# Roadmap

TabTune does not keep a separate roadmap document. These are the gaps that are visible in
the code and the [Changelog](changelog.md), in one place. If you want to work on one,
check [Contributing](contributing.md) first and search the
[open issues](https://github.com/Lexsi-Labs/TabTune/issues): someone may be on it.

## Known gaps

- **The validated fine-tuning loop covers three models.** Kumo, LimiX-2 and TabAbax use
  it. TabICL, the Orion-MSP family, Orion-BiX, Mitra, TabDPT, TabFM, iLTM and EXAONE still
  run the older unvalidated episodic loop, which trains for the requested epochs and keeps
  the result whether or not it beat zero-shot. Porting them is mostly a matter of supplying
  an episode loss and a validation hook per model.
- **PEFT on TabPFN v2 and ConTextTab is experimental**, and TabPFN v2.6 has no LoRA target
  table at all.
- **EXAONE regression needs a locally supplied checkpoint**; LG AI Research publishes only
  the classification weights.
- **The tabular side has no command line.** `tabtune` has the `timeseries` group only.
- **Tabular models as forecasters are inference-only.** Fine-tuning a `TabularTS-*` model
  on series data means building the table yourself.
- **Conformal coverage on time series is approximate** and documented as such; a
  block-aware or adaptive conformal method would tighten it.
- **No published real-weight benchmark table on this site.** `examples/22` and
  `examples/timeseries/benchmark.py` produce one on your hardware; publishing one needs a
  fixed GPU and a fixed set of datasets.

## Recently shipped

See the [Changelog](changelog.md). Most recently: LimiX-2, Kumo Tabular and TabAbax on
the validated fine-tuning driver, and Timestral and T0 for time series; before that, the time series pipeline with nine vendored
models, and TabPFN v3.5, Causilo, TabLDM and Mitra v2.
