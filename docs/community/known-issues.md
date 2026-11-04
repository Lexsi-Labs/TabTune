# Known Issues

Found by reading the code and running the suites, not by trusting earlier docs. Organised
by how much each should worry you.

## Looks like it works, but does less than you think

- **The older episodic fine-tuning loop does not validate.** For TabICL, Orion-MSP,
  Orion-BiX, Mitra, TabDPT, TabFM, iLTM and EXAONE, `finetune` and `peft` train for the
  requested epochs and keep the final weights. On the Kumo and LimiX-2 benchmark that loop
  made every fine-tuned run worse than zero-shot. Those two and TabAbax now use the
  [validated driver](../user-guide/fine-tuning.md); for the others, compare against an
  `inference` pipeline on a held-out split before trusting a fine-tuned one.
- **EXAONE's `peft` is a full fine-tune.** Its projections are raw `nn.Parameter` tensors
  applied through `F.linear`, so the LoRA injector wraps zero adapters and the run proceeds
  with every weight trainable. The log says so; the memory profile is `finetune`'s.
- **xRFM's `peft` is not LoRA.** It is low-rank adaptation of the learned feature matrix.
  Different thing, same strategy name.
- **PEFT on TabPFN v2 and ConTextTab is experimental.** Adapters attach, but training can
  be unstable. The registry flags it; use `finetune` if a run misbehaves.
- **TabAbax and LimiX-2 refuse to fine-tune above ten classes.** Inference handles more
  (hierarchical classification in TabAbax), fine-tuning is limited to the head's native class
  count. The error is explicit.
- **Kumo Tabular above ten classes uses ECOC**, one forward per codebook row, so inference
  cost grows with the class count.

## Setup and environment

- **TabPFN v2.6 and later, TabFM, ConTextTab, EXAONE, Causilo and LimiX-2 weights are
  non-commercial**, and TabLDM, TabDPT, TiRex, TiRex-2 and Time-MoE are unverified.
  `license_mode="commercial"` refuses the first group and warns on the second; it cannot
  read a licence you have negotiated separately, so use `license_mode="ignore"` in that
  case.
- **TabFM needs typeguard 2.x.** Its vendored estimator decorates functions with
  `jaxtyping.jaxtyped(typechecker=typeguard.typechecked)`, and typeguard 3 and 4 parse
  jaxtyping's shape strings as Python and raise `SyntaxError` at import. 0.2.2 pins
  `typeguard>=2.13.3,<3`; an environment that upgrades typeguard breaks TabFM.
- **Toto 2 needs Python 3.12** and its own extra (`tabtune[toto2]`). On 3.11 the extra
  installs only with `--ignore-requires-python`.
- **Time-MoE does not accept NaN in the history.** Impute first (the pipeline's
  `imputation` task with another model works) or pick another model; the registry records
  `native_missing=False`.
- **EXAONE regression needs a locally supplied checkpoint.** Only the classification
  weights are published.
- **`skrub` is a core dependency** since TabPFN v3.5 and is capped at `<0.11`; a newer skrub
  in the same environment breaks the v3.5 text and datetime preprocessing.
- **Long fine-tuning runs fragment CUDA memory.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  helps; see [Memory Optimization](../advanced/memory-optimization.md).

## Documentation

- The capability and licence tables on [Models](../models/overview.md) and
  [Time Series Models](../timeseries/models.md) are generated from the registry at the time
  the docs were built. The registry is the source of truth; query it if a number matters.
- The example notebooks on [Notebooks](../tutorials/notebooks.md) predate the time series
  pipeline and the validated fine-tuning loop; the `examples/` scripts in the repository are
  current.

If you find something that belongs here, open an
[issue](https://github.com/Lexsi-Labs/TabTune/issues).
