# Time-MoE — Third-Party Attribution

The following files in this directory are **vendored** from the Time-MoE
repository and are licensed under the **Apache License, Version 2.0** (full
text in `LICENSE`, copied verbatim; upstream has no `NOTICE` file):

| File | Upstream path | Upstream sha256 | Changes |
|---|---|---|---|
| `configuration_time_moe.py` | `time_moe/models/configuration_time_moe.py` | `d9b2e91a18fcc79753af7504a93382d43fa87d168318db01eb8431ef8d1b1125` | header comment only |
| `modeling_time_moe.py` | `time_moe/models/modeling_time_moe.py` | `cc012e5709211350ac53e7315d5f9469f043f20e001e721bd3b96c2cc586840c` | **Modified** (see below) |
| `LICENSE` | `LICENSE` | `7cd9951c712ed85dc4e03d25fff036a925ee7d1a0fceb5aa42a7e8788b40eac9` | verbatim |

**Upstream project:** https://github.com/Time-MoE/Time-MoE
**Commit:** `915bfda4c78a544d62a2bec6ab22948423059236` ("Support transformers == 4.57.0", 2026-03-22)
**Paper:** Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin.
*Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts.*
ICLR 2025 (Spotlight). https://arxiv.org/abs/2409.16040
**Pretrained weights:** `Maple728/TimeMoE-50M` and `Maple728/TimeMoE-200M` on Hugging Face.
The weight license was not verifiable from the environment this was written in
(Hugging Face unreachable); the TabTune registry records it as unverified.

The upstream README states "This project is licensed under the Apache-2.0
License". Its `LICENSE` file begins with the line "Copyright (c) 2023
Salesforce, Inc."; it is kept verbatim.

## Changes to `modeling_time_moe.py`

Each change is listed in the file's header comment.

1. `TimeMoeForPrediction` no longer inherits `TSGenerationMixin`, and
   `ts_generation_mixin.py` is not vendored. The mixin overrides
   `GenerationMixin._sample` and imports generation internals
   (`validate_stopping_criteria`, `EosTokenCriteria`) that have moved between
   transformers releases; upstream has needed a patch for each release.
   TabTune's adapter decodes with its own loop over `forward` that reproduces
   the mixin's semantics: at each step the largest output head no longer than
   the remaining horizon predicts from the last position, and its output is
   appended to the input.
2. The optional `flash_attn` import catches `ImportError` rather than using a
   bare `except:`.
3. `TimeMoeDecoderLayer` falls back to the eager attention class when
   `config._attn_implementation` is not one Time-MoE implements. Recent
   transformers versions may set it to `None` or `"sdpa"`, which used to raise
   `KeyError`.
4. `TimeMoeModel.forward` uses `input_ids.unsqueeze(-1)` instead of the
   in-place `unsqueeze_`, so the caller's tensor is not reshaped.

5. `TimeMoeForPrediction.prepare_inputs_for_generation` and `_reorder_cache`
   are removed. They only serve `generate`, which needs the mixin; with them
   and without it, transformers >= 4.50 warns at every load that the model
   cannot call `generate`.

6. `calc_ar_loss` calls `torch.nn.functional.huber_loss(..., reduction="none",
   delta=2.0)` directly, and the `self.loss_function = nn.HuberLoss(...)`
   assignment in `__init__` is removed. In recent transformers releases
   `PreTrainedModel.loss_function` is a property with a setter, but
   `nn.Module.__setattr__` stores a module value as a child module without
   calling the setter. The getter then returns transformers' default causal-LM
   loss, and `forward(labels=...)` fails with `TypeError: ForCausalLMLoss()
   missing 1 required positional argument: 'vocab_size'`. The Huber loss, its
   delta and the reduction are unchanged; `HuberLoss` has no parameters, so
   checkpoints are unaffected.

No model, attention, expert-routing or output-head computation was changed.

## Not vendored

- `ts_generation_mixin.py` (see above).
- `time_moe/datasets`, `time_moe/trainer`, `time_moe/runner.py`, `main.py`,
  `run_eval.py`: training and evaluation scripts. TabTune fine-tunes through
  its own loop, with the same Huber objective and auxiliary load-balancing
  loss, which the `forward` method computes.

`__init__.py` and this file are original TabTune code.
