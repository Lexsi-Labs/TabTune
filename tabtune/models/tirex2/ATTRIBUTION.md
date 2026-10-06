# TiRex-2: third-party attribution

The model code in this directory is vendored from TiRex-2, the xLSTM-based multivariate time
series foundation model by NXAI. It needs PyTorch, NumPy, PyYAML and, for downloads only,
`huggingface_hub`. It does not import the `tirex-2` package or its dependencies `xlstm`,
`flashrnn` and `mlstm_kernels`.

- **License:** Apache License, Version 2.0. `LICENSE` and `NOTICE` are copied verbatim from the
  upstream repository.
- **Paper:** Patrick Podest, Marco Pichler, Elias Bürger, Levente Zólyomi, Bernhard Voggenberger,
  Wilhelm Berghammer, Daniel Klotz, Sebastian Böck, Günter Klambauer, Sepp Hochreiter.
  *TiRex-2: Generalizing TiRex to Multivariate Data and Streaming.* arXiv:2607.01204, 2026.
- **Pretrained weights:** `NX-AI/TiRex-2` on Hugging Face. They are not part of this directory;
  check the model card for their license.

## Sources

- **tirex-2 0.3.0**, https://github.com/NX-AI/tirex-2 (`src/tirex2/`). Every file keeps its
  upstream header. Not vendored: `demo.py`, `plotting.py`, `api_adapter/dataframe_adapter.py`,
  `api_adapter/gluon.py` and `api_adapter/standard_adapter.py`.
- **xlstm 2.0.6** (Apache-2.0), https://github.com/NX-AI/xlstm. Copyright (c) NXAI GmbH and its
  affiliates 2024; authors Maximilian Beck, Korbinian Pöppel, Andreas Auer. The classes TiRex-2
  uses from `xlstm/components/{conv,init,linear_headwise}.py` and
  `xlstm/xlstm_large/{components,model,utils}.py` are ported into
  `model/component/xlstm_layers.py`. `ParameterProxy` is replaced by plain parameters with the
  same names.
- **flashrnn 1.0.8** (Apache-2.0), https://github.com/NX-AI/flashrnn. Copyright 2024 NXAI GmbH,
  Korbinian Poeppel. The recurrence of the pure-PyTorch `vanilla` backend
  (`flashrnn/vanilla/{__init__,slstm}.py`) and the shape handling of `flashrnn/flashrnn.py` are
  ported into `model/component/slstm_kernel.py`. Its `NOTICE` is identical to `NOTICE` here.
- **mLSTM kernel.** `model/component/mlstm_kernel.py` is an independent implementation of the
  stabilised chunkwise mLSTM, written from the published equations (arXiv:2405.04517 and
  arXiv:2503.14376). No code from `mlstm_kernels` (NXAI Community License) was copied or adapted.
  It follows the semantics of the `chunkwise--native_autograd` kernel that TiRex-2 uses on CPU.

## Changes

1. **Imports.** Every `xlstm`, `flashrnn` and `mlstm_kernels` import points at
   `xlstm_layers.py`, `slstm_kernel.py` or `mlstm_kernel.py`. The mLSTM backend configuration in
   `mlstm_block.py` keeps only `chunk_size` and `eps`, since the kernel is the same on every
   device.
2. **`flashrnn_slstm.py`.** `FlashRNNLayerConfig` is a plain dataclass. The recurrent kernel and
   bias are plain parameters with the upstream names (`_recurrent_kernel_`, its alias
   `recurrent_kernel`, `_bias_`), so upstream checkpoints load unchanged. The unused streaming
   `step`, `zero_state` and `get_state` methods and the backend selection are removed.
3. **No FlexAttention.** `attention_block.py` always uses dense `scaled_dot_product_attention`
   with the same mask. `use_flex_attention` in a checkpoint config is accepted and ignored.
4. **`load_model`.** The default device is `"cpu"` and `"cuda:N"` is accepted; the
   `use_flex_attention` argument is removed. `huggingface_hub` is imported only for downloads.
   `compile=True` compiles the mLSTM layers, sLSTM layers and residual blocks.
5. **`api_adapter/forecast.py`.** Only `ForecastModel.forecast` with `output_type` `"torch"` or
   `"numpy"` is kept, with its batching and out-of-memory back-off; `forecast_gluon`,
   `forecast_df` and `forecast_fev` are removed. `ForecastModel.embed` is new.
6. **`model/tirex2.py`.** `forward` is split into `_encode` and the output head without numeric
   change. `predict` runs under `torch.no_grad()` unless `preserve_grad=True`, which fine-tuning
   uses. `embed` is new. `act_func` must name a `torch.nn.Module` class.
7. **`residual_block.py`.** The class-level `@torch.compile` decorator is removed;
   `load_model(compile=True)` compiles on request.
8. **Docstrings** import from `tabtune.models.tirex2` instead of `tirex2`.

On CPU the vendored model reproduces tirex-2 0.3.0 forecasts to within 4e-5 absolute in float32.
On CUDA, upstream runs the mLSTM in bfloat16 Triton kernels and the sLSTM in FlashRNN's CUDA
kernel; this code runs the same float32 PyTorch on every device, so its CUDA forecasts follow
upstream's CPU results.
