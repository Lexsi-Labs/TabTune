# TiRex: Third-Party Attribution

Built with technology from NXAI

This directory vendors the model code of TiRex (v1), the pre-trained xLSTM time series forecasting model
developed by NXAI GmbH. It is distributed under the NXAI Community License Agreement, copied verbatim in
`LICENSE`, and not under TabTune's own license.

- Source repository: https://github.com/NX-AI/tirex
- Version: tirex-ts 1.4.2
- Pretrained weights: https://huggingface.co/NX-AI/TiRex

This product includes materials developed at NXAI that are licensed under the NXAI Community License,
Copyright © NXAI GmbH, All Rights Reserved.

The same notice, required by section 1.b.iii of the license, is in the `NOTICE` file.

## Source

Every file keeps the upstream copyright header. The code follows the GitHub source tree of
tirex-ts 1.4.2, which adds the `full_rollout` and `dynamic_padding` forecast options to the PyPI
wheel and computes the sLSTM recurrent product with `torch.einsum`; the recurrence runs in bfloat16,
so outputs can differ from the wheel by rounding.

## Modifications

- Imports are relative. The package does not import `tirex`, `xlstm`, `flashrnn`, `mlstm_kernels` or
  `sklearn`; `huggingface_hub` is imported only when `load_model` downloads from the Hub.
- sLSTM: removed the CUDA backend (`xlstm` kernels) and the `backend` argument. Only the PyTorch cell
  remains.
- `TiRexZero`: added `forward`, the gradient-enabled forecast path that `_forecast_quantiles` now wraps in
  `torch.inference_mode`, and `embed`, which wraps `TiRexEmbedding`. Removed the unused logger.
- `TiRexEmbedding` takes a model instance instead of loading `NX-AI/TiRex` itself; removed its `device`
  and `compile` arguments.
- `load_model` / `PretrainedModel.from_pretrained`: accepts a local directory containing `model.ckpt`, a
  checkpoint file or a Hugging Face repo id. Checkpoints are always read with `torch.load(...,
  weights_only=True)` and `load_state_dict(strict=True)`, and the model is returned in eval mode. Removed
  the `backend`, `compile` and `ckp_kwargs` arguments and the model registry lookup by repo name; the
  default device is `cpu`.
- `ForecastModel.forecast` accepts tensors, arrays and lists of 1-D series and returns torch tensors.
  Removed frequency resampling, `output_type`, `yield_per_batch`, `forecast_gluon`, `forecast_hfdata` and
  `max_context_length`. The batching helpers of `standard_adapter.py` are merged into `forecast.py`.
- `util.py` keeps only `round_up_to_next_multiple_of`, `dataclass_from_dict`, `nanmax`, `nanmin`, `nanvar`
  and `nanstd`. Plotting, frequency resampling, FFT analysis, scikit-learn helpers and `EarlyStopping` were
  dropped.
- Not vendored: `models/trainer.py`, `models/classification`, `models/regression`, `models/base`,
  `api_adapter/gluon.py`, `api_adapter/hf_data.py`.
